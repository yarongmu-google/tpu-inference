# Benchmark recipe — chunk-prefill optimization A/B

How to measure the impact of the static-q-len PREFILL kernel routing on
real serving. Methodology mirrors the QC repo (`bm-infra`) so we can
graduate to that infra when we move to its standard models (Qwen3 family,
gpt-oss-20b).

## Prerequisites (one-time, on the TPU VM)

```bash
# vLLM source checkout (we read its benchmarks/sonnet.txt). The path
# below is what bm-infra uses; adjust to wherever your vllm is.
export VLLM_DIR=$HOME/vllm
export DOWNLOAD_DIR=$HOME/hf-cache  # or wherever you cache HF weights
export HF_TOKEN=hf_xxx              # only needed for gated models

# tpu-inference itself: be on rpa3 branch.
cd ~/tpu-inference
git fetch origin && git checkout rpa3 && git pull origin rpa3
```

## Cases (under `tools/benchmark/cases/`)

| Case file                              | Workload         | Targets the PREFILL opt? |
|----------------------------------------|------------------|--------------------------|
| `llama3_8b_v7x_balanced.env`           | sonnet 1024/1024 | Mild — mixed prefill+decode in steady state. |
| `llama3_8b_v7x_prefill_heavy.env`      | sonnet 1800/128  | **Yes** — prefill-dominated. The PREFILL kernel routing should win the most here. |

The `1800/128` shape comes from bm-infra's `accuracy_jax.csv`. The QC repo
doesn't currently run a prefill-heavy *perf* benchmark on v7x; we add it
here because it's the natural workload to stress the chunk-prefill
optimization. We can graduate to a QC-blessed shape when they add one.

## Running

```bash
tools/benchmark/run_benchmark.sh <case_file> [--result-tag TAG] [--rate RATE]
```

- `--result-tag TAG` controls the output dir name. Default is a timestamp.
- `--rate RATE` is the `--request-rate` passed to `vllm bench serve`.
  Default `inf` (max throughput, no limit).

Output lands in `tmp/bench_<case_name>_<tag>/` with:

- `meta.txt` — config + git commit + branch (handy for tracking what produced what)
- `vllm.log` — server stdout/stderr
- `bench.log` — `vllm bench serve` stdout/stderr (full percentile tables)
- `metrics.txt` — flat key=value list of the headline numbers

## A/B procedure: baseline vs chunk-prefill optimized

Both runs use the same `rpa3` branch — toggle the optimization via the
scheduler's `long_prefill_token_threshold` knob (which our runner reads).

### 1. Baseline (PREFILL routing OFF)

In the case file, leave `LONG_PREFILL_TOKEN_THRESHOLD` unset (so vLLM
auto-derives it from `max_model_len`), then disable our routing by
exporting `LONG_PREFILL_TOKEN_THRESHOLD=0` for the run:

```bash
LONG_PREFILL_TOKEN_THRESHOLD=0 \
  tools/benchmark/run_benchmark.sh \
    tools/benchmark/cases/llama3_8b_v7x_prefill_heavy.env \
    --result-tag baseline_K0
```

With `K=0`, our `_chunk_prefill_size` stays None, the runner produces
`[D, D, T]`, and the kernel skips its PREFILL pass — exactly today's
main-branch behavior.

### 2. Optimized (PREFILL routing ON, tuned blocks)

Pick a K from the tuning results (`tmp/tuning_llama3_8b_v7x_prefill_2026_05_03.md`).
For prefill-heavy with 1800-token prompts, K=512 or K=1024 are reasonable
(prompts produce ~3–4 chunks). Set the matching tuned block sizes via
`RPA_P_BLOCK_SIZES` and force the scheduler to that K:

```bash
# K=512, page=128 winners: bq_sz=256, bkv_sz=1024, bq_csz=128, bkv_csz=1024
RPA_P_BLOCK_SIZES="256,1024,128,1024" \
LONG_PREFILL_TOKEN_THRESHOLD=512 \
  tools/benchmark/run_benchmark.sh \
    tools/benchmark/cases/llama3_8b_v7x_prefill_heavy.env \
    --result-tag opt_K512
```

```bash
# K=1024, page=128 winners: bq_sz=256, bkv_sz=2048, bq_csz=256, bkv_csz=512
RPA_P_BLOCK_SIZES="256,2048,256,512" \
LONG_PREFILL_TOKEN_THRESHOLD=1024 \
  tools/benchmark/run_benchmark.sh \
    tools/benchmark/cases/llama3_8b_v7x_prefill_heavy.env \
    --result-tag opt_K1024
```

### 3. Compare

Each run produces `tmp/bench_<case>_<tag>/metrics.txt`. Diff them:

```bash
diff -u tmp/bench_llama3_8b_v7x_prefill_heavy_baseline_K0/metrics.txt \
        tmp/bench_llama3_8b_v7x_prefill_heavy_opt_K512/metrics.txt
```

Headline metrics to look at:
- `RequestThroughput` (req/s, max-throughput run)
- `MeanTTFT`, `P99TTFT` (time to first token — prefill latency)
- `MeanITL`, `P99ITL` (inter-token latency — decode tail)
- `OutputTokenThroughput`, `TotalTokenThroughput`

## Interpreting results

The chunk-prefill optimization should produce:
- **Higher request/token throughput** on prefill-heavy at max rate.
- **Lower TTFT** because each prefill chunk runs on a faster kernel.
- **Approximately unchanged ITL** — decode kernel is unaffected.
- **Smaller win on the balanced workload** (1024/1024) than on
  prefill-heavy (1800/128) because uniform-K chunks are a smaller
  fraction of total work.

If TTFT improves but throughput doesn't, that means the kernel is no
longer the bottleneck (suspect host overhead or DMA bandwidth). If neither
moves, the PREFILL bucket is probably never large enough — verify with the
runner's `request_distribution` debug output that we're actually filling
the bucket.

## Sweeping rate (for SLO-bounded throughput, like bm-infra does)

For an SLO-style metric ("max throughput where P99 E2EL ≤ 3.6s"), bm-infra
binary-searches `--request-rate`. Our script doesn't bake that in, but
manually:

```bash
for rate in inf 20 10 5 2; do
  tools/benchmark/run_benchmark.sh \
    tools/benchmark/cases/llama3_8b_v7x_prefill_heavy.env \
    --result-tag opt_K512_rate${rate} \
    --rate $rate
done
```

Then pick the highest rate where `P99E2EL` (in `metrics.txt`) is under your
target. Add the binary-search wrapper as a follow-up if we end up needing
it often.

## Adding more cases later

As we move to bm-infra's v7x models, copy a row from
`/path/to/bm-infra/cases/hourly_tt_v7.csv` into a new `.env` file:

```
Device,Model,MaxNumSeqs,MaxNumBatchedTokens,TensorParallelSize,MaxModelLen,Dataset,InputLen,OutputLen
tpu7x-2,Qwen/Qwen3-Coder-30B-A3B-Instruct,128,10275,1,10275,sonnet,1024,1024
```

…becomes `tools/benchmark/cases/qwen3_coder_30b_v7x_balanced.env` with
the same fields. Re-tune the kernel for the new model first
(different `num_q_heads` / `num_kv_heads` / `head_dim` →
different winning block sizes).

## Parameter Ownership: Tuning vs Sweeping

When optimizing for maximum throughput, it is critical to understand which parameters are responsible for hardware-level XLA shapes (Kernel Tuning) versus vLLM queuing logic (Python Scheduler Sweeps).

| Python / Internal Variable | Environment Variable | Action | Reason / Phase Responsibility |
| :--- | :--- | :--- | :--- |
| `p_block_sizes`<br>`d_block_sizes`<br>`m_block_sizes` | `RPA_P_BLOCK_SIZES`<br>`RPA_D_BLOCK_SIZES`<br>`RPA_M_BLOCK_SIZES` | **Tuned**<br>(Pinned in Sweep) | **Hardware Kernel Optimization:** The tuner systematically searches for the lowest-latency `[bq, bkv, bq_c, bkv_c]` memory tiles for the XLA loops. These winning values are then strictly pinned during the sweep. |
| `page_size` | `BLOCK_SIZE` | **Tuned**<br>(Pinned in Sweep) | **Hardware Kernel Optimization:** The KV Cache page size is a static dimension of the XLA tensor. The tuner evaluates different page sizes to find the best memory layout. The winner is pinned in the sweep. |
| `chunk_prefill_size` | `LONG_PREFILL_TOKEN_THRESHOLD` | **Tuned**<br>(Pinned in Sweep) | **Hardware Kernel Optimization:** Dictates the static sequence length (`q_len`) for the PREFILL attention loop. The tuner identifies the most efficient chunk size (K). The sweep pins this value. |
| `max_num_batched_tokens` | `MAX_NUM_BATCHED_TOKENS` | **Swept** | **Python Scheduler Optimization:** Dictates how many total tokens vLLM can pull from the queue per step. Swept to find the value that keeps the XLA kernel perfectly saturated without causing queue starvation or excessive chunking overhead. |
| `max_num_seqs` | `MAX_NUM_SEQS` | **Swept** | **Python Scheduler Optimization:** Dictates the maximum concurrency. Swept to find the concurrency limit that maximizes HBM utilization (filling the batch) without causing frequent KV-cache evictions or out-of-memory errors. |
| `max_model_len` | `MAX_MODEL_LEN` | **Fixed**<br>(Workload Key) | **Workload Constraint:** Defines the maximum sequence length. Dictates maximum XLA loop bounds and bounds checks. Changing this requires a new tuning and sweeping cycle. |
| Prompts / Request Rate | `NUM_PROMPTS`<br>`REQUEST_RATE` | **Fixed**<br>(Workload Key) | **Workload Constraint (Saturation):** Must be set high enough to generate a massive backlog. This ensures the TPU runs in a steady state for a sustained duration, providing stable and accurate throughput measurements. |

## Future Work: Hybrid VMEM-Aware Tuning Pruning

To avoid wasting TPU time on combinations that are mathematically guaranteed to fail, we will implement a hybrid tuning strategy. 

**The Goal:**
1. **Pre-flight Calculation (Theoretical):** Before sending a `TuningKey` to XLA, use `get_smem_estimate_bytes()` and `get_vmem_estimate_bytes()` (from `tpu_inference/kernels/ragged_paged_attention/v3/kernel.py`) to estimate the VMEM footprint of the specific `(bq_sz, bkv_sz)` tile against the specific Model architecture (`head_dim`, `num_q_heads`, etc).
2. **Static Pruning:** If the estimated VMEM exceeds the TPU hardware limits (~16MB on v6e/v7x), silently drop that combination from the tuning queue.
3. **Dynamic Pruning (Empirical bounds-checking):** For the remaining cases, wrap the JAX execution in a `try/except` block. If the compiler still throws a `ResourceExhausted` error (because of XLA padding overheads), immediately record that boundary and prune all larger compute/memory tiles from the remaining search space for that specific `page_size` / `chunk_prefill_size`.

This transforms the tuner from a "dumb grid search" into an intelligent, hardware-aware optimizer, allowing us to safely define massive search spaces (like `K=8192` and `bq_sz=8192`) without the penalty of watching the compiler crash thousands of times.
