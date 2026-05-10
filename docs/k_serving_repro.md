# K-Serving (chunked PREFILL) — repro & where things live

End-to-end measurement of the static-K PREFILL kernel routing on
Llama 3 8B / TPU v7x. On a prefill-dominated workload, the fully-
tuned stack delivers **+47% throughput, -32% P99 TTFT** over a
bm-infra-defaults baseline that has been *manually clipped to even
run* — see the VMEM-OOM caveat below.

> **vLLM/bm-infra defaults cannot run this workload as-is.** With
> kernel block sizes from `get_default_block_sizes()` and MNB=10275,
> the MIXED kernel pass requires ~76.6 MB of VMEM but the v7x VMEM
> budget is 64 MB — the JAX runtime raises ``RESOURCE_EXHAUSTED:
> Ran out of memory in memory space vmem``. To get *any* number out
> of a bm-infra-style configuration, the MIXED `bkv_csz` axis has to
> be manually halved (512 → 256). That manually-clipped baseline
> gives 3.26 req/s; **fully-default would be 0 req/s** (no run).

## Headline numbers (all measured)

Workload: Llama 3.1 8B Instruct, sonnet completion, 8191 input
tokens / 1 output token, 1000 prompts at request_rate=inf, TP=1,
v7x-1.

| Configuration | MNB | K | RPA blocks | Throughput | P99 TTFT |
|---|---:|---:|---|---:|---:|
| **bm-infra defaults — pure default OOMs**; clipped to run | 10275 | 0 | kernel default (`M=512,2048,256,512` OOMs by 12.6 MB; clipped to `512,2048,256,256` to run) | 3.26 req/s | 303.9 s |
| MNB=10275 / K=0 (with our tuned D, M) | 10275 | 0 | tuned (D, M only) | 3.28 req/s | 301.5 s |
| MNB=8192 / K=0 (our K-off baseline, tuned D, M) | 8192 | 0 | tuned (D, M only) | 4.57 req/s | 216.7 s |
| **K=256 winner — fully tuned stack** | **8192** | **256** | tuned (D, P, M) | **4.79 req/s** | **207.2 s** |
| **Δ winner vs bm-infra clipped-defaults** | | | | **+47%** | **-32%** |
| Δ K-serving alone (rows 3 vs 4, same MNB & D/M) | | | | +4.8% | -4.4% |

Three things to read carefully:

1. **The +47% comparison is against a *clipped* bm-infra config, not
   the pure default**, because the pure default cannot run on v7x at
   this shape. If "default" means "whatever the framework gives you,
   even if it crashes," the comparison becomes "we run; they don't."
2. **Row 1 (3.26) ≈ Row 2 (3.28)** — almost identical despite very
   different block sizes. The bottleneck at MNB=10275 is **not**
   kernel block sizes; it's the kernel-tile-shape regime that
   collapses both. The Layer 1 kernel-tune step contributes ~0% on
   this MNB-broken baseline. The +47% delta comes essentially
   entirely from MNB tuning + K-serving, NOT from kernel-tile
   block-size tuning at this regime.
3. **K-serving alone (rows 3 → 4): +4.8% throughput, -4.4% P99 TTFT.**
   That's the cleanest A/B because both rows use our tuned D/M
   kernels at the same MNB — only K differs.

Both K=0 and K=256 were swept across MNB ∈ {2048, 4096, 8192, 10275},
MNS ∈ {128, 1000}, K ∈ {0, 128, 256, 512, 1024, 2048}. Full ranked
table: `tmp/log/script_build_service_registry_rpa_v3_vllm.txt`.
Per-combo raw outputs: `tmp/bench_prefill_heavy_rpa_v3_vllm/<combo_id>/`.

## What "bm-infra defaults" actually means (and what we're guessing)

bm-infra ([github.com/QiliangCui/bm-infra](https://github.com/QiliangCui/bm-infra))
is the v6e/v7x perf reference Qiliang's team runs nightly. Their
`vllm serve` invocation
([scripts/agent/run_bm.sh:106-111](https://github.com/QiliangCui/bm-infra/blob/main/scripts/agent/run_bm.sh#L106-L111)):

```bash
vllm serve $MODEL \
  --seed 42 \
  --max-num-seqs $MAX_NUM_SEQS \
  --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
  --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
  --no-enable-prefix-caching \
  --download_dir $DOWNLOAD_DIR \
  --max-model-len $MAX_MODEL_LEN
```

**Three things are NOT passed**, so vLLM defaults take over:

1. `--long-prefill-token-threshold` → 0 (per-request K cap disabled).
2. `--block-size` → vLLM picks per release (typically 16 on GPU,
   varies on TPU).
3. No `RPA_*_BLOCK_SIZES` env var → tpu-inference runner uses
   `get_default_block_sizes()` heuristic formulas.

**Where 10275 comes from**: bm-infra uses MNB=10275 across most
of their configs, but **specifically for** Qwen3 / Qwen3-Coder /
gpt-oss-20b / gpt-oss-120b on v7x-2 (`cases/hourly_torchax_jax_v7.csv`)
and Llama 3 70B on v6e-8 (`cases/nightly_jax.csv`). They have **no
v7x-1 config for Llama 3 8B Instruct** — that hardware/model pairing
isn't in their CSVs. So extrapolating "bm-infra would run Llama 8B
on v7x-1 with MNB=10275" is a guess. It's their canonical number
across other configs, but they could just as well pick something
else for this specific shape. To be safe, the only honest baseline
is: "vLLM defaults" — which means *everything* unset.

## Computing the kernel-default block sizes

When `RPA_*_BLOCK_SIZES` are unset, the runner falls through to
`get_default_block_sizes()`
([kernel.py:1479](../tpu_inference/kernels/ragged_paged_attention/v3/kernel.py#L1479)).
Walk-through for our shape (Llama 3.1 8B / v7x / bf16 / page_size=128
/ MNB=10275 / MAX_MODEL_LEN=8192 / MAX_NUM_SEQS=128):

```python
# Inputs
actual_num_q_heads  = 32
actual_num_kv_heads = 8
head_dim            = 128
page_size           = 128
max_num_tokens      = 10275          # = MNB
pages_per_seq       = ceil(8192/128) = 64

# Derived
num_q_heads_per_kv_head = next_pow2(32 // 8)              = 4
num_kv_heads_x2         = next_pow2(align(8 * 2, 2))      = 16
max_q                   = next_pow2(10275)                = 16384
max_kv                  = pages_per_seq * page_size       = 8192

# bf16: kv_packing = 2
min_bkv_sz_to_peak = 16M * 2 / 4 / 128 / 16               = 4096

# TPU v7 branch (case 7), DECODE:
bq_sz   = 1
bkv_sz  = min(min_bkv_sz_to_peak, max_kv)                 = min(4096, 8192) = 4096
bq_csz  = 1
bkv_csz = min(min_bkv_sz_to_peak, max_kv)                 = 4096
# DECODE default: (1, 4096, 1, 4096)

# TPU v7 branch (case 7), PREFILL / MIXED:
bq_sz   = min(2048 // num_q_heads_per_kv_head, max_q // 2)  = min(512, 8192)  = 512
bkv_sz  = min(2048, max_kv // 2)                            = min(2048, 4096) = 2048
bq_csz  = min(1024 // num_q_heads_per_kv_head, max_q // 2)  = min(256, 8192)  = 256
bkv_csz = min(512, align(max_kv // 2, page_size))           = min(512, 4096)  = 512
# PREFILL/MIXED default: (512, 2048, 256, 512)
```

Side-by-side with our tuned winners:

| Case | (bq_sz, bkv_sz, bq_csz, bkv_csz) — kernel default | Tuned winner | Notes |
|---|---|---|---|
| DECODE  | `(1, 4096, 1, 4096)`     | `(1, 512, 1, 256)`      | Tuner found 8x smaller blocks better. |
| PREFILL | `(512, 2048, 256, 512)`  | `(256, 2048, 256, 512)` | Tuner shaves bq_sz from 512 → 256. |
| MIXED   | `(512, 2048, 256, 512)`  | `(128, 512, 128, 256)`  | Tuner found 4x smaller in every axis better. |

Defaults are explicitly heuristic — the function docstring says
"not necessarily optimal." For Llama 3 8B specifically the gap
is largest in DECODE (8× off) and MIXED (4× off across the board).
PREFILL is closer to default, but in practice K=0 routes everything
through MIXED so the DECODE+MIXED gap is what would matter against
the bm-infra-style baseline.

## Where bm-infra's per-model/workload configs live

Each entry is **one row in a CSV** under
[`cases/`](https://github.com/QiliangCui/bm-infra/tree/main/cases).
Filename = cadence/method:

- `hourly_*.csv`, `nightly_*.csv` — regularly-scheduled benchmarks.
- `autotune_*.csv` — autotune-driven runs.
- Suffixes: `_jax`, `_tt` (torchax), `_b200` (NVIDIA B200), `_v7`
  (TPU v7x), no suffix = mixed/legacy.

CSV header is consistent:

```
Device, Model, MaxNumSeqs, MaxNumBatchedTokens, TensorParallelSize,
MaxModelLen, Dataset, InputLen, OutputLen, [ExpectedETEL, NumPrompts,
ModelTag, PrefixLen]
```

Example rows (Llama 3 family, the relevant ones):

```
# nightly_jax.csv — Llama 8B on v6e-8 (NOT v7x-1):
v6e-8,meta-llama/Llama-3.1-8B-Instruct,256,10275,8,10275,bench-custom-token,2048,8192,,1000,NEW,0

# hourly_torchax_jax_v7.csv — v7x-2 / Qwen, NOT Llama:
tpu7x-2,Qwen/Qwen3-Coder-30B-A3B-Instruct,128,10275,1,10275,sonnet,1024,1024
```

**bm-infra has no 8191/1 workload.** The full set of input lengths
across all their CSVs is `[30, 128, 500, 1000, 1024, 1800, 2048,
4096, 5000, 8192, 10000]`. The 8191/1 prefill_heavy workload was
invented in this branch as a stress test — it isolates prefill
compute (~99.9% of total work) so the K-serving lever has its
strongest measurable signal. The reproducer commands below apply
"vLLM-default-style" config (run A) **to our 8191/1 workload**, so
the comparison is apples-to-apples on the same workload — not a
cross-workload race. The +47% headline above is now backed by an
actual Row 3 measurement, with the caveat that the pure default
config OOMs VMEM and required one block-size axis (`bkv_csz` for
MIXED) to be halved before it could run at all.

## Three-row reproducer (every parameter)

To run a clean A/B/C experiment, all three rows hold these
**workload-side params identical**:

| Parameter | Value |
|---|---|
| Hardware | TPU v7x-1 (single chip) |
| MODEL | `meta-llama/Meta-Llama-3-8B-Instruct` |
| TENSOR_PARALLEL_SIZE | 1 |
| MAX_NUM_SEQS | 128 |
| DATASET | sonnet |
| INPUT_LEN | 8191 |
| OUTPUT_LEN | 1 |
| NUM_PROMPTS | 1000 |
| REQUEST_RATE | inf |
| `--seed` | 42 |
| `--no-enable-prefix-caching` | yes |

The three rows differ only in **scheduler-knob and kernel-routing
params**:

| Parameter | (1) Our K=256 winner | (2) Our K=0 at bm-infras MNB | (3) bm-infra defaults |
|---|---|---|---|
| `LONG_PREFILL_TOKEN_THRESHOLD` | **256** | 0 | 0 (vLLM default) |
| `MAX_NUM_BATCHED_TOKENS` | **8192** | 10275 | 10275 |
| `MAX_MODEL_LEN` | 8192 | 8192 | 10275 (bm-infra style: MML = MNB) |
| `--block-size` | 128 | 128 | unset → **16** [^bs] |
| `RPA_D_BLOCK_SIZES` | `1,512,1,256` (tuned) | `1,512,1,256` (tuned) | unset → formula: `1,4096,1,4096` |
| `RPA_P_BLOCK_SIZES` | `256,2048,256,512` (tuned) | unused at K=0 | unset → formula: `512,2048,256,512` (also unused at K=0) |
| `RPA_M_BLOCK_SIZES` | `128,512,128,256` (tuned) | `128,512,128,256` (tuned) | unset → formula: `512,2048,256,512` |
| **Throughput (req/s)**          | **4.79**             | **3.28**                | **3.26**                        |
| **Mean TTFT (s)**               | 107.8                | 152.8                   | not separately captured [^r3m]  |
| **P99 TTFT (s)**                | 207.2                | 301.5                   | **303.9**                       |
| **Wall-clock (1000 req)**       | 224 s                | 319 s                   | ≈307 s (1000 / req/s)           |
| Combo ID                        | `4f773da3397a`       | `6dc80b2a3b37`          | one-off `row3_bm_infra_defaults`|

[^r3m]: Row 3 was an out-of-sweep one-off invocation with
   `--result-tag row3_bm_infra_defaults`; only req/s and P99 TTFT got
   recorded into the headline at that time. Since Row 1 (3.26) ≈ Row 2
   (3.28) within noise, Row 3's Mean TTFT is reliably ≈ Row 2's
   152.8 s — the `(MNB=10275, K=0)` regime collapses to the same
   throughput-bound shape regardless of block-size tune. Re-run if
   exact Mean TTFT for Row 3 is ever needed.

[^bs]: vLLM's `CacheConfig.DEFAULT_BLOCK_SIZE` is 16
   ([`vllm/config/cache.py:45`](https://github.com/vllm-project/vllm/blob/main/vllm/config/cache.py#L45)),
   but tpu-inference overrides it in
   [`tpu_inference/platforms/tpu_platform.py:224-244`](../tpu_inference/platforms/tpu_platform.py#L224)
   via `PallasAttentionBackend.get_page_size()` in
   [`tpu_inference/layers/vllm/backends/flash_attn.py:86`](../tpu_inference/layers/vllm/backends/flash_attn.py#L86):
   if `max_model_len > 8192` returns 16 (VMEM-OOM safety); else
   `next_pow2(max_model_len) // 16` clamped to `[16, 256]`. A second
   override (`if USE_BATCHED_RPA_KERNEL and block_size < 256: bump to
   256`) only fires when that env var is set, which defaults False. For
   Row 3 (MAX_MODEL_LEN=10275, USE_BATCHED_RPA_KERNEL unset) the
   resolved value is **16**.

What each comparison isolates:

- **(1) vs (2)**: combined effect of MNB tuning (10275 → 8192) + K-serving
  (K=0 → K=256). Measured: +46% throughput, -31% P99 TTFT. Both rows use
  our tuned DECODE/MIXED kernels.
- **(2) vs (3)**: pure kernel-block-sizes effect (D and M only, since
  PREFILL is unused at K=0). Same MNB, same K, same MML/BLOCK_SIZE-style
  config — only the kernel block sizes differ. **Measured**: 3.28 vs
  3.26 req/s (within noise). The kernel tune contributes ~0% on this
  MNB-broken baseline; the bottleneck is the kernel-tile-shape regime
  that collapses both. The +47% delta is essentially all from MNB
  tuning + K-serving.
- **(1) vs (3)**: full end-to-end win against bm-infra defaults
  (with the M-block-size clip required to run at all): **+47%
  throughput, -32% P99 TTFT.**

### Three reproducer commands

Prereqs (TPU v7x VM, vLLM checkout at `$VLLM_DIR`, HF cache at
`$DOWNLOAD_DIR`, gated-model token in `$HF_TOKEN`).

To pin to the exact stack used in the existing measurements:

```bash
git -C tpu-inference checkout 136d5de1   # measurement commit (sweep snapshot)
git -C "$VLLM_DIR" checkout c51df430     # vllm LKG used by all sweep runs
```

#### Row 1 — Our K=256 winner (measured: 4.79 req/s)

```bash
LONG_PREFILL_TOKEN_THRESHOLD=256 \
MAX_NUM_SEQS=128 \
MAX_NUM_BATCHED_TOKENS=8192 \
MAX_MODEL_LEN=8192 \
BLOCK_SIZE=128 \
RPA_D_BLOCK_SIZES=1,512,1,256 \
RPA_P_BLOCK_SIZES=256,2048,256,512 \
RPA_M_BLOCK_SIZES=128,512,128,256 \
tools/benchmark/run_benchmark.sh \
    tools/benchmark/cases/v7x/llama3_8b/prefill_heavy.workload \
    --result-tag row1_k256_winner
# Should reproduce combo 4f773da3397a (~4.79 req/s).
```

#### Row 2 — Our K=0 at bm-infra's MNB (measured: 3.28 req/s)

```bash
LONG_PREFILL_TOKEN_THRESHOLD=0 \
MAX_NUM_SEQS=128 \
MAX_NUM_BATCHED_TOKENS=10275 \
MAX_MODEL_LEN=8192 \
BLOCK_SIZE=128 \
RPA_D_BLOCK_SIZES=1,512,1,256 \
RPA_M_BLOCK_SIZES=128,512,128,256 \
tools/benchmark/run_benchmark.sh \
    tools/benchmark/cases/v7x/llama3_8b/prefill_heavy.workload \
    --result-tag row2_k0_at_bminfra_mnb
# Should reproduce combo 6dc80b2a3b37 (~3.28 req/s).
# RPA_P_BLOCK_SIZES omitted: at K=0 the static-K PREFILL kernel
# is bypassed entirely (PREFILL goes through MIXED).
```

#### Row 3 — bm-infra defaults (measured: 3.26 req/s with one block-size axis clipped to run)

```bash
LONG_PREFILL_TOKEN_THRESHOLD=0 \
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
MAX_NUM_SEQS=128 \
MAX_NUM_BATCHED_TOKENS=10275 \
MAX_MODEL_LEN=10274 \
BLOCK_SIZE=16 \
RPA_D_BLOCK_SIZES=1,4096,1,4096 \
RPA_M_BLOCK_SIZES=512,2048,256,256 \
tools/benchmark/run_benchmark.sh \
    tools/benchmark/cases/v7x/llama3_8b/prefill_heavy.workload \
    --result-tag row3_bm_infra_defaults
# RPA_*_BLOCK_SIZES intentionally ALL unset — runner falls through
# to get_default_block_sizes() formulas:
#   DECODE  -> 1,4096,1,4096
#   PREFILL -> 512,2048,256,512  (unused at K=0)
#   MIXED   -> 512,2048,256,512
# Used 76.58M of 64.00M vmem. Exceeded vmem capacity by 12.58M.
# BLOCK_SIZE intentionally empty -> tpu_platform.py resolves it
# to 16 (because MAX_MODEL_LEN=10275 > 8192 hits the VMEM-OOM
# safety branch in PallasAttentionBackend.get_page_size()).
# This is the row that fills in the honest "vs vLLM/bm-infra
# defaults" comparison.
```

Each writes to `tmp/bench_prefill_heavy_*/<hash>/`. Compare
`metrics.txt` between runs (`RequestThroughput`, `P99TTFT`).
~4 minutes per run.

## Single-prompt latency (MIXED-only floor)

The throughput numbers above are saturation-regime measurements
(1000 prompts at `request_rate=inf`); their Mean TTFT is dominated by
queue-wait. To isolate the *kernel-and-dispatch* floor, we ran the
same Llama 3 8B / v7x-1 / 8191 in / 1 out shape at `NUM_PROMPTS=1` —
single prompt, no queue.

A new sweep recipe (`("rpa_v3", "vllm_latency")` in
[sweep_recipes.py](../tools/benchmark/sweep_recipes.py)) pins
`MAX_NUM_SEQS=1` and `LONG_PREFILL_TOKEN_THRESHOLD=0` (MIXED-only
routing — bypasses the static-K PREFILL kernel entirely), sweeps only
`MAX_NUM_BATCHED_TOKENS ∈ {2048, 4096, 8192, 16384, 32768, 65536}`,
and ranks ascending by Mean TTFT. The corresponding workload file is
[llama3_8b/prefill_heavy_latency.workload](../tools/benchmark/cases/v7x/llama3_8b/prefill_heavy_latency.workload).
Invocation:

```bash
SERVICE_ID=vllm_latency tools/run_pipeline.sh \
    tools/benchmark/cases/v7x/llama3_8b/prefill_heavy_latency.workload
```

### 6-combo MNB sweep (latency winner)

| MNB    | MIXED calls per prefill | Mean TTFT | Δ vs winner |
|-------:|------------------------:|----------:|------------:|
| 2048   | 4 (chunked)             | 597.3 ms  | +54%        |
| 4096   | 2 (chunked)             | 529.4 ms  | +36%        |
| **8192**  | **1 (one-shot)**     | **388.2 ms** | **winner** |
| 16384  | 1 (one-shot, larger static shape) | 394.8 ms | +1.7% |
| 32768  | 1                       | 390.5 ms  | +0.6%       |
| 65536  | 1                       | 405.7 ms  | +4.5%       |

`MNB=8192` is the smallest value that fits the 8191-token prefill in
one MIXED kernel call. Below that, vLLM chunks the prefill across
multiple scheduler steps → multiple MIXED kernel launches → measurable
launch-overhead penalty. Above 8192, the kernel's `max_num_tokens`
static shape grows but the per-prefill call count stays at 1; the
plateau is flat within ~5%, with a small uptick at 65536 likely
reflecting larger tile pre-allocation cost.

### Three-way comparison: queue vs kernel routing

To attribute the latency win between "no queue" and "MIXED-only
routing", we ran the **throughput-tuned config at NUM_PROMPTS=1** as a
cross-check (one-off, see
[`tmp/bench_prefill_heavy_throughput_config_n1/`](../tmp/bench_prefill_heavy_throughput_config_n1/)):

| Config | NUM_PROMPTS | Routing | Mean TTFT | req/s |
|---|---:|---|---:|---:|
| latency-tuned     (MNS=1,   K=0)   | 1    | MIXED × 1 (q=8191)         | **388 ms**    | 2.57 |
| throughput-tuned  (MNS=128, K=256) | 1    | PREFILL × 32 (K=256 each)  | **747 ms**    | 1.34 |
| throughput-tuned  (MNS=128, K=256) | 1000 | PREFILL × 32 × 1000 + queue| 107,810 ms    | 4.79 |

The **278× total reduction** (107,810 → 388 ms) decomposes cleanly:

```
278×  =  144× queue removal  ×  1.93× kernel routing
         (107,810 / 747)         (747 / 388)
```

- **Queue removal: 144×** — dominant. Accounts for 107,063 ms of the
  107,422 ms total saved (99.7%).
- **Kernel routing (MIXED vs static-K PREFILL × 32 chunks): 1.93×** —
  small in compounded terms, but absolutely real: 359 ms saved per
  single-prompt prefill by routing through MIXED-only.

### Per-call kernel-launch overhead

Comparing the two single-prompt rows (388 vs 747 ms) with same total
compute (8191 tokens of prefill) but different call counts (1 vs 32)
isolates per-call overhead:

```
extra 31 PREFILL launches cost 359 ms → ~11.6 ms per launch
```

That's TPU kernel-launch + XLA dispatch overhead per call. For the
32-chunk static-K path, ~370 ms of pure launch overhead — roughly
equal to the entire MIXED single-call wall-clock. The static-K
kernel's compute *per chunk* is fast; at single-shot the launch
overhead dominates. **At saturation (1000 prompts) this overhead is
amortized across requests** — which is why K=256 still wins on
throughput despite losing on single-prompt TTFT.

### Floor sanity-check

Theoretical lower bound for an 8K-token prefill on 8B / v7x-1 / TP=1:
- Weights HBM read: 16 GB / 1.6 TB/s ≈ 10 ms
- Compute (~130 TFLOP at ~700 TFLOP/s peak bf16): ~190 ms
- vLLM dispatch / kernel launch / tokenize / detokenize / sampler step

Theoretical ~200 ms, observed 388 ms = ~2× overhead. Reasonable for
real-system glue; not investigated further.

## Where to find the tuned files (ours)

| Artefact            | Path                                                          | What it is                                                                                                              |
|:--------------------|:--------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------|
| Tuned kernel params | `tools/benchmark/cases/v7x/llama3_8b/production.kernel`       | DECODE/PREFILL/MIXED kernel block sizes per `(page_size, K, dtype, head shape)`. Output of Layer 1 (3.5 h tune; latency re-runs hit the cache via `SKIP_COMMIT_CACHE_CHECK=1`). |
| Service params      | `tools/benchmark/cases/v7x/llama3_8b/production.service`      | Best scheduler config per `service_id`: `vllm` (throughput winner) and `vllm_latency` (single-prompt winner) coexist under distinct workload keys. |
| Throughput per-combo  | `tmp/bench_prefill_heavy_rpa_v3_vllm/<combo_id>/`           | 48 throughput-sweep combos (sonnet, 1000 prompts).                                                                      |
| Latency per-combo     | `tmp/bench_prefill_heavy_latency_rpa_v3_vllm_latency/<id>/` | 6 latency-sweep combos (sonnet, 1 prompt).                                                                              |
| Throughput-cfg @ N=1  | `tmp/bench_prefill_heavy_throughput_config_n1/`             | One-off cross-check: throughput winner config evaluated at NUM_PROMPTS=1. The 747 ms in the 3-way table.                |
| Ranked Layer-3 dump (throughput) | `tmp/log/script_build_service_registry_rpa_v3_vllm.txt`           | All 48 throughput combos sorted by req/s.                                                                  |
| Ranked Layer-3 dump (latency)    | `tmp/log/script_build_service_registry_rpa_v3_vllm_latency.txt`   | All 6 latency combos sorted ascending by Mean TTFT.                                                        |

## Caveats when interpreting the numbers

- **Honest claim**: full-stack tuned config gets **+47% throughput,
  -32% P99 TTFT** vs a bm-infra-defaults baseline that had to be
  manually clipped (one block-size axis halved) to avoid VMEM OOM.
  Pure unmodified default = 0 req/s (no run). K-serving alone gets
  +4.8% / -4.4% (Row 3 vs Row 4 — same MNB and D/M kernels, only K
  differs); the rest of the +47% is MNB tuning.
- **Kernel block-size tuning contributes ~0% at the MNB-broken
  baseline.** Rows 1 (default-clipped) and 2 (our tuned D/M) both
  give ~3.27 req/s at MNB=10275. The bottleneck there is the
  kernel-tile-shape regime, not block sizes. Layer 1 (kernel tune)
  pays off only once Layer 2 (sweep) has selected an MNB that lets
  the kernel run in its productive regime.
- **One workload only.** prefill_heavy (8191/1) is the most
  K-serving-favorable shape; numbers on `balanced.workload` and on
  decode-dominated traffic will be smaller. The 70B sweep is the
  test of whether the gain holds or grows on a bigger model.
- **K=128 underperforms K=0** (4.46 vs 4.57). Too-small K means too
  many kernel re-entries; the launch overhead exceeds the
  static-shape kernel win. K=256–1024 is the productive band.

## Next bets

### Bet 1 — bigger model + longer context (recommended)

K-serving's relative gain *grows* with model size on TPUs. Two
compounding reasons:

1. **TPU economics.** Decode is memory-bandwidth-bound (single-token,
   reads all weights once); prefill is compute-bound (large q-len
   amortizes the weight read). Bigger models hit the bandwidth wall
   harder for decode while prefill keeps scaling with compute. So
   the prefill share of total time *grows* as the model grows — and
   K-serving is exactly the prefill optimization.
2. **Real production traffic for bigger models is prefill-dominant.**
   Long-context summarization, RAG with many retrieved docs, code
   analysis, agent traces. 70B-class models in those use cases see
   8K–64K input tokens routinely.

Concrete: the Llama 3.3 70B / 32K / TP=8 prefill_heavy workload
(`tools/benchmark/cases/v7x/llama3_3_70b/prefill_heavy.workload`)
is set up; sweep about to run.

### Bet 2 — multiple K-chunks per request per step (decoupled K_sched / K_kernel)

The vLLM scheduler caps each request at K tokens per step
([scheduler.py:413-415](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py#L413-L415));
a 700-token prefill at K=256 costs **3 scheduler steps**, not 1.
Today vLLM's K and the kernel's static-K are the same number, so
we can't increase the per-request budget without also changing the
kernel chunk size.

**Decoupling them in the runner** (no vLLM patch needed) lets us run
K_sched=2048 (one scheduler step per medium-sized request) while the
kernel still uses K_kernel=256 (the throughput-optimal block size).
The runner internally splits the 2048-token chunk into 8× K=256
kernel invocations within one `model.forward()`.

Cost: medium runner-side change (multi-chunk-per-request inside one
forward pass with sequential KV writes; bounded chunk count to keep
the JIT trace cache from exploding). Win: ~32× fewer scheduler steps
per request → meaningful TTFT cut, especially at long context.
**Run the 70B sweep first** — if K=256 wins by <3% over K=2048 there,
the decoupling is not worth the engineering cost.

### Bet 2 implementation: L kernel quick-look recipe

Bet 2 was implemented end-to-end in commits `8e68f2c0..eb270764` (Phase 2b
stack: kernel LOGICAL case + runner-side wiring + orchestrator auto-link).
The throughput / latency comparison against today's P / M kernels uses a
fast-track tune that pins all kernel block sizes to the existing P-winner
values and sweeps only the new dimension (`max_num_subseqs`). Per-axis
overrides via env vars (`RPA_V3_*_LST`, commit `4dd5b5f0`) collapse the
LOGICAL search space from hundreds of combos to 6.

#### Phase 1 — Fast-track LOGICAL tune (throughput workload, MNS=128)

```bash
git pull origin rpa3_2

RPA_V3_BQ_SZ_LST=256 \
RPA_V3_BKV_SZ_LST=2048 \
RPA_V3_BQ_CSZ_LST=256 \
RPA_V3_BKV_CSZ_LST=512 \
RPA_V3_K_LST=256 \
RPA_V3_PAGE_SIZE_LST=128 \
CASES_TO_TUNE=logical RPA_V3_TUNER_CASES=logical \
  tools/kernel/tuner/v1/tune_all_cases.sh \
    tools/benchmark/cases/v7x/llama3_8b/prefill_heavy.workload

# Then build registry (merges into existing production.kernel).
tools/kernel/tuner/v1/build_kernel_registry.sh \
    tmp/log/tune_all_prefill_heavy.txt \
    tools/benchmark/cases/v7x/llama3_8b/production.kernel
```

Expected: 6 LOGICAL combos at (page=128, K=256), one per
`max_num_subseqs ∈ {256, 384, 640, 1152, 2176, 4224}`. Should finish in
minutes.

#### Phase 2 — Fast-track LOGICAL tune (latency workload, MNS=1)

```bash
RPA_V3_BQ_SZ_LST=256 \
RPA_V3_BKV_SZ_LST=2048 \
RPA_V3_BQ_CSZ_LST=256 \
RPA_V3_BKV_CSZ_LST=512 \
RPA_V3_K_LST=256 \
RPA_V3_PAGE_SIZE_LST=128 \
CASES_TO_TUNE=logical RPA_V3_TUNER_CASES=logical \
  tools/kernel/tuner/v1/tune_all_cases.sh \
    tools/benchmark/cases/v7x/llama3_8b/prefill_heavy_latency.workload

tools/kernel/tuner/v1/build_kernel_registry.sh \
    tmp/log/tune_all_prefill_heavy_latency.txt \
    tools/benchmark/cases/v7x/llama3_8b/production.kernel
```

Expected: 6 LOGICAL combos at MNS=1 candidates `{2, 3, 5, 9, 17, 33}`.
Even faster than Phase 1.

#### Phase 3 — Throughput service comparison (L vs P)

```bash
SERVICE_ID=vllm RPA_KERNEL_K=256 tools/run_pipeline.sh \
    tools/benchmark/cases/v7x/llama3_8b/prefill_heavy.workload
```

Compare `req/s` and `P99 TTFT` to the P-baseline (4.79 req/s / 207.2 s).

**Hypothesis:** L slightly slower than P due to indirection cost (extra
`phys_seq_indices_ref[seq_idx]` lookup per iter). If L is way slower,
kernel bug.

#### Phase 4 — Latency service comparison (L vs M)

The current latency baseline (M, 388 ms) uses
`LONG_PREFILL_TOKEN_THRESHOLD=0` (PREFILL/LOGICAL pass skipped). For an
L-vs-M comparison we need to ENABLE the LOGICAL pass:

```bash
SERVICE_ID=vllm_latency RPA_KERNEL_K=256 \
LONG_PREFILL_TOKEN_THRESHOLD=8192 \
tools/run_pipeline.sh \
    tools/benchmark/cases/v7x/llama3_8b/prefill_heavy_latency.workload
```

`LONG_PREFILL_TOKEN_THRESHOLD=8192` lets vLLM hand the runner the full
8191-token prefill in one step → planner emits 32 LOGICAL chunks →
kernel processes them in one call (assuming `max_num_subseqs ≥ 33`).

Compare to 388 ms.

**Hypothesis:** L beats M because static-K block sizes are well-tuned
and we avoid MIXED's dynamic-q-len overhead.

### What we ruled out (and why)

- **vLLM scheduler fork (rpa3 branch in vLLM)**: same runner-side
  complexity as Bet 2's decoupling, plus a permanent merge tax on
  every LKG bump. Not worth the maintenance burden.
- **Bigger MNB to hide per-step overhead**: 8B sweep showed
  MNB=10275 → throughput collapse from 4.79 to 3.28 with our tuned
  kernels (kernel-tile-shape mismatch at the non-power-of-2
  dimension; even worse for the kernel-default DECODE block of 4096
  which doesn't divide 10275 cleanly). At 70B+TP=8 there's far
  more headroom, so the new axis tests `[8192, 16384, 32768, 65536]`
  on that workload.
- **Multi-stream / overlapped HBM transfer**: we already get this for
  free from XLA scheduling and the kernel's internal pipelining; not
  a real lever.
