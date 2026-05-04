# RPA v3 PREFILL tuning — Llama 3 8B on TPU v7x

First tuning run for the static-q-len PREFILL kernel flavor.

## Setup

- **Model**: Llama 3 8B — `num_q_heads=32, num_kv_heads=8, head_dim=128`, GQA 4:1
- **Dtypes**: q=bfloat16, kv=bfloat16
- **Context**: `max_model_len=8192`, no sliding window
- **TPU**: v7x (Ironwood), 8 chips × 2 cores
- **Run**: 1740 cases, ~3 hours wall-clock
  - 1434 SUCCESS / 96 SKIPPED (pre-flight VMEM) / 210 OOM at compile
- **Workload shape**: `max_num_tokens=2048`, `max_num_seqs=32`,
  `total_num_pages=4096`. Each prefill seq has `q_len = K`, `kv_len =
  max_model_len = 8192` to mirror steady-state chunked prefill (most KV
  read from paged cache, K new K/V tokens written this step).
- **Timing**: per-iter wall-clock (not async-dispatch — the async-dispatch
  patch landed *after* this run; subsequent runs will be tighter).

## Best `(bq_sz, bkv_sz, bq_csz, bkv_csz)` per (page_size, K)

| page_size | K     | latency (µs) | bq_sz | bkv_sz | bq_csz | bkv_csz | case_id |
|----------:|------:|-------------:|------:|-------:|-------:|--------:|--------:|
|        64 |   128 |       1342   |   128 |   2048 |    128 |    1024 |     385 |
|       128 |   128 |       1324   |   128 |   2048 |    128 |     512 |    1250 |
|        64 |   256 |       1288   |   256 |   2048 |    256 |     512 |     618 |
|       128 |   256 |       1279   |   256 |   1024 |    128 |    1024 |    1412 |
|        64 |   512 |       1279   |   256 |   2048 |    256 |     512 |     619 |
|       128 |   512 |       1267   |   256 |   1024 |    128 |    1024 |    1413 |
|        64 |  1024 |       1253   |   256 |   2048 |    256 |     512 |     620 |
|       128 |  1024 |       1258   |   256 |   2048 |    256 |     512 |    1490 |
|        64 |  2048 |       1213   |   256 |   1024 |    256 |     512 |     557 |
|       128 |  2048 |       1184   |   256 |    512 |    256 |     512 |    1367 |

## Observations

- **Larger K → faster per call.** K=2048 is ~12% faster than K=128.
  Makes sense: more amortization of fixed per-call overhead (DMA setup,
  kernel launch, etc.) over more useful tokens.
- **page_size=128 wins by a hair** vs page_size=64 (~1–3% across all K).
  Small enough to be in the noise, but consistent.
- **Block sizes converge**: `bq_sz=256, bq_csz=128–256, bkv_csz=512` is
  the winner zone for K ≥ 256. K=128 forces `bq_sz=128` (constraint
  `K % bq_sz == 0`).
- **bkv_sz drifts**: bigger for medium K (1024–2048), smaller for K=2048
  (where the seq-internal KV is shorter relative to the work).

## How to apply (quick A/B path)

For example, K=2048 + page_size=128 (best per-call latency):

```bash
export RPA_P_BLOCK_SIZES="256,512,256,512"
# vllm scheduler: --long-prefill-token-threshold=2048 (= K)
# Match block_size to page_size=128 in the engine config.
```

Note: per-call latency optimum (large K) ≠ end-to-end serving optimum.
Smaller K is better for decode-interleaved tail latency. A real A/B
should sweep K against your TTFT/ITL targets.

## Caveats for future runs

- Timing was per-iter sync wall-clock (current code is async-dispatch).
  Top configs within ~1% of each other should be re-measured.
- This used `max_num_tokens=2048`. If your serve uses a larger token
  budget (e.g. 8192), re-run with that — VMEM scratch grows with budget.
- Quantized KV (fp8) was not tuned here. Re-sweep for that dtype combo
  if you serve quantized.

## Raw inspector output (for reproducibility)

```
$ python3 -m tools.kernel.tuner.v1.inspect_result_cli --source=local --db-path=/tmp/kernel_tuner_run_2026_05_03_19_21_10 query_min_latency --case_set_id=rpa3-prefill-trial-1 --run_id=0
  tuning_key={"page_size": 128, "q_dtype": "bfloat16", "kv_dtype": "bfloat16", "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "max_model_len": 8192, "sliding_window": null, "case": "prefill", "chunk_prefill_size": 1024}  best_latency_us=1258  warmup_us=2888366  tunable_params={"bq_sz": 256, "bkv_sz": 2048, "bq_csz": 256, "bkv_csz": 512}  case_id=1490
  tuning_key={"page_size": 64, "q_dtype": "bfloat16", "kv_dtype": "bfloat16", "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "max_model_len": 8192, "sliding_window": null, "case": "prefill", "chunk_prefill_size": 1024}  best_latency_us=1253  warmup_us=2949052  tunable_params={"bq_sz": 256, "bkv_sz": 2048, "bq_csz": 256, "bkv_csz": 512}  case_id=620
  tuning_key={"page_size": 128, "q_dtype": "bfloat16", "kv_dtype": "bfloat16", "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "max_model_len": 8192, "sliding_window": null, "case": "prefill", "chunk_prefill_size": 128}  best_latency_us=1324  warmup_us=1923010  tunable_params={"bq_sz": 128, "bkv_sz": 2048, "bq_csz": 128, "bkv_csz": 512}  case_id=1250
  tuning_key={"page_size": 64, "q_dtype": "bfloat16", "kv_dtype": "bfloat16", "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "max_model_len": 8192, "sliding_window": null, "case": "prefill", "chunk_prefill_size": 128}  best_latency_us=1342  warmup_us=2734051  tunable_params={"bq_sz": 128, "bkv_sz": 2048, "bq_csz": 128, "bkv_csz": 1024}  case_id=385
  tuning_key={"page_size": 128, "q_dtype": "bfloat16", "kv_dtype": "bfloat16", "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "max_model_len": 8192, "sliding_window": null, "case": "prefill", "chunk_prefill_size": 2048}  best_latency_us=1184  warmup_us=2825129  tunable_params={"bq_sz": 256, "bkv_sz": 512, "bq_csz": 256, "bkv_csz": 512}  case_id=1367
  tuning_key={"page_size": 64, "q_dtype": "bfloat16", "kv_dtype": "bfloat16", "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "max_model_len": 8192, "sliding_window": null, "case": "prefill", "chunk_prefill_size": 2048}  best_latency_us=1213  warmup_us=2890882  tunable_params={"bq_sz": 256, "bkv_sz": 1024, "bq_csz": 256, "bkv_csz": 512}  case_id=557
  tuning_key={"page_size": 128, "q_dtype": "bfloat16", "kv_dtype": "bfloat16", "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "max_model_len": 8192, "sliding_window": null, "case": "prefill", "chunk_prefill_size": 256}  best_latency_us=1279  warmup_us=4859196  tunable_params={"bq_sz": 256, "bkv_sz": 1024, "bq_csz": 128, "bkv_csz": 1024}  case_id=1412
  tuning_key={"page_size": 64, "q_dtype": "bfloat16", "kv_dtype": "bfloat16", "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "max_model_len": 8192, "sliding_window": null, "case": "prefill", "chunk_prefill_size": 256}  best_latency_us=1288  warmup_us=2967696  tunable_params={"bq_sz": 256, "bkv_sz": 2048, "bq_csz": 256, "bkv_csz": 512}  case_id=618
  tuning_key={"page_size": 128, "q_dtype": "bfloat16", "kv_dtype": "bfloat16", "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "max_model_len": 8192, "sliding_window": null, "case": "prefill", "chunk_prefill_size": 512}  best_latency_us=1267  warmup_us=4910402  tunable_params={"bq_sz": 256, "bkv_sz": 1024, "bq_csz": 128, "bkv_csz": 1024}  case_id=1413
  tuning_key={"page_size": 64, "q_dtype": "bfloat16", "kv_dtype": "bfloat16", "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128, "max_model_len": 8192, "sliding_window": null, "case": "prefill", "chunk_prefill_size": 512}  best_latency_us=1279  warmup_us=2989478  tunable_params={"bq_sz": 256, "bkv_sz": 2048, "bq_csz": 256, "bkv_csz": 512}  case_id=619
```
