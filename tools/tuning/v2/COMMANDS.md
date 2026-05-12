# Tune-v2 — `if X, run Y`

All commands assume `cwd = repo root`. The Python `logging` library
prints `HH:MM:SS LEVEL tools.tuning.v2.<module>: <msg>` on stderr.

> ⚠️ **Never run smoke recipes against production workload dirs**
> (`cases/v7x/...`). `MOCK_TPU=1` / `MOCK_BENCH=1` stamp `mock: true`
> on raw rows BUT the projection layer doesn't filter them out
> today — a mock row in a production `.kernel.raw` / `.service.raw`
> partition will project as a winner and poison `lookup_env`.
> Smoke recipes below use the throwaway `cases/smoke/` subtree to
> stay isolated.

---

## Smoke (all four cases D/M/P/L, off-TPU, no commits, no real bench)

```bash
SMOKE_TEST=1 KERNEL_TUNER_NO_COMMIT=1 MOCK_TPU=1 MOCK_BENCH=1 \
EXTRA_TUNE_FLAGS="--iters 1 --warmup 0" \
tools/tuning/v2/scripts/run_pipeline.sh \
  tools/benchmark/cases/smoke/llama3_8b/throughput/prefill_heavy.workload
```

## Smoke on TPU (real kernel, mock bench)

```bash
SMOKE_TEST=1 KERNEL_TUNER_NO_COMMIT=1 MOCK_BENCH=1 \
EXTRA_TUNE_FLAGS="--iters 1 --warmup 0" \
tools/tuning/v2/scripts/run_pipeline.sh \
  tools/benchmark/cases/smoke/llama3_8b/throughput/prefill_heavy.workload
```

## Smoke for latency scenario

```bash
SMOKE_TEST=1 KERNEL_TUNER_NO_COMMIT=1 MOCK_BENCH=1 \
EXTRA_TUNE_FLAGS="--iters 1 --warmup 0" \
tools/tuning/v2/scripts/run_pipeline.sh \
  tools/benchmark/cases/smoke/llama3_8b/latency/prefill_heavy.workload
```

---

## Real throughput tune — v1-cadence (fastest, noisier)

```bash
EXTRA_TUNE_FLAGS="--iters 1 --warmup 0" \
KERNEL_TUNER_NO_COMMIT=1 \
tools/tuning/v2/scripts/run_pipeline.sh \
  tools/benchmark/cases/v7x/llama3_8b/throughput/prefill_heavy.workload \
  2>&1 | tee tmp/throughput_run.log
```

## Real throughput tune — clean steady-state (recommended)

```bash
EXTRA_TUNE_FLAGS="--iters 3 --warmup 1" \
KERNEL_TUNER_NO_COMMIT=1 \
tools/tuning/v2/scripts/run_pipeline.sh \
  tools/benchmark/cases/v7x/llama3_8b/throughput/prefill_heavy.workload \
  2>&1 | tee tmp/throughput_run.log
```

## Real latency tune

```bash
EXTRA_TUNE_FLAGS="--iters 3 --warmup 1" \
KERNEL_TUNER_NO_COMMIT=1 \
tools/tuning/v2/scripts/run_pipeline.sh \
  tools/benchmark/cases/v7x/llama3_8b/latency/prefill_heavy.workload \
  2>&1 | tee tmp/latency_run.log
```

---

## Save the log to a timestamped file

```bash
2>&1 | tee "tmp/run_$(date +%Y%m%d_%H%M%S).log"
```
(Append to the end of any command above instead of the plain `tee tmp/foo.log`.)

## Distributed tune across N TPU VMs (parallel workers)

Each worker hashes its combos into N buckets and measures only its
own. All workers append to the same `<workload>.kernel.raw/<sha>.jsonl`
and `<workload>.service.raw/<sha>.jsonl` (POSIX `O_APPEND` is atomic).

On each TPU VM, with `--worker-id` from 0 to N-1 and the same
`--worker-count=N`:

```bash
# VM 0:
EXTRA_TUNE_FLAGS="--iters 3 --warmup 1 --worker-id 0 --worker-count 4" \
EXTRA_SWEEP_FLAGS="--worker-id 0 --worker-count 4" \
KERNEL_TUNER_NO_COMMIT=1 \
tools/tuning/v2/scripts/run_pipeline.sh \
  tools/benchmark/cases/v7x/llama3_8b/throughput/prefill_heavy.workload \
  2>&1 | tee tmp/throughput_w0.log

# VMs 1, 2, 3: same recipe, change --worker-id.
```

To project / aggregate once all workers finish, re-run on any one
VM with `--from project_kernel`:
```bash
tools/tuning/v2/scripts/run_pipeline.sh --from project_kernel \
  tools/benchmark/cases/v7x/llama3_8b/throughput/prefill_heavy.workload
```

## Run in background, survive SSH disconnect

```bash
nohup bash -c '
  KERNEL_TUNER_NO_COMMIT=1 \
  tools/tuning/v2/scripts/run_pipeline.sh \
    tools/benchmark/cases/v7x/llama3_8b/throughput/prefill_heavy.workload \
    > tmp/throughput_run.log 2>&1
' &
echo "PID: $!"

# From any shell, watch progress:
tail -f tmp/throughput_run.log
```

## Push the log so I can read it

```bash
git add tmp/*.log && git commit -m "tune log" && git push
```

---

## Resume after Ctrl-C or crash

Re-run the same command. The skip-set picks up where it left off
(combos with `SUCCESS` / `FAILED_OOM` / `SKIPPED` status are
skipped; `UNKNOWN_ERROR` retried).

## Force a fully fresh tune (discard partial results)

```bash
rm tools/benchmark/cases/v7x/llama3_8b/throughput/prefill_heavy.kernel.raw/*.jsonl
```
Then re-run.

## Skip past steps that already succeeded

```bash
# Already tuned the kernel? Pick up at projection:
KERNEL_TUNER_NO_COMMIT=1 \
tools/tuning/v2/scripts/run_pipeline.sh \
  --from project_kernel \
  tools/benchmark/cases/v7x/llama3_8b/throughput/prefill_heavy.workload

# Already swept? Pick up at projection:
KERNEL_TUNER_NO_COMMIT=1 \
tools/tuning/v2/scripts/run_pipeline.sh \
  --from project_service \
  tools/benchmark/cases/v7x/llama3_8b/throughput/prefill_heavy.workload
```

Valid `--from` values: `validate`, `tune_kernel`, `project_kernel`,
`sweep_service`, `project_service`, `aggregate`.

---

## Land the results (commit + push)

Drop `KERNEL_TUNER_NO_COMMIT=1`. To commit but not push, use
`KERNEL_TUNER_NO_PUSH=1` instead.

```bash
EXTRA_TUNE_FLAGS="--iters 3 --warmup 1" \
tools/tuning/v2/scripts/run_pipeline.sh \
  tools/benchmark/cases/v7x/llama3_8b/throughput/prefill_heavy.workload \
  2>&1 | tee tmp/throughput_run.log
```

---

## Individual steps

```bash
# Validate one workload schema.
tools/tuning/v2/scripts/validate.sh \
  tools/benchmark/cases/v7x/llama3_8b/throughput/prefill_heavy.workload

# Kernel tune only (real TPU).
EXTRA_TUNE_FLAGS="--iters 3 --warmup 1" \
KERNEL_TUNER_NO_COMMIT=1 \
tools/tuning/v2/scripts/tune_kernel.sh \
  tools/benchmark/cases/v7x/llama3_8b/throughput/prefill_heavy.workload \
  --iters 3 --warmup 1

# Project kernel raw → .kernel.
KERNEL_TUNER_NO_COMMIT=1 \
tools/tuning/v2/scripts/project_kernel.sh \
  tools/benchmark/cases/v7x/llama3_8b/throughput/prefill_heavy.workload

# Service sweep only.
KERNEL_TUNER_NO_COMMIT=1 \
tools/tuning/v2/scripts/sweep_service.sh \
  tools/benchmark/cases/v7x/llama3_8b/throughput/prefill_heavy.workload

# Project service raw → .service.
KERNEL_TUNER_NO_COMMIT=1 \
tools/tuning/v2/scripts/project_service.sh \
  tools/benchmark/cases/v7x/llama3_8b/throughput/prefill_heavy.workload

# Aggregate per-workload winners → production.{kernel,service}.
KERNEL_TUNER_NO_COMMIT=1 \
tools/tuning/v2/scripts/aggregate.sh \
  tools/benchmark/cases/v7x/llama3_8b/throughput
```

---

## Single-combo TPU sanity test

```bash
cat > /tmp/combo.json <<'EOF'
{
  "tuning_key": {
    "kernel_variant": "rpa_v3",
    "hardware":       "tpu_v7x",
    "schema_version": 1,
    "case":           "logical",
    "page_size":      128,
    "kernel_K":       256,
    "max_num_seqs":   128,
    "code_revision":  "smoke",
    "num_q_heads":    32,
    "num_kv_heads":   8,
    "head_dim":       128,
    "max_model_len":  8192,
    "q_dtype":        "bfloat16",
    "kv_dtype":       "bfloat16",
    "sliding_window": null
  },
  "tunable_params": {
    "bq_sz": 256, "bkv_sz": 2048,
    "bq_csz": 256, "bkv_csz": 512,
    "mnss": 4224
  }
}
EOF

MAX_NUM_SEQS=128 MAX_NUM_BATCHED_TOKENS=8192 \
NUM_Q_HEADS=32 NUM_KV_HEADS=8 HEAD_DIM=128 MAX_MODEL_LEN=8192 \
python3 -m tools.tuning.v2.kernel.measurement_tpu /tmp/combo.json \
  --iters 1 --warmup 0
```

---

## Impact analysis (read-only)

```bash
# Which workloads use kernel_K=256?
tools/tuning/v2/scripts/impact.sh by-kernel-key kernel_K 256

# Which deployments use MAX_NUM_BATCHED_TOKENS=131072?
tools/tuning/v2/scripts/impact.sh by-service-combo \
  MAX_NUM_BATCHED_TOKENS 131072

# Which workloads have stale tunes?
tools/tuning/v2/scripts/impact.sh stale-tunes <current_kernel_sha>
```

---

## Verbose / debug logs

```bash
LOG_LEVEL=DEBUG \
tools/tuning/v2/scripts/run_pipeline.sh \
  tools/benchmark/cases/v7x/llama3_8b/throughput/prefill_heavy.workload
```

## Quiet logs (warnings + errors only)

```bash
LOG_LEVEL=WARNING \
tools/tuning/v2/scripts/run_pipeline.sh \
  tools/benchmark/cases/v7x/llama3_8b/throughput/prefill_heavy.workload
```

---

## Tests + coverage

```bash
python3 -m unittest discover tests.tuning_v2

python3 -m coverage run --branch --source=tools.tuning.v2 \
  -m unittest discover tests.tuning_v2 && \
  python3 -m coverage report --skip-empty
```

---

## Bi-directional lookup

Two directions, both implemented:

**Top-down** (deploy time, `workload → service → kernel → env vars`):
```bash
tools/tuning/v2/scripts/lookup.sh <workload>.workload [--objective throughput_max]
```
Reads `<workload>.service`, picks the objective winner, follows its
`kernel_pin_keys` into `<workload>.kernel`, emits the merged env-var
set as `KEY=VALUE` lines.

**Bottom-up** (impact analysis, `attribute → which workloads`):
```bash
# "I changed the kernel source. Which workloads need re-tune?"
tools/tuning/v2/scripts/impact.sh stale-tunes <new_kernel_sha>

# "Which workloads use kernel_K=256?"
tools/tuning/v2/scripts/impact.sh by-kernel-key kernel_K 256

# "Which deployments are served at MAX_NUM_BATCHED_TOKENS=131072?"
tools/tuning/v2/scripts/impact.sh by-service-combo MAX_NUM_BATCHED_TOKENS 131072
```

---

## Future work

- **Smarter search** (Helion-style surrogate + 2-pass benchmarking):
  see follow-up notes; would turn hour-long tunes into ~15 min ones
  at comparable winner quality. Helion repo cloned for reference at
  `../helion`.
- **Top-N refinement** (re-measure top winners at higher iters +
  finer-grained neighbor search): deferred.

## v1 deprecation (planned)

v1 tuner (`tools/kernel/tuner/v1/`) produces results in a format
incompatible with v2's `.kernel` schema. v1 results are NOT
auto-migrated. Re-tune everything in v2; once v2 winners are
validated against v1's recorded production tunes (sanity check —
results should agree on shared combos), v1 retires.
