# Tune-v2 — copy-paste command recipes

All commands assume `cwd = repo root`. Pull first if anything looks stale:

```bash
git pull
```

---

## Smoke tests (no commits, no real bench, single combo)

Both scenarios share the same prelude. `KERNEL_TUNER_NO_COMMIT=1` keeps the
working tree clean; `MOCK_BENCH=1` short-circuits the sweep step to synthetic
metrics so no vLLM server is needed; `SMOKE_TEST=1` makes the runner stop at
the first SUCCESS row (the search space is unchanged — earlier "truncate to
one combo" picked infeasible configs and dead-ended the pipeline);
`EXTRA_TUNE_FLAGS="--iters 1 --warmup 0"` makes each TPU measurement fast.

Under SMOKE_TEST, the kernel-tune may write a handful of SKIPPED rows
before landing a SUCCESS — that's the SMEM/VMEM estimator correctly
rejecting some combos. The first SUCCESS is what the rest of the pipeline
consumes.

### Throughput (MNS swept at service level)

```bash
SMOKE_TEST=1 KERNEL_TUNER_NO_COMMIT=1 MOCK_BENCH=1 \
EXTRA_TUNE_FLAGS="--iters 1 --warmup 0" \
tools/tuning/v2/scripts/run_pipeline.sh \
  tools/benchmark/cases/v7x/llama3_8b/throughput/prefill_heavy.workload
```

### Latency (MNS=1 pinned)

```bash
SMOKE_TEST=1 KERNEL_TUNER_NO_COMMIT=1 MOCK_BENCH=1 \
EXTRA_TUNE_FLAGS="--iters 1 --warmup 0" \
tools/tuning/v2/scripts/run_pipeline.sh \
  tools/benchmark/cases/v7x/llama3_8b/latency/prefill_heavy.workload
```

Expected output per scenario: six step banners ending in
`=== Tune-v2 pipeline complete ===`. Files produced (relative to the
workload's parent dir):

- `prefill_heavy.kernel.raw/<sha>.jsonl` — one row, real TPU latency.
- `prefill_heavy.kernel` — one winner (`case: logical`).
- `prefill_heavy.service.raw/<sha>.jsonl` — one row, `mock: true`.
- `prefill_heavy.service` — three objective winners (throughput_max,
  ttft_min, p99_min) all pointing at the synthetic combo.
- `production.kernel` + `production.service` — aggregated envelope.

---

## Individual steps (advance one stage at a time)

```bash
# Step 1: validate the workload schema.
tools/tuning/v2/scripts/validate.sh \
  tools/benchmark/cases/v7x/llama3_8b/throughput/prefill_heavy.workload

# Step 2: kernel tune (real TPU). Add SMOKE_TEST=1 for one combo.
SMOKE_TEST=1 KERNEL_TUNER_NO_COMMIT=1 \
tools/tuning/v2/scripts/tune_kernel.sh \
  tools/benchmark/cases/v7x/llama3_8b/throughput/prefill_heavy.workload \
  --iters 1 --warmup 0

# Step 3: project kernel raw → .kernel.
KERNEL_TUNER_NO_COMMIT=1 \
tools/tuning/v2/scripts/project_kernel.sh \
  tools/benchmark/cases/v7x/llama3_8b/throughput/prefill_heavy.workload

# Step 4: service sweep (real vLLM bench). Add MOCK_BENCH=1 for synthetic.
SMOKE_TEST=1 KERNEL_TUNER_NO_COMMIT=1 MOCK_BENCH=1 \
tools/tuning/v2/scripts/sweep_service.sh \
  tools/benchmark/cases/v7x/llama3_8b/throughput/prefill_heavy.workload

# Step 5: project service raw → .service.
KERNEL_TUNER_NO_COMMIT=1 \
tools/tuning/v2/scripts/project_service.sh \
  tools/benchmark/cases/v7x/llama3_8b/throughput/prefill_heavy.workload

# Step 6: aggregate per-workload winners into production.{kernel,service}.
KERNEL_TUNER_NO_COMMIT=1 \
tools/tuning/v2/scripts/aggregate.sh \
  tools/benchmark/cases/v7x/llama3_8b/throughput
```

### Resume from a specific step

`run_pipeline.sh` accepts `--from <step>` (skips earlier steps that already
succeeded). Valid step names: `validate`, `tune_kernel`, `project_kernel`,
`sweep_service`, `project_service`, `aggregate`.

```bash
# Re-do everything from project_kernel onward (kernel tune already finished).
KERNEL_TUNER_NO_COMMIT=1 MOCK_BENCH=1 \
tools/tuning/v2/scripts/run_pipeline.sh --from project_kernel \
  tools/benchmark/cases/v7x/llama3_8b/throughput/prefill_heavy.workload
```

---

## Single-combo TPU smoke (kernel only, no pipeline)

Useful for "is the TPU env alive" sanity-check without setting up
workload+overlay files.

```bash
# Write a combo JSON, then run measurement_tpu.main directly.
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

Expected: a JSON dict on stdout with `"status": "SUCCESS"` and a
`"latency_us"` field. Exit 0.

---

## Impact-analysis queries (read-only, no TPU)

```bash
# "Which workloads use kernel_K=256?"
tools/tuning/v2/scripts/impact.sh by-kernel-key kernel_K 256

# "Which deployments are served by MAX_NUM_BATCHED_TOKENS=131072?"
tools/tuning/v2/scripts/impact.sh by-service-combo \
  MAX_NUM_BATCHED_TOKENS 131072

# "I changed the kernel source. Which workloads have stale tunes?"
tools/tuning/v2/scripts/impact.sh stale-tunes <new_kernel_sha>
```

---

## Production retune (full sweep, real bench, commits + pushes enabled)

⚠️ Drops the `NO_COMMIT` and `MOCK_BENCH` flags. Real bench server stands up
per combo; full default search spaces enumerate; raw stores commit + push
periodically.

```bash
tools/tuning/v2/scripts/run_pipeline.sh \
  tools/benchmark/cases/v7x/llama3_8b/throughput/prefill_heavy.workload
```

To skip the push but keep local commits, set `KERNEL_TUNER_NO_PUSH=1`.

---

## Tests + coverage (laptop, no TPU)

```bash
python3 -m unittest discover tests.tuning_v2 -v
python3 -m coverage run --branch --source=tools.tuning.v2 \
  -m unittest discover tests.tuning_v2 && \
  python3 -m coverage report --skip-empty
```
