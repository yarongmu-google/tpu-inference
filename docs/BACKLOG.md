# Backlog

Things we want to do but haven't yet. Add new items at the bottom of
the relevant section with a date stamp. Mark done with `~~strikethrough~~`
+ a `→ <commit-sha>` link, or just delete after the work lands.

Loose priority convention:
- **P0** = blocking the next sweep / debug session
- **P1** = unblocks a known throughput / correctness gain we've measured or modelled
- **P2** = quality-of-life / refactor / nice-to-have

---

## Pipeline / sweep correctness

- **P1 (2026-05-16)** Add `MAX_NUM_BATCHED_TOKENS=475136` to the throughput
  sweep recipe. Just verified to fit at `GPU_MEM_UTIL=0.77` (zero KV-cache
  margin, 0.86 G XLA margin). Pushes single-chip ceiling from 56 →
  58 prompts/call (~3.5% throughput at the largest MNB band).
  Touch: `tools/benchmark/sweep_recipes.py::_throughput_coupled_axes()`.

- **P1 (2026-05-16)** Pattern-match XLA HBM OOM in `sweep.py` and short-circuit
  larger combos. When a combo dies with `RESOURCE_EXHAUSTED` /
  `CompileTimeHbmOom` / `RuntimeProgramAllocationFailure`, mark that
  `(MNB, util)` failed and skip every subsequent combo at higher MNB with
  ≥ that util (same MNS). Saves ~5-10 min per skipped combo (engine init
  time before OOM). Today the sweep wastes time on doomed combos.

- **P1 (2026-05-16)** Throughput / latency pipeline split — two parallel
  `tools/run_pipeline_{throughput,latency}.sh` scripts sharing common
  helpers, instead of one mode-flagged script. Already captured in
  `memory/project_pipeline_split_latency_throughput.md`. The benchmark
  recipe + run_benchmark.sh changes today only address the throughput path.

---

## Documentation / methodology

- **P1 (2026-05-16)** Three-artifact "HBM capacity estimation" methodology
  for future agents:
  1. `docs/hbm_capacity_estimation.md` — the methodology doc.
     Conceptual two-pool model + formula + calibration recipe + worked
     Llama-3-8B/v7x-1 example. Format optimized for an LLM dropped into
     a new (model, hardware, TP) configuration to apply unchanged.
  2. `tools/benchmark/calibrate_hbm.py` — runnable script that performs
     the 4 debug invocations and fits the per-(model, hardware, TP)
     coefficients. Outputs the calibrated constants ready to paste into
     `sweep_recipes.py::estimate_gpu_memory_utilization`'s defaults.
  3. 1-line docstring pointer in
     `tools/benchmark/sweep_recipes.py::estimate_gpu_memory_utilization`
     to the doc above.
  See the 2026-05-16 conversation that produced the formula
  (calibrated against 4 verified MNB points on Llama 3 8B / v7x-1 / TP=1).

---

## Refactors

- **P2 (2026-05-16)** Convert `.sh` orchestration scripts to Python.
  `tools/debug_vllm_engine_init.sh` (~280 lines, complex) first; then
  `tools/benchmark/run_benchmark.sh` (~370 lines); then
  `tools/run_pipeline.sh` (~370 lines). Reasons captured in the
  2026-05-16 conversation: bash 3.2/4 syntax issues, `set -u` array
  quirks, quoting hell, hard to unit-test, swallowed-stderr defensive
  silencing. Python versions get subprocess.run + argparse + pytest
  per-function tests. Sweep.sh, build_*_registry.sh, tune_all_cases.sh
  are thin enough to stay bash.

---

## Performance investigations (open questions)

- **P2 (2026-05-16)** Investigate XLA `tpu_shared_memory_percent` flag
  (or whatever its actual libtpu name is — needs verification). Hypothesis:
  the persistent ~2-3 GiB "mystery overhead" we observed across 4 debug
  runs may be partly XLA shared-memory reservation. If we can shrink it,
  we recover headroom for either more KV cache or larger MNB at the same
  util. Mechanism: set `LIBTPU_INIT_ARGS=--xla_tpu_shared_memory_percent=N`
  (env_override.py already prepends, so it composes). See
  https://openxla.org/xla/errors/error_0101 for the doc that mentions
  this knob.

- **P2 (2026-05-16)** TP > 1 for MNB > 475136 (the single-chip ceiling).
  The kernel theoretically supports 132 prompts/call (= MNB=1,081,344),
  but the HBM math says total need ≈ 193 GiB at that scale vs 94.75 GiB
  per chip. TP > 1 shards weights AND KV cache across chips and is the
  only path past 475136 on this hardware.

- **P1 (2026-05-16)** Verify actual throughput at the unlocked MNBs vs
  the linear prediction. Today's prediction: 4.9 → ~17 req/s at MNB=458752
  (~3.5×). Could be wrong if kernel doesn't actually scale linearly per
  prompt (memory bandwidth saturation, scheduler overhead, MXU utilisation).
  Tonight's sweep is the experiment; result tells us if the linear model
  holds.

- **P2 (2026-05-12)** Re-bench best production.service config with
  `--save-detailed` and plot the TTFT CDF via `plot_ttft_cdf.py`. Captured
  in `memory/project_pending_ttft_cdf_verify.md`. Untouched since the
  latency vs throughput regime split surfaced.

---

## Logs / data hygiene

- **P2 (2026-05-16)** Backfill `*.log` files from the debug runs that
  happened BEFORE the `git add -f` fix (commit `aeb3457c`). Several
  per-run dirs under `tmp/debug_vllm_engine_init/` have committed
  `meta.txt` + `failure.txt` but the `vllm.log` and `script.log` are
  only on local disk (silently skipped by `git add` due to the global
  `*.log` gitignore rule). Recoverable via
  `find tmp/debug_vllm_engine_init -name "*.log" | xargs git add -f`
  and a single backfill commit. Cosmetic — newer runs are fine.

- **P2 (2026-05-16)** Set `SKIP_COMMIT_CACHE_CHECK=1` as a default in
  `tools/run_pipeline.sh` (or its successor). The kernel-tuner's
  resume cache currently invalidates on any tpu-inference commit, even
  documentation-only ones. Captured in
  `memory/project_kernel_tuner_durability.md` and the related project
  memory. Without this, every doc edit forces a re-tune of the
  resumed buckets.
