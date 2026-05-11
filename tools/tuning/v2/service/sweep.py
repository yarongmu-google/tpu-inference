# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Service-sweep runner: glues search_space + measurement.

Mirror of `tools.tuning.v2.kernel.tune` for the service layer. One
function:

  run_service_sweep(*, workload_env, workload_dir, workload_name,
                    raw_path, measurement_fn, service_revision,
                    commit_every=5, on_progress=None) -> int

Enumerates combos (cartesian product of sweep axes from
`service_search_space`), skips ones already in `.service.raw/<sha>.jsonl`
with permanent status, calls `measurement_fn(combo)` for the rest,
appends result + combo as one raw row, periodic commit+push.

`measurement_fn` is the vLLM-bench boundary. Production wires it to
`vllm bench serve` (a long-running subprocess); tests inject a mock
returning deterministic metrics. The runner imports no vLLM-side code.

`commit_every` defaults to 5 (smaller than the kernel-tune's 25)
because each service combo takes minutes, not seconds — committing
every 5 keeps the remote close to up-to-date without spam.
"""

import itertools
import sys
from pathlib import Path
from typing import Any, Callable, Iterator

from tools.tuning.v2.core.git_atomic import commit_and_push
from tools.tuning.v2.core.keyset import service_combo_key
from tools.tuning.v2.core.raw_store import append_row, build_skip_set
from tools.tuning.v2.service.search_space import service_search_space


# Statuses that are permanent for resume — re-running them wastes time
# or fails identically. Mirrors PERMANENT_STATUSES in kernel/tune.py.
# Service-side statuses also distinguish OOM (FAILED_OOM) from other
# bench-tool failures (FAILED). Both are permanent here; UNKNOWN_ERROR
# is retryable.
PERMANENT_STATUSES = frozenset({"SUCCESS", "FAILED_OOM", "FAILED", "SKIPPED"})


def enumerate_service_combos(
    *,
    search_space: dict[str, list[Any]],
) -> Iterator[dict[str, Any]]:
    """Yield each cartesian-product combo of the sweep axes.

    Args:
      search_space: dict of `axis_name -> list_of_values`.

    Yields:
      One dict per combo, mapping each axis to its chosen value.
      Iteration order follows the natural product order of axis keys
      (sorted) so the output is deterministic across runs.
    """
    if not search_space:
        return
    keys = sorted(search_space.keys())
    value_lists = [search_space[k] for k in keys]
    for values in itertools.product(*value_lists):
        yield dict(zip(keys, values))


def run_service_sweep(
    *,
    workload_env: dict[str, str],
    workload_dir: Path,
    workload_name: str,
    raw_path: Path,
    measurement_fn: Callable[[dict[str, Any]], dict[str, Any]],
    service_revision: str,
    commit_every: int = 5,
    on_progress: Callable[[int], None] | None = None,
) -> int:
    """Run a service-sweep pass for one workload.

    Args:
      workload_env: parsed `.workload` env dict. Not directly used by
                    the runner, but carried for symmetry with
                    kernel/tune.run_kernel_tune so the orchestrator can
                    pass the same dict to both layers.
      workload_dir: directory containing the workload's files.
      workload_name: workload stem.
      raw_path: append-only JSONL store, typically
                `<workload_dir>/<workload_name>.service.raw/<sha>.jsonl`.
      measurement_fn: `(combo: dict) -> dict`. Must include a `status`
                      key. Should also include a `metrics` sub-dict
                      with at least `req_per_sec`, `ttft_mean_ms`,
                      `ttft_p99_ms` so the projection has fields to
                      pick winners by.
      service_revision: `<tpu_inference_sha>-<vllm_sha>` for stamping
                        rows. Unused inside the loop but recorded for
                        provenance.
      commit_every: number of NEW combos between periodic commits.
                    `<= 0` disables periodic; final commit always runs.
      on_progress: optional `(n_new) -> None` callback per row.

    Returns:
      Count of new rows appended in this run.
    """
    del workload_env  # workload-env stays in .workload; not used here
    del service_revision  # recorded externally (stamped via raw filename)
    search_space = service_search_space(workload_dir, workload_name)

    skip_set = build_skip_set(
        raw_path,
        key_fn=lambda r: service_combo_key(r["combo"]),
        status_filter=PERMANENT_STATUSES,
    )

    n_new = 0
    for combo in enumerate_service_combos(search_space=search_space):
        if service_combo_key(combo) in skip_set:
            continue
        try:
            result = measurement_fn(combo)
        except Exception as e:    # pylint: disable=broad-except
            result = {
                "status": "UNKNOWN_ERROR",
                "error": f"{type(e).__name__}: {e}",
            }
        row = {"combo": combo, **result}
        append_row(raw_path, row)
        n_new += 1
        if on_progress is not None:
            on_progress(n_new)
        if commit_every > 0 and n_new % commit_every == 0:
            commit_and_push(
                [raw_path],
                f"[Sweep-v2] progress: {n_new} combos ({workload_name})",
            )

    if n_new > 0:
        commit_and_push(
            [raw_path],
            f"[Sweep-v2] complete: {n_new} combos ({workload_name})",
        )

    return n_new


def main(argv: list[str] | None = None) -> int:
    """CLI placeholder. vLLM-bench binding is in a follow-up commit."""
    import argparse
    p = argparse.ArgumentParser(
        description="Run a service sweep for one workload.",
    )
    p.add_argument("workload", type=Path)
    p.add_argument("--commit-every", type=int, default=5)
    args = p.parse_args(argv)
    if not args.workload.exists():
        print(f"workload not found: {args.workload}", file=sys.stderr)
        return 1
    print(
        "tools.tuning.v2.service.sweep CLI: vLLM-bench binding is in "
        "a follow-up commit. The runnable today is the library "
        "function run_service_sweep(..., measurement_fn=...).",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":   # pragma: no cover
    sys.exit(main())
