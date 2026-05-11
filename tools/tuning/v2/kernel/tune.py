# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Kernel-tune runner: glues search_space + enumerator + measurement.

Standalone entry point for the kernel layer. The `tune_kernel.sh`
script (sibling, scripts/) invokes this via `python3 -m`.

Architecture:

   workload_env  ─┐
   workload_dir  ─┼─▶ kernel_search_space  ─▶  search_space dict
   workload_name ─┘                                   │
                                                      ▼
   workload + search_space ───▶  enumerator  ───▶  (tk, tp) stream
                                                      │
                                                      ▼
   skip_set ◀── raw_store.build_skip_set ◀──  prior .raw/<sha>.jsonl
                                                      │
                                                      ▼
              filtered stream  ───▶  measurement_fn  ───▶  metric dict
                                                      │
                                                      ▼
              raw_store.append_row(.raw/<sha>.jsonl)
                                                      │
                                  every commit_every cases
                                                      ▼
              git_atomic.commit_and_push(.raw)

`measurement_fn` is the TPU boundary. Production wires it to a
`pallas_call` invocation; tests inject a mock that returns
deterministic values without touching JAX/TPU. The runner itself
imports zero TPU-side code.
"""

import sys
from pathlib import Path
from typing import Any, Callable, Iterator

from tools.tuning.v2.core.git_atomic import commit_and_push
from tools.tuning.v2.core.keyset import combo_key
from tools.tuning.v2.core.raw_store import append_row, build_skip_set
from tools.tuning.v2.kernel.enumerate_logical import enumerate_logical_combos
from tools.tuning.v2.kernel.search_space import kernel_search_space


# Statuses that are permanent for resume — i.e., re-running them is a
# waste of time or will fail identically. UNKNOWN_ERROR is intentionally
# excluded so post-bugfix re-runs retry it (the regression that
# 35b570d7 fixed in v1).
PERMANENT_STATUSES = frozenset({"SUCCESS", "FAILED_OOM", "SKIPPED"})


def model_shape_from_workload(workload_env: dict[str, str]) -> dict[str, Any]:
    """Extract the model-shape sub-dict from a workload env dict.

    Args:
      workload_env: dict of `VAR -> value`, typically parsed from the
                    `.workload` bash file. String values; ints cast here.

    Returns:
      Dict with `num_q_heads`, `num_kv_heads`, `head_dim`,
      `max_model_len`, `q_dtype`, `kv_dtype`, `sliding_window`.
    """
    sw = workload_env.get("SLIDING_WINDOW", "")
    return {
        "num_q_heads":    int(workload_env["NUM_Q_HEADS"]),
        "num_kv_heads":   int(workload_env["NUM_KV_HEADS"]),
        "head_dim":       int(workload_env["HEAD_DIM"]),
        "max_model_len":  int(workload_env["MAX_MODEL_LEN"]),
        "q_dtype":        workload_env.get("Q_DTYPE", "bfloat16"),
        "kv_dtype":       workload_env.get("KV_DTYPE", "bfloat16"),
        "sliding_window": int(sw) if sw else None,
    }


def run_kernel_tune(
    *,
    workload_env: dict[str, str],
    workload_dir: Path,
    workload_name: str,
    raw_path: Path,
    measurement_fn: Callable[[dict[str, Any], dict[str, Any]],
                             dict[str, Any]],
    code_revision: str,
    enumerator: Callable[..., Iterator[tuple[dict, dict]]]
        = enumerate_logical_combos,
    commit_every: int = 25,
    on_progress: Callable[[int], None] | None = None,
) -> int:
    """Run a kernel-tune pass for one workload + one case.

    Args:
      workload_env: parsed `.workload` env (strings; ints cast inside).
                    Must include MAX_NUM_SEQS, MAX_NUM_BATCHED_TOKENS,
                    NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, MAX_MODEL_LEN.
      workload_dir: directory containing the `.workload` and the
                    optional `<workload>.kernel_axes.json` overlay.
      workload_name: workload stem (e.g. `"prefill_heavy"`).
      raw_path: append-only JSONL store, typically
                `<workload_dir>/<workload_name>.kernel.raw/<sha>.jsonl`.
      measurement_fn: `(tuning_key, tunable_params) -> dict`. The
                      result dict is merged with tuning_key +
                      tunable_params and appended as one raw row.
                      Must include a `status` key. Exceptions are
                      caught and recorded as a status=UNKNOWN_ERROR
                      row so the loop survives transient failures.
      code_revision: kernel-source SHA (8-char) to stamp on each
                     tuning_key. Use `core.sha.kernel_sha()`.
      enumerator: callable matching `enumerate_logical_combos`'s
                  signature. Defaults to LOGICAL; pluggable for other
                  cases.
      commit_every: number of NEW rows between periodic
                    `git_atomic.commit_and_push` calls.
                    `<= 0` disables periodic commits (still does a
                    final commit at the end).
      on_progress: optional callback `(n_new) -> None`. Called once
                   per appended row. Tests use this for assertions
                   without parsing the raw file.

    Returns:
      Count of new rows appended in this run (excludes skipped /
      already-done from previous runs).
    """
    search_space = kernel_search_space(
        workload_dir, workload_name,
        max_num_seqs=int(workload_env["MAX_NUM_SEQS"]),
    )
    model_shape = model_shape_from_workload(workload_env)

    skip_set = build_skip_set(
        raw_path,
        key_fn=lambda r: combo_key(r["tuning_key"], r["tunable_params"]),
        status_filter=PERMANENT_STATUSES,
    )

    combos = enumerator(
        max_num_seqs=int(workload_env["MAX_NUM_SEQS"]),
        max_num_batched_tokens=int(workload_env["MAX_NUM_BATCHED_TOKENS"]),
        model_shape=model_shape,
        code_revision=code_revision,
        search_space=search_space,
    )

    n_new = 0
    for tuning_key, tunable_params in combos:
        if combo_key(tuning_key, tunable_params) in skip_set:
            continue
        try:
            result = measurement_fn(tuning_key, tunable_params)
        except Exception as e:    # pylint: disable=broad-except
            result = {
                "status": "UNKNOWN_ERROR",
                "error": f"{type(e).__name__}: {e}",
            }
        # Defensive (fix #24 + followup): a buggy measurement_fn
        # returning None / non-dict / partial dict would either crash
        # the `**result` spread or land rows whose status is missing
        # or None — those wedge the skip-set (no status match against
        # PERMANENT_STATUSES means resume re-attempts them forever).
        # Coerce all three failure modes to UNKNOWN_ERROR so resume
        # treats them as retryable and the operator sees a typed
        # error.
        if not isinstance(result, dict):
            result = {
                "status": "UNKNOWN_ERROR",
                "error":  f"measurement_fn returned non-dict: "
                          f"{type(result).__name__}",
            }
        elif result.get("status") is None:
            result = {
                **result,
                "status": "UNKNOWN_ERROR",
                "error":  result.get("error",
                                     "measurement_fn returned dict "
                                     "with missing or None status"),
            }
        row = {
            "tuning_key":     tuning_key,
            "tunable_params": tunable_params,
            **result,
        }
        append_row(raw_path, row)
        n_new += 1
        if on_progress is not None:
            on_progress(n_new)
        if commit_every > 0 and n_new % commit_every == 0:
            commit_and_push(
                [raw_path],
                f"[Tune-v2] progress: {n_new} cases ({workload_name})",
            )

    # Final commit fires only if the last batch of rows wasn't already
    # picked up by a periodic commit. When n_new is an exact multiple
    # of commit_every, the periodic at the boundary already covered
    # everything — a second commit here would be a no-op empty diff
    # but a noisy git log entry. (fix #10)
    if n_new > 0 and (commit_every <= 0 or n_new % commit_every != 0):
        commit_and_push(
            [raw_path],
            f"[Tune-v2] complete: {n_new} cases ({workload_name})",
        )

    return n_new


# ---------------------------------------------------------------------
# CLI entry — usable via `python3 -m tools.tuning.v2.kernel.tune ...`
# ---------------------------------------------------------------------

def _parse_workload_env(workload_path: Path) -> dict[str, str]:
    """Parse a `.workload` bash file into a dict.

    Uses bash's own substitution semantics: sources the file in a
    sub-shell with `set -a`, then prints `env`. Tolerant to comments
    and `: "${VAR:=default}"` patterns.
    """
    import subprocess
    proc = subprocess.run(
        ["bash", "-c", f"set -a; source '{workload_path}'; env"],
        capture_output=True, text=True, check=True,
    )
    env: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            env[k] = v
    return env


def main(argv: list[str] | None = None) -> int:
    """CLI entry. Reads `.workload`, runs the tune with a real TPU
    measurement function (imported lazily so this module's import
    surface stays TPU-free)."""
    import argparse
    p = argparse.ArgumentParser(
        description="Run a kernel tune for one workload + one case.",
    )
    p.add_argument("workload", type=Path,
                   help="Path to a `.workload` file.")
    p.add_argument("--commit-every", type=int, default=25,
                   help="Commit raw store every N cases.")
    args = p.parse_args(argv)

    if not args.workload.exists():
        print(f"workload file not found: {args.workload}", file=sys.stderr)
        return 1

    workload_env = _parse_workload_env(args.workload)
    workload_dir = args.workload.parent
    workload_name = args.workload.stem

    # Lazy TPU-side import; only at CLI time.
    from tools.tuning.v2.core.sha import kernel_sha
    # The default production measurement function would invoke the
    # kernel via pallas_call. That binding lives in a separate module
    # (kernel/measurement_tpu.py) added in a later commit alongside
    # the actual JAX/Pallas wiring. For now, the CLI is a placeholder
    # that requires the caller to pass a measurement function via
    # injection — covered by tests, not by `python3 -m`.
    print(
        "tools.tuning.v2.kernel.tune CLI: TPU measurement binding is "
        "in a follow-up commit. The runnable today is the library "
        "function `run_kernel_tune(..., measurement_fn=...)`.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":   # pragma: no cover
    sys.exit(main())
