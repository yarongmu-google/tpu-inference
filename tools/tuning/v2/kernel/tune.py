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

import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Iterator

from tools.tuning.v2.core.git_atomic import commit_and_push
from tools.tuning.v2.core.keyset import combo_key, worker_bucket
from tools.tuning.v2.core.raw_store import append_row, build_skip_set
from tools.tuning.v2.kernel.enumerate import enumerate_all_combos
from tools.tuning.v2.kernel.enumerate_logical import (
    DEFAULT_HARDWARE,
    DEFAULT_KERNEL_VARIANT,
    enumerate_logical_combos,
)
from tools.tuning.v2.kernel.search_space import kernel_search_space
from tools.tuning.v2.service.search_space import service_search_space

logger = logging.getLogger(__spec__.name if __spec__ is not None else __name__)


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
        = enumerate_all_combos,
    kernel_variant: str = DEFAULT_KERNEL_VARIANT,
    hardware: str = DEFAULT_HARDWARE,
    commit_every: int = 25,
    worker_id: int = 0,
    worker_count: int = 1,
    on_progress: Callable[[int], None] | None = None,
) -> int:
    """Run a kernel-tune pass for one workload + one case.

    Args:
      workload_env: parsed `.workload` env (strings; ints cast inside).
                    Must include MAX_NUM_SEQS, NUM_Q_HEADS, NUM_KV_HEADS,
                    HEAD_DIM, MAX_MODEL_LEN. MAX_NUM_BATCHED_TOKENS
                    is read from the service search space, NOT
                    workload_env (per architecture doc §2).
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
    # MAX_NUM_SEQS resolution (architecture-doc §2 line 73 + your
    # latency/throughput convention):
    #   - workload pins MAX_NUM_SEQS -> latency scenario, single MNS.
    #   - workload silent -> throughput scenario; kernel-tune uses
    #     max(service_axes.MAX_NUM_SEQS) so the tuned kernel handles
    #     the worst-case concurrency the service-sweep may pick.
    service_space = service_search_space(workload_dir, workload_name)
    if "MAX_NUM_SEQS" in workload_env and workload_env["MAX_NUM_SEQS"]:
        max_num_seqs = int(workload_env["MAX_NUM_SEQS"])
    else:
        max_num_seqs = max(service_space["MAX_NUM_SEQS"])

    search_space = kernel_search_space(
        workload_dir, workload_name,
        max_num_seqs=max_num_seqs,
    )
    model_shape = model_shape_from_workload(workload_env)

    # MAX_NUM_BATCHED_TOKENS is category 5 (service-tuned) per
    # architecture-doc §2 — it does NOT belong in .workload. The
    # kernel-tune still needs an MNB upper bound to prune combos
    # whose `per_phys_q > MNB`; source it from the service search
    # space's max (doc §2, line 83: "kernel-tune must be aware of,
    # or recomputed against, the sweep's MNB candidates"). This is
    # the coupling between layers the architecture acknowledges.
    mnb_ceiling = max(service_space["MAX_NUM_BATCHED_TOKENS"])

    skip_set = build_skip_set(
        raw_path,
        key_fn=lambda r: combo_key(r["tuning_key"], r["tunable_params"]),
        status_filter=PERMANENT_STATUSES,
    )

    combos = enumerator(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=mnb_ceiling,
        model_shape=model_shape,
        code_revision=code_revision,
        search_space=search_space,
        kernel_variant=kernel_variant,
        hardware=hardware,
    )

    n_new = 0
    # SMOKE_TEST tracking: once a case has its first SUCCESS row,
    # subsequent combos for that case are skipped. With the
    # all-cases default enumerator (D / M / P / L), smoke ends up
    # with one SUCCESS per case — enough for the projection to
    # produce all four winners and for the sweep step's
    # resolve_kernel_pin_keys to find a LOGICAL or PREFILL winner.
    succeeded_cases: set[str] = set()
    if worker_count < 1 or not (0 <= worker_id < worker_count):
        raise ValueError(
            f"invalid worker config: worker_id={worker_id} "
            f"worker_count={worker_count} — require 1 <= count and "
            f"0 <= id < count.",
        )
    for tuning_key, tunable_params in combos:
        case = tuning_key.get("case")
        if (os.environ.get("SMOKE_TEST") == "1"
                and case in succeeded_cases):
            continue
        ck = combo_key(tuning_key, tunable_params)
        if ck in skip_set:
            continue
        # Multi-worker partitioning (arch §8): hash-based bucket
        # assignment. Each worker measures only its own bucket;
        # workers append to the SAME .raw file so the projection
        # sees all results regardless of which worker produced them.
        # `worker_count=1` (default) keeps every combo for the
        # single-worker case.
        if worker_count > 1 and worker_bucket(ck, worker_count) != worker_id:
            continue

        # Pre-measurement progress line so a hung combo is visible.
        # The post-measurement line confirms completion + status; this
        # one names the combo BEFORE we call into pallas_call, so a
        # multi-minute JIT compile doesn't look like a hang.
        worker_tag = (f"w{worker_id}/{worker_count} "
                      if worker_count > 1 else "")
        logger.info(
            "[tune %s  *] case=%-7s page=%-3s K=%-5s mnss=%-6s "
            "bq=%-5s → measuring...",
            worker_tag,
            tuning_key.get("case"),
            tuning_key.get("page_size"),
            tuning_key.get("kernel_K"),
            tunable_params.get("mnss"),
            tunable_params.get("bq_sz"),
        )

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
            # Dict-literal precedence (same honesty as fix #14): the
            # `status` and `error` keys below land AFTER the **result
            # spread, so they unconditionally overwrite any same-named
            # keys that came back from measurement_fn. The `.get("error",
            # ...)` only matters if `error` was NOT in result; if it was,
            # the explicit "error" key wins.
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

        # Post-measurement progress line — see the pre-measurement
        # block above. Raw JSONL stays the source of truth.
        _status = result.get("status", "?")
        _summary = (
            f"latency_us={result['latency_us']:.1f}"
            if _status == "SUCCESS" and "latency_us" in result
            else (
                f"error={result.get('error', '?')[:60]}"
                if "error" in result else ""
            )
        )
        logger.info(
            "[tune %s%4d] case=%-7s page=%-3s K=%-5s mnss=%-6s "
            "bq=%-5s → %-13s %s",
            worker_tag,
            n_new,
            tuning_key.get("case"),
            tuning_key.get("page_size"),
            tuning_key.get("kernel_K"),
            tunable_params.get("mnss"),
            tunable_params.get("bq_sz"),
            _status,
            _summary,
        )

        if on_progress is not None:
            on_progress(n_new)
        if commit_every > 0 and n_new % commit_every == 0:
            commit_and_push(
                [raw_path],
                f"[Tune-v2] progress: {n_new} cases ({workload_name})",
            )

        # SMOKE_TEST=1: register this case as succeeded so the
        # pre-loop skip drops the rest of this case's combos. The
        # next NEW case proceeds normally. With the all-cases
        # default enumerator, smoke produces one SUCCESS row per
        # case (D / M / P / L) — enough for full pipeline flow.
        if (os.environ.get("SMOKE_TEST") == "1"
                and result.get("status") == "SUCCESS"):
            succeeded_cases.add(case)

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
    """CLI entry. Reads `.workload`, runs the tune with the rpa_v3
    TPU measurement function (lazy import keeps this module
    import-able on non-TPU hosts).

    Workload env vars are pushed into `os.environ` BEFORE the
    measurement_fn is built, because v1's RpaV3KernelTuner reads
    them at construction time (MAX_NUM_SEQS, MAX_NUM_BATCHED_TOKENS,
    NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, MAX_MODEL_LEN, ...).
    """
    import argparse
    p = argparse.ArgumentParser(
        description="Run a kernel tune for one workload + one case.",
    )
    p.add_argument("workload", type=Path,
                   help="Path to a `.workload` file.")
    p.add_argument("--commit-every", type=int, default=25,
                   help="Commit raw store every N cases.")
    p.add_argument("--iters", type=int, default=10,
                   help="Timed iterations per combo.")
    p.add_argument("--warmup", type=int, default=2,
                   help="Untimed warmup iterations per combo. 0 to skip.")
    p.add_argument(
        "--worker-id", type=int, default=0,
        help="This worker's bucket index in [0, --worker-count). "
             "Used for distributed tuning across multiple TPU VMs; "
             "every worker measures only the combos whose stable "
             "hash matches its bucket. Workers share one .raw file.",
    )
    p.add_argument(
        "--worker-count", type=int, default=1,
        help="Total number of workers participating. Default 1 "
             "(single-worker, no partitioning).",
    )
    args = p.parse_args(argv)

    from tools.tuning.v2.core.logs import configure as configure_logging
    configure_logging()

    if not args.workload.exists():
        logger.error("workload file not found: %s", args.workload)
        return 1

    workload_env = _parse_workload_env(args.workload)
    workload_dir = args.workload.parent
    workload_name = args.workload.stem

    # Push workload env into the process environment so the v1
    # RpaV3KernelTuner can read NUM_Q_HEADS / HEAD_DIM / etc.
    # from os.environ at construction time.
    import os as _os
    _os.environ.update(workload_env)

    # Neither MAX_NUM_BATCHED_TOKENS nor MAX_NUM_SEQS (when omitted
    # for throughput scenarios) live in .workload per architecture-
    # doc §2. The v1 tuner reads BOTH from os.environ; supply them
    # from the service search space's max so the kernel layer's
    # pruning aligns with the sweep's upper bound. (Cross-layer
    # coupling §2 line 83.)
    from tools.tuning.v2.service.search_space import service_search_space
    _service_space = service_search_space(workload_dir, workload_name)
    _os.environ["MAX_NUM_BATCHED_TOKENS"] = str(
        max(_service_space["MAX_NUM_BATCHED_TOKENS"]),
    )
    if "MAX_NUM_SEQS" not in workload_env or not workload_env["MAX_NUM_SEQS"]:
        # Throughput scenario: tune at the worst-case MNS the
        # service may sweep. Latency scenarios pin MNS in .workload
        # explicitly and we just inherit that.
        _os.environ["MAX_NUM_SEQS"] = str(
            max(_service_space["MAX_NUM_SEQS"]),
        )

    # Lazy imports — only at CLI time, only when actually running on
    # TPU. Tests use `run_kernel_tune` directly with a mocked
    # measurement_fn and never hit this path.
    from tools.tuning.v2.core.sha import kernel_sha
    from tools.tuning.v2.kernel.measurement_tpu import make_measurement_fn

    code_revision = kernel_sha()
    raw_path = (
        workload_dir / f"{workload_name}.kernel.raw" /
        f"{code_revision}.jsonl"
    )

    measurement_fn = make_measurement_fn(
        iters=args.iters, warmup_iters=args.warmup,
    )

    n_new = run_kernel_tune(
        workload_env=workload_env,
        workload_dir=workload_dir,
        workload_name=workload_name,
        raw_path=raw_path,
        measurement_fn=measurement_fn,
        code_revision=code_revision,
        commit_every=args.commit_every,
        worker_id=args.worker_id,
        worker_count=args.worker_count,
    )
    logger.info("Tune-v2: %d new rows written to %s", n_new, raw_path)
    # The path is the machine-parseable result — keep on stdout for
    # shell-wrapper consumption (no timestamp).
    print(raw_path)
    return 0


if __name__ == "__main__":   # pragma: no cover
    sys.exit(main())
