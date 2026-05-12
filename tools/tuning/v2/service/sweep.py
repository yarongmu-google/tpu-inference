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
                    kernel_pin_keys=None, commit_every=5,
                    on_progress=None) -> int

Enumerates combos (cartesian product of sweep axes from
`service_search_space`), skips ones already in `.service.raw/<sha>.jsonl`
with permanent status, calls `measurement_fn(combo)` for the rest,
appends result + combo + kernel_pin_keys as one raw row, periodic
commit+push.

**kernel_pin_keys.** Every raw row carries the kernel-tune identity
that the service relied on. This is the load-bearing field for deploy
lookup: at deploy time, `.service.winners[objective].kernel_pin_keys`
identifies which `.kernel` winner was actually used during the sweep
(architecture doc §6 + §7, "Pin keys" Glossary). Without it, the
deploy lookup falls back to "first kernel winner per case" — which
silently ships wrong pins if the kernel was re-tuned at a different
shape between sweep and deploy.

The runner discovers pin_keys by reading `<workload>.kernel` at
startup, picking the LOGICAL winner (decoupled-K) or falling back to
the PREFILL winner (coupled-K). If neither exists, the runner raises
KernelRegistryMissingError so the operator sees a loud failure
instead of a sweep that silently records FAILED rows for every combo.

Callers may inject `kernel_pin_keys=dict(...)` explicitly to bypass
the .kernel read (tests use this).

`measurement_fn` is the vLLM-bench boundary. Production wires it to
`vllm bench serve` (a long-running subprocess); tests inject a mock
returning deterministic metrics. The runner imports no vLLM-side code.

`commit_every` defaults to 5 (smaller than the kernel-tune's 25)
because each service combo takes minutes, not seconds — committing
every 5 keeps the remote close to up-to-date without spam.
"""

import itertools
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Iterator

from tools.tuning.v2.core.discriminator import (
    DEFAULT_HARDWARE,
    DEFAULT_KERNEL_VARIANT,
    KNOWN_KERNEL_VARIANTS,
)
from tools.tuning.v2.core.git_atomic import commit_and_push
from tools.tuning.v2.core.keyset import service_combo_key, worker_bucket
from tools.tuning.v2.core.raw_store import append_row, build_skip_set
from tools.tuning.v2.service.search_space import service_search_space

logger = logging.getLogger(__spec__.name if __spec__ is not None else __name__)


# Statuses that are permanent for resume — re-running them wastes time
# or fails identically. Mirrors PERMANENT_STATUSES in kernel/tune.py.
# Service-side statuses also distinguish OOM (FAILED_OOM) from other
# bench-tool failures (FAILED). Both are permanent here; UNKNOWN_ERROR
# is retryable.
PERMANENT_STATUSES = frozenset({"SUCCESS", "FAILED_OOM", "FAILED", "SKIPPED"})


def _is_feasible(
    combo: dict[str, Any],
    workload_env: dict[str, str],
    kernel_pin_keys: dict[str, Any] | None = None,
) -> tuple[bool, str | None]:
    """Pre-filter check for service-sweep combos (fix #7 + followup).

    Returns `(feasible, reason)`. Feasible means the combo COULD plausibly
    produce a measurement; reason is the explanation we stamp on the
    SKIPPED row when it can't.

    Rules (each must deterministically fail to qualify):
      - `MAX_NUM_BATCHED_TOKENS < INPUT_LEN`: a single prompt's prefill
        won't fit in one batch slot without chunked prefill (which v2
        sweeps don't enable). vLLM will reject the request.
      - `MAX_NUM_SEQS > kernel_pin_keys.mnss`: the kernel was tuned
        with a fixed iter-slot capacity (mnss = mns_tune × (M+1)).
        Sweeping a service MNS that exceeds that capacity means there
        are more concurrent sequences than slots — out-of-bounds at
        runtime. Only checked when pin_keys is provided AND carries
        mnss (resolve_kernel_pin_keys always populates it now).

    We deliberately do NOT add a too-high MNB ceiling — Phase 5 of
    rpa3_2 missed the MNB=131,072 sweet spot because the search axis
    capped at 65,536, so we'd rather waste a few combos at the top
    than risk re-introducing that bug.
    """
    mnb_raw = combo.get("MAX_NUM_BATCHED_TOKENS")
    inp_raw = workload_env.get("INPUT_LEN")
    if mnb_raw is not None and inp_raw:
        try:
            mnb = int(mnb_raw)
            inp = int(inp_raw)
        except (TypeError, ValueError):
            mnb = inp = None    # treat as unfilterable
        if mnb is not None and inp is not None and mnb < inp:
            return False, (
                f"MAX_NUM_BATCHED_TOKENS={mnb} < INPUT_LEN={inp}; "
                f"vLLM will reject single-prompt prefill at this MNB."
            )

    mns_raw = combo.get("MAX_NUM_SEQS")
    pin_mnss = (kernel_pin_keys or {}).get("mnss")
    if mns_raw is not None and pin_mnss is not None:
        try:
            mns = int(mns_raw)
            mnss = int(pin_mnss)
        except (TypeError, ValueError):
            mns = mnss = None
        if mns is not None and mnss is not None and mns > mnss:
            return False, (
                f"MAX_NUM_SEQS={mns} > kernel iter capacity mnss="
                f"{mnss} (from pinned kernel winner); insufficient "
                f"slots for the swept concurrency."
            )
    return True, None


class KernelRegistryMissingError(RuntimeError):
    """Raised at sweep startup when the workload's `.kernel` file is
    missing or has no LOGICAL/PREFILL winner. Without one we can't
    stamp `kernel_pin_keys` on raw rows, and a sweep that runs without
    pin_keys silently breaks the deploy-time lookup contract."""


def resolve_kernel_pin_keys(
    workload_dir: Path,
    workload_name: str,
) -> dict[str, Any]:
    """Read `<workload>.kernel` and extract the pin_keys.

    Prefers the LOGICAL case (decoupled-K) over PREFILL (coupled-K).
    Pin keys returned: `case`, `page_size`, `kernel_K`,
    `code_revision`, `mnss`. The `mnss` field is the kernel's
    iter-slot capacity — for LOGICAL it comes from the chosen
    winner's tunable_params; for PREFILL (coupled-K, no
    decoupling) it equals max_num_seqs from the tuning_key. The
    service-sweep pre-filter uses it to drop infeasible
    high-concurrency combos.

    Raises:
      KernelRegistryMissingError if the file doesn't exist or has no
      LOGICAL/PREFILL winner.
    """
    kernel_path = workload_dir / f"{workload_name}.kernel"
    if not kernel_path.exists():
        raise KernelRegistryMissingError(
            f"no kernel registry at {kernel_path}; run "
            f"project_kernel.sh first (or pass kernel_pin_keys "
            f"explicitly to run_service_sweep).",
        )
    with open(kernel_path, "r", encoding="utf-8") as f:
        kernel_doc = json.load(f)

    logical = None
    prefill = None
    for w in kernel_doc.get("winners", []):
        case = w.get("tuning_key", {}).get("case")
        if case == "logical" and logical is None:
            logical = w
        elif case == "prefill" and prefill is None:
            prefill = w
    chosen = logical if logical is not None else prefill
    if chosen is None:
        # Surface the most-likely root cause: an empty winners list
        # means the kernel-tune produced zero SUCCESS rows. Tell the
        # operator where to look (the .kernel.raw partition) rather
        # than just naming the absent winners.
        n_winners = len(kernel_doc.get("winners", []))
        if n_winners == 0:
            raise KernelRegistryMissingError(
                f"{kernel_path} has zero winners (n_winners=0). The "
                f"kernel-tune produced no SUCCESS rows — every "
                f"measurement was SKIPPED / FAILED_OOM / UNKNOWN_ERROR. "
                f"Inspect {kernel_path.parent}/"
                f"{kernel_path.stem}.kernel.raw/<sha>.jsonl rows for "
                f"their `status` (and `error` if present) to "
                f"diagnose. Common causes: SMEM/VMEM estimator above "
                f"limit (mnss too large), or pallas_call failure.",
            )
        cases = sorted({
            (w.get("tuning_key") or {}).get("case") for w in
            kernel_doc.get("winners", [])
        })
        raise KernelRegistryMissingError(
            f"no LOGICAL or PREFILL winner in {kernel_path}; only "
            f"cases present: {cases}. Service sweep cannot resolve "
            f"kernel_pin_keys without one of the chunked-prefill "
            f"cases. (The pipeline currently only enumerates LOGICAL "
            f"— D/M/P enumerators not yet implemented.)",
        )
    tk = chosen["tuning_key"]
    tp = chosen.get("tunable_params", {})
    # LOGICAL winners carry mnss in tunable_params (decoupled-K iter
    # capacity); PREFILL winners are coupled-K so mnss == mns.
    if tk.get("case") == "logical":
        mnss = tp.get("mnss")
    else:
        mnss = tk.get("max_num_seqs")
    # Discriminators (architecture doc §13.4.1): forward-compat
    # defaults for older .kernel files predating the stamp — missing
    # is tolerable history.
    kernel_variant = tk.get("kernel_variant", DEFAULT_KERNEL_VARIANT)
    hardware = tk.get("hardware", DEFAULT_HARDWARE)

    # Symmetric gate with cli/lookup: fail loud at sweep startup if
    # the .kernel file came from a plugin we don't know how to drive.
    # Catches "wrong .kernel was committed under this workload"
    # BEFORE the sweep accumulates thousands of FAILED rows
    # tagged with the foreign variant. Same KNOWN_KERNEL_VARIANTS
    # gate cli/lookup uses at deploy time — defense in depth.
    if kernel_variant not in KNOWN_KERNEL_VARIANTS:
        raise KernelRegistryMissingError(
            f"{kernel_path} winner has kernel_variant="
            f"{kernel_variant!r}, not in KNOWN_KERNEL_VARIANTS="
            f"{sorted(KNOWN_KERNEL_VARIANTS)}. The .kernel file is "
            f"from a plugin this codebase doesn't dispatch on yet — "
            f"add an emit branch in cli/lookup.py and the variant "
            f"to core/discriminator.KNOWN_KERNEL_VARIANTS first.",
        )

    return {
        "case":           tk.get("case"),
        "page_size":      tk.get("page_size"),
        "kernel_K":       tk.get("kernel_K"),
        "code_revision":  tk.get("code_revision"),
        "mnss":           mnss,
        "kernel_variant": kernel_variant,
        "hardware":       hardware,
    }


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
    kernel_pin_keys: dict[str, Any] | None = None,
    commit_every: int = 5,
    worker_id: int = 0,
    worker_count: int = 1,
    on_progress: Callable[[int], None] | None = None,
) -> int:
    """Run a service-sweep pass for one workload.

    Args:
      workload_env: parsed `.workload` env dict. Not directly used by
                    the runner, but carried for symmetry with
                    kernel/tune.run_kernel_tune.
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
      kernel_pin_keys: optional explicit pin_keys to stamp on every
                       row. If None (default), discovered by reading
                       `<workload>.kernel` and picking LOGICAL (else
                       PREFILL). Raises KernelRegistryMissingError if
                       neither is available — this is the pre-flight
                       check that catches "sweep ran but .kernel
                       missing" before recording any FAILED rows.
      commit_every: number of NEW combos between periodic commits.
                    `<= 0` disables periodic; final commit always runs.
      on_progress: optional `(n_new) -> None` callback per row.

    Returns:
      Count of new rows appended in this run.
    """
    # workload_env IS now used (fix #7: feasibility pre-filter reads
    # INPUT_LEN). Previously del'd here as "passed for symmetry".
    # service_revision: stamped per row (review followup) so the
    # projection step can cross-validate every row's recorded
    # revision against the .raw/<sha>.jsonl filename — symmetric
    # with the kernel-side fix-2 cross-validation.

    if kernel_pin_keys is None:
        kernel_pin_keys = resolve_kernel_pin_keys(workload_dir, workload_name)

    search_space = service_search_space(workload_dir, workload_name)

    skip_set = build_skip_set(
        raw_path,
        key_fn=lambda r: service_combo_key(r["combo"]),
        status_filter=PERMANENT_STATUSES,
    )

    if worker_count < 1 or not (0 <= worker_id < worker_count):
        raise ValueError(
            f"invalid worker config: worker_id={worker_id} "
            f"worker_count={worker_count} — require 1 <= count and "
            f"0 <= id < count.",
        )

    worker_tag = (f"w{worker_id}/{worker_count} "
                  if worker_count > 1 else "")
    n_new = 0
    for combo in enumerate_service_combos(search_space=search_space):
        ck = service_combo_key(combo)
        if ck in skip_set:
            continue
        # Multi-worker bucket assignment (mirror of kernel/tune).
        # `worker_count=1` (default) keeps every combo.
        if worker_count > 1 and worker_bucket(ck, worker_count) != worker_id:
            continue
        feasible, skip_reason = _is_feasible(
            combo, workload_env, kernel_pin_keys=kernel_pin_keys,
        )
        if not feasible:
            # Pre-filter: record as SKIPPED so resume picks it up and
            # doesn't re-attempt next sweep. Don't call measurement_fn.
            result = {"status": "SKIPPED", "reason": skip_reason}
        else:
            # Pre-measurement progress line — see kernel/tune for the
            # same pattern. A bench combo can take 10+ min; without
            # this log the operator can't tell what's happening.
            logger.info(
                "[sweep %s *] MNB=%-7s MNS=%-5s → measuring...",
                worker_tag,
                combo.get("MAX_NUM_BATCHED_TOKENS"),
                combo.get("MAX_NUM_SEQS"),
            )
            try:
                result = measurement_fn(combo)
            except Exception as e:    # pylint: disable=broad-except
                result = {
                    "status": "UNKNOWN_ERROR",
                    "error": f"{type(e).__name__}: {e}",
                }
        # Defensive (fix #24 + followup): coerce non-dict and
        # partial-dict (missing or None status) measurement returns
        # to UNKNOWN_ERROR. Partial-dict is hostile because a row
        # with status=None never matches PERMANENT_STATUSES, so
        # resume re-attempts it forever. UNKNOWN_ERROR is retryable
        # (the user might fix the bug) but at least typed.
        if not isinstance(result, dict):
            result = {
                "status": "UNKNOWN_ERROR",
                "error":  f"measurement_fn returned non-dict: "
                          f"{type(result).__name__}",
            }
        elif result.get("status") is None:
            # Dict-literal precedence (same honesty as fix #14): the
            # `status` / `error` keys below land AFTER the **result
            # spread, so they unconditionally overwrite same-named
            # keys from measurement_fn. The `.get("error", ...)`
            # only matters if `error` was absent in result.
            result = {
                **result,
                "status": "UNKNOWN_ERROR",
                "error":  result.get("error",
                                     "measurement_fn returned dict "
                                     "with missing or None status"),
            }
        row = {
            "combo":            combo,
            "kernel_pin_keys":  kernel_pin_keys,
            "service_revision": service_revision,
            **result,
        }
        append_row(raw_path, row)
        n_new += 1

        # Per-combo progress log — mirror of kernel/tune.
        _status = result.get("status", "?")
        _metrics = result.get("metrics") or {}
        if _status == "SUCCESS":
            _summary = (
                f"req/s={_metrics.get('req_per_sec', 0):.2f} "
                f"ttft={_metrics.get('ttft_mean_ms', 0):.0f}ms "
                f"p99={_metrics.get('ttft_p99_ms', 0):.0f}ms"
            )
            if result.get("mock"):
                _summary += " (MOCK)"
        elif "error" in result:
            _summary = f"error={result['error'][:60]}"
        else:
            _summary = ""
        logger.info(
            "[sweep %s%4d] MNB=%-7s MNS=%-5s → %-13s %s",
            worker_tag,
            n_new,
            combo.get("MAX_NUM_BATCHED_TOKENS"),
            combo.get("MAX_NUM_SEQS"),
            _status,
            _summary,
        )

        if on_progress is not None:
            on_progress(n_new)
        if commit_every > 0 and n_new % commit_every == 0:
            commit_and_push(
                [raw_path],
                f"[Sweep-v2] progress: {n_new} combos ({workload_name})",
            )

        # SMOKE_TEST=1: stop at the first SUCCESS row. Mirrors the
        # kernel-tune behavior — see kernel/tune.run_kernel_tune
        # for the rationale (truncation to one combo could pick an
        # infeasible config; first-SUCCESS is robust).
        if (os.environ.get("SMOKE_TEST") == "1"
                and result.get("status") == "SUCCESS"):
            break

    # Final commit fires only if there are uncommitted rows since the
    # last periodic checkpoint. When n_new is an exact multiple of
    # commit_every, the periodic commit already covered them — a
    # second commit here would be an empty no-op (fix #10).
    if n_new > 0 and (commit_every <= 0 or n_new % commit_every != 0):
        commit_and_push(
            [raw_path],
            f"[Sweep-v2] complete: {n_new} combos ({workload_name})",
        )

    return n_new


def main(argv: list[str] | None = None) -> int:
    """CLI entry. Reads `.workload`, sweeps service axes via vLLM bench.

    Workload env vars are pushed into `os.environ` BEFORE the
    measurement adapter is built so `run_benchmark.sh` (the v1
    bench wrapper) sees them when it sources its own internals.
    """
    import argparse
    p = argparse.ArgumentParser(
        description="Run a service sweep for one workload.",
    )
    p.add_argument("workload", type=Path)
    p.add_argument("--commit-every", type=int, default=5,
                   help="Commit raw store every N combos.")
    p.add_argument("--timeout", type=int, default=1800,
                   help="Per-combo bench timeout in seconds.")
    p.add_argument(
        "--worker-id", type=int, default=0,
        help="This worker's bucket index in [0, --worker-count).",
    )
    p.add_argument(
        "--worker-count", type=int, default=1,
        help="Total number of workers participating. Default 1.",
    )
    args = p.parse_args(argv)
    from tools.tuning.v2.core.logs import configure as configure_logging
    configure_logging()

    if not args.workload.exists():
        logger.error("workload not found: %s", args.workload)
        return 1

    # Parse workload + push into os.environ (run_benchmark.sh reads
    # MAX_MODEL_LEN / TENSOR_PARALLEL_SIZE / etc. via bash sourcing).
    from tools.tuning.v2.cli.validate import parse_workload_env
    workload_env = parse_workload_env(args.workload)
    import os as _os
    _os.environ.update(workload_env)

    workload_dir = args.workload.parent
    workload_name = args.workload.stem

    # Lazy import — bench adapter pulls tools.benchmark.parse_bench_log
    # and (transitively) requires the bench shell script on disk.
    from tools.tuning.v2.core.sha import service_sha
    from tools.tuning.v2.service.measurement_bench import (
        make_measurement_fn,
    )

    service_revision = service_sha()
    raw_path = (
        workload_dir / f"{workload_name}.service.raw" /
        f"{service_revision}.jsonl"
    )
    measurement_fn = make_measurement_fn(
        args.workload, timeout_seconds=args.timeout,
    )

    n_new = run_service_sweep(
        workload_env=workload_env,
        workload_dir=workload_dir,
        workload_name=workload_name,
        raw_path=raw_path,
        measurement_fn=measurement_fn,
        service_revision=service_revision,
        commit_every=args.commit_every,
        worker_id=args.worker_id,
        worker_count=args.worker_count,
    )
    logger.info("Sweep-v2: %d new rows written to %s", n_new, raw_path)
    # Machine-parseable result on stdout.
    print(raw_path)
    return 0


if __name__ == "__main__":   # pragma: no cover
    sys.exit(main())
