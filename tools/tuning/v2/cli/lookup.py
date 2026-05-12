# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Top-down deploy lookup: (workload, objective) -> env vars.

CLI entry: `python3 -m tools.tuning.v2.cli.lookup <workload> [objective]`.

Implements the "Prod (look down)" path from §7 of
`docs/tuning_architecture.md`:

  1. Read `<workload>.service` -> pick the winner for the requested
     objective (default: `throughput_max`).
  2. Read `<workload>.kernel` -> grab kernel vars (block sizes,
     page_size, kernel_K, mnss) per case.
  3. Build the deploy-time env-var dict, applying the kernel-derived
     pinning (BLOCK_SIZE = page_size; LPTT = mnss x kernel_K; etc.).
  4. Print as `K=V` lines for `eval $(...)`-style consumption.

The merge is the v2 replacement for v1's `sweep.py:_apply_auto_link`,
factored out of the sweep flow so it's runnable at deploy time
without re-enumerating combos.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__spec__.name if __spec__ is not None else __name__)

from tools.tuning.v2.core.discriminator import (
    DEFAULT_KERNEL_VARIANT,
    KNOWN_KERNEL_VARIANTS,
)


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _case_winner(
    kernel_doc: dict[str, Any],
    case: str,
    pin_keys: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Find the kernel winner whose tuning_key.case matches.

    If `pin_keys` is given, also requires `tuning_key.page_size` and
    `tuning_key.kernel_K` to match — this is the load-bearing
    constraint that prevents shipping wrong pins when the kernel was
    re-tuned at a different shape than the service was swept against.
    For DECODE / MIXED winners, `kernel_K` in the pin_keys is
    irrelevant (those cases don't carry one) — we match on
    page_size only.
    """
    for winner in kernel_doc.get("winners", []):
        tk = winner.get("tuning_key", {})
        if tk.get("case") != case:
            continue
        if pin_keys is not None:
            if tk.get("page_size") != pin_keys.get("page_size"):
                continue
            # kernel_K only matters for chunked-prefill cases (logical /
            # prefill); decode and mixed don't carry it.
            if case in ("logical", "prefill"):
                if tk.get("kernel_K") != pin_keys.get("kernel_K"):
                    continue
        return winner
    return None


def _fmt_block_sizes(tp: dict[str, Any]) -> str:
    """Format `(bq_sz, bkv_sz, bq_csz, bkv_csz)` as a comma string."""
    return (
        f"{tp['bq_sz']},{tp['bkv_sz']},{tp['bq_csz']},{tp['bkv_csz']}"
    )


def lookup_env(
    workload_dir: Path,
    workload_name: str,
    objective: str = "throughput_max",
) -> dict[str, str]:
    """Return the merged deploy-time env-var dict for a workload+objective.

    Args:
      workload_dir: directory containing `<workload_name>.kernel` and
                    `<workload_name>.service`.
      workload_name: workload stem.
      objective: name of the service-side objective to pick. Must be
                 present in the `.service` winners dict (default
                 `"throughput_max"`).

    Returns:
      Dict of `ENV_VAR -> str`. Includes all kernel-derived pins
      (BLOCK_SIZE, RPA_KERNEL_K, RPA_MAX_NUM_SUBSEQS,
      LONG_PREFILL_TOKEN_THRESHOLD, RPA_D/M/P_BLOCK_SIZES as
      applicable) plus the service combo's vars.

    Raises:
      FileNotFoundError: missing `.kernel` or `.service` file.
      KeyError: objective not in `.service`, or no logical/prefill
                kernel winner exists (the decoupled-K invariant cannot
                be satisfied without the LOGICAL or PREFILL case).
    """
    kernel_path = workload_dir / f"{workload_name}.kernel"
    service_path = workload_dir / f"{workload_name}.service"
    if not kernel_path.exists():
        raise FileNotFoundError(f"missing kernel registry: {kernel_path}")
    if not service_path.exists():
        raise FileNotFoundError(f"missing service registry: {service_path}")

    kernel_doc = _read_json(kernel_path)
    service_doc = _read_json(service_path)

    service_winner = service_doc.get("winners", {}).get(objective)
    if service_winner is None:
        raise KeyError(
            f"no service winner for objective {objective!r} in "
            f"{service_path}",
        )

    # The service winner MUST carry the kernel pin_keys it ran against.
    # Without them we can't resolve to the right .kernel row — first-
    # match by case alone would silently ship pins from a kernel SHA
    # that wasn't the one the sweep tested. This is the
    # architecture-doc §6 contract. The producer (service/project.py
    # via service/sweep.py stamping every raw row) guarantees the
    # field; absence here means stale / hand-edited service file.
    pin_keys = service_winner.get("kernel_pin_keys")
    if pin_keys is None:
        raise KeyError(
            f"service winner for {objective!r} in {service_path} is "
            f"missing kernel_pin_keys. Re-run project_service against "
            f"a .service.raw produced by the current sweep_service "
            f"runner (which stamps pin_keys on every row).",
        )

    # Dispatch on kernel_variant so a future plugin (other TPU kernel
    # variant or a GPU port) can branch on its own emitter. Missing
    # is tolerable history (older .kernel files predate the stamp,
    # default to rpa_v3); mismatching an unknown value is fatal —
    # we don't know which env vars to emit. Architecture doc §13.4.1.
    kernel_variant = pin_keys.get("kernel_variant", DEFAULT_KERNEL_VARIANT)
    if kernel_variant not in KNOWN_KERNEL_VARIANTS:
        raise KeyError(
            f"service winner pin_keys.kernel_variant={kernel_variant!r} "
            f"is not in KNOWN_KERNEL_VARIANTS={sorted(KNOWN_KERNEL_VARIANTS)}; "
            f"add an emit branch in cli/lookup.py before tuning rows "
            f"from this plugin can be looked up.",
        )

    env: dict[str, str] = {}

    # Service combo first (deploy-time sched knobs).
    for k, v in service_winner.get("combo", {}).items():
        env[str(k)] = str(v)

    # Kernel-derived pins per case. Filter by pin_keys so we get the
    # exact kernel winner the service was swept against — not "first
    # match by case", which would silently pick a wrong kernel SHA.
    decode = _case_winner(kernel_doc, "decode", pin_keys=pin_keys)
    mixed = _case_winner(kernel_doc, "mixed", pin_keys=pin_keys)
    logical = _case_winner(kernel_doc, "logical", pin_keys=pin_keys)
    prefill = _case_winner(kernel_doc, "prefill", pin_keys=pin_keys)

    if decode is not None:
        env["RPA_D_BLOCK_SIZES"] = _fmt_block_sizes(decode["tunable_params"])
    if mixed is not None:
        env["RPA_M_BLOCK_SIZES"] = _fmt_block_sizes(mixed["tunable_params"])

    # P-or-L: decoupled-K (LOGICAL) wins over coupled-K (PREFILL) when
    # both are tuned for this workload. This is the "deprecate P" path
    # captured in docs/k_serving_repro.md Phase 5.
    chunked = logical if logical is not None else prefill
    if chunked is None:
        raise KeyError(
            f"no LOGICAL or PREFILL kernel winner in {kernel_path} "
            f"matching pin_keys={pin_keys!r}; deploy lookup cannot "
            f"resolve decoupled-K env vars. The kernel was likely "
            f"re-tuned at a different shape since the service sweep "
            f"ran — re-sweep with the current kernel.",
        )

    tp = chunked["tunable_params"]
    tk = chunked["tuning_key"]
    env["RPA_P_BLOCK_SIZES"] = _fmt_block_sizes(tp)
    env["BLOCK_SIZE"] = str(tk["page_size"])

    # Decoupled-K pins (only for LOGICAL). Under coupled-K (PREFILL
    # only), we OMIT both RPA_KERNEL_K and the decoupled-K-derived
    # vars. Emitting RPA_KERNEL_K=0 would be ambiguous — consumers
    # might interpret 0 as "decoupled-K active with K=0" instead of
    # "decoupled-K absent". (fix #7)
    if chunked is logical:
        mnss = int(tp["mnss"])
        kernel_K = int(tk["kernel_K"])
        env["RPA_KERNEL_K"] = str(kernel_K)
        env["RPA_MAX_NUM_SUBSEQS"] = str(mnss)
        env["LONG_PREFILL_TOKEN_THRESHOLD"] = str(mnss * kernel_K)

    return env


def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        description="Look up deploy-time env vars for a workload.",
    )
    p.add_argument("workload", type=Path,
                   help="Path to the .workload file (sibling to "
                        ".kernel and .service).")
    p.add_argument("--objective", default="throughput_max",
                   help="Service objective to pick (e.g. "
                        "throughput_max, ttft_min, p99_min).")
    args = p.parse_args(argv)

    from tools.tuning.v2.core.logs import configure as configure_logging
    configure_logging()

    if not args.workload.exists():
        logger.error("workload not found: %s", args.workload)
        return 1
    workload_dir = args.workload.parent
    workload_name = args.workload.stem
    try:
        env = lookup_env(workload_dir, workload_name, objective=args.objective)
    except (FileNotFoundError, KeyError) as e:
        logger.error("lookup failed: %s", e)
        return 1
    # K=V env-var lines on stdout (machine-parseable, eval-able).
    for k in sorted(env.keys()):
        print(f"{k}={env[k]}")
    return 0


if __name__ == "__main__":   # pragma: no cover
    sys.exit(main())
