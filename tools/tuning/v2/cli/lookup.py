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
import sys
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _case_winner(kernel_doc: dict[str, Any], case: str) -> dict[str, Any] | None:
    """Find the first winner in kernel_doc whose tuning_key.case matches."""
    for winner in kernel_doc.get("winners", []):
        if winner.get("tuning_key", {}).get("case") == case:
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

    env: dict[str, str] = {}

    # Service combo first (deploy-time sched knobs).
    for k, v in service_winner.get("combo", {}).items():
        env[str(k)] = str(v)

    # Kernel-derived pins per case.
    decode = _case_winner(kernel_doc, "decode")
    mixed = _case_winner(kernel_doc, "mixed")
    logical = _case_winner(kernel_doc, "logical")
    prefill = _case_winner(kernel_doc, "prefill")

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
            f"no LOGICAL or PREFILL kernel winner in {kernel_path}; "
            "cannot resolve decoupled-K env vars.",
        )

    tp = chunked["tunable_params"]
    tk = chunked["tuning_key"]
    env["RPA_P_BLOCK_SIZES"] = _fmt_block_sizes(tp)
    env["BLOCK_SIZE"] = str(tk["page_size"])
    env["RPA_KERNEL_K"] = str(tk.get("kernel_K", 0))

    # Decoupled-K pins (only for LOGICAL).
    if chunked is logical:
        mnss = int(tp["mnss"])
        kernel_K = int(tk["kernel_K"])
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

    if not args.workload.exists():
        print(f"workload not found: {args.workload}", file=sys.stderr)
        return 1
    workload_dir = args.workload.parent
    workload_name = args.workload.stem
    try:
        env = lookup_env(workload_dir, workload_name, objective=args.objective)
    except (FileNotFoundError, KeyError) as e:
        print(f"lookup failed: {e}", file=sys.stderr)
        return 1
    for k in sorted(env.keys()):
        print(f"{k}={env[k]}")
    return 0


if __name__ == "__main__":   # pragma: no cover
    sys.exit(main())
