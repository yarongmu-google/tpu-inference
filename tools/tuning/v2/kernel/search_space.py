# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Kernel-tune search space: default candidate ranges + per-workload overlay.

Defaults inherit v1's `RPA_V3_*_LST` defaults plus the implicit
`mnss = mns × (M+1)` formula for M ∈ {1, 2, 4, 8, 16, 32}. Liberal by
intent — the tuner's SMEM/VMEM-fit pruner drops infeasible combos at
case-generation time, so a wider candidate list just rotates the prune
fraction higher rather than trapping us in a too-narrow regime. The
narrow-axes bug from rpa3_2's Phase 5 (where `MAX_NUM_BATCHED_TOKENS`
stopped at 65,536 and missed the MNB=131,072 sweet spot) is exactly
what the liberal default prevents at the kernel layer.

Per-workload overlay file `<workload>.kernel_axes.json` lives alongside
the `.workload` file. Schema (any subset of axes; missing axes inherit
the default):

```json
{
  "bq_sz":     [256],
  "bkv_sz":    [2048],
  "bq_csz":    [256],
  "bkv_csz":   [512],
  "kernel_K":  [256],
  "page_size": [128],
  "mnss":      [33, 4224]
}
```

The `mnss` overlay bypasses the `mns × (M+1)` derivation entirely.
Useful when the operator wants to test specific mnss values not in
the natural candidate list — e.g., `mnss=33` for the Phase-4 latency
case at MNS=1, or `mnss=4224` only for a narrow throughput sweep.

This module is pure-data + filesystem. No TPU/JAX imports.
"""

import json
from pathlib import Path

from tools.tuning.v2.core.overlay import validate_overlay_schema

# v1-inherited defaults. Wider than what any single workload typically
# wants (~thousands of combos pre-prune); the kernel-tune SMEM/VMEM
# check + the per-workload overlay narrow these down.
DEFAULT_BQ_SZ:     list[int] = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
DEFAULT_BKV_SZ:    list[int] = [512, 1024, 2048, 4096, 8192]
DEFAULT_BQ_CSZ:    list[int] = [32, 64, 128, 256, 512, 1024, 2048]
DEFAULT_BKV_CSZ:   list[int] = [128, 256, 512, 1024, 2048]
DEFAULT_KERNEL_K:  list[int] = [128, 256, 512, 1024, 2048, 4096, 8192]
DEFAULT_PAGE_SIZE: list[int] = [64, 128, 256]

# mnss candidates are mns-dependent: mnss = mns × (M+1) for M in this
# list. M=1 corresponds to "one chunk per phys req"; M=32 to "32 chunks
# per phys" (the long-prefill regime that the L kernel was designed for).
DEFAULT_M_VALUES:  list[int] = [1, 2, 4, 8, 16, 32]


def derive_mnss_candidates(
    mns: int,
    m_values: list[int] | None = None,
) -> list[int]:
    """Compute mnss = mns × (M + 1) for each M, sorted and de-duplicated.

    Args:
      mns: workload `MAX_NUM_SEQS` (persistent batch capacity).
      m_values: M values to sweep. Defaults to DEFAULT_M_VALUES.

    Returns:
      Sorted list of distinct mnss values.
    """
    if m_values is None:
        m_values = DEFAULT_M_VALUES
    return sorted({mns * (m + 1) for m in m_values})


def kernel_search_space(
    workload_dir: Path,
    workload_name: str,
    max_num_seqs: int,
) -> dict[str, list[int]]:
    """Full kernel-tune search space for one workload.

    Args:
      workload_dir: Directory containing the `.workload` file and the
                    optional overlay.
      workload_name: Workload stem (e.g. `"prefill_heavy"`).
      max_num_seqs: Workload's `MAX_NUM_SEQS`. Used to derive default
                    mnss candidates if the overlay doesn't override.

    Returns:
      Dict mapping each axis name to its candidate list. Axes:
      `bq_sz`, `bkv_sz`, `bq_csz`, `bkv_csz`, `kernel_K`, `page_size`,
      `mnss`. Per-workload overlay (`<workload>.kernel_axes.json`)
      overrides any subset; the rest inherit defaults.

    Errors propagate: a malformed overlay JSON raises `JSONDecodeError`.
    Missing overlay file is normal (returns defaults).
    """
    space: dict[str, list[int]] = {
        "bq_sz":     list(DEFAULT_BQ_SZ),
        "bkv_sz":    list(DEFAULT_BKV_SZ),
        "bq_csz":    list(DEFAULT_BQ_CSZ),
        "bkv_csz":   list(DEFAULT_BKV_CSZ),
        "kernel_K":  list(DEFAULT_KERNEL_K),
        "page_size": list(DEFAULT_PAGE_SIZE),
        "mnss":      derive_mnss_candidates(max_num_seqs),
    }
    overlay = _load_overlay(workload_dir, workload_name)
    space.update(overlay)
    return space


def _load_overlay(
    workload_dir: Path,
    workload_name: str,
) -> dict[str, list[int]]:
    """Load `<workload>.kernel_axes.json` if present, else empty dict.

    Raises `OverlayValidationError` (via `validate_overlay_schema`) if
    the file exists but doesn't match the flat list[positive int]
    schema (fix #7).
    """
    overlay_path = workload_dir / f"{workload_name}.kernel_axes.json"
    if not overlay_path.exists():
        return {}
    with open(overlay_path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    validate_overlay_schema(doc, overlay_path)
    return doc
