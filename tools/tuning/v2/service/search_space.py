# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Service-sweep search space: default sweep axes + per-workload overlay.

Symmetric to `tools.tuning.v2.kernel.search_space` but for the service
layer. The service-sweep search axes are vLLM-side knobs (MNB, MNS,
…) plus any other CLI flag the operator wants to vary.

Defaults are liberal per refactor item 7 — wide enough to surface the
sweet spot without hand-tuning. The rpa3_2 Phase 5 lesson:
`MAX_NUM_BATCHED_TOKENS` topping at 65,536 missed the MNB=131,072
peak by a factor of 16x. v2 defaults extend up to 1,081,344
(= mnss × kernel_K at the L kernel's typical configuration) so the
upper end is included by default.

Per-workload overlay file `<workload>.service_axes.json` (schema-flat,
any subset of axes; missing axes inherit defaults):

```json
{
  "MAX_NUM_BATCHED_TOKENS": [8192, 131072, 1081344],
  "MAX_NUM_SEQS":           [128, 1000]
}
```

Decoupled-K / kernel-derived knobs (`RPA_KERNEL_K`, `BLOCK_SIZE`,
`LONG_PREFILL_TOKEN_THRESHOLD`, `RPA_*_BLOCK_SIZES`) are NOT sweep
axes — they come from the kernel layer's auto-link. Setting them
here would violate the kernel-derived-pinned contract from §2 of
`docs/tuning_architecture.md`.

Pure data + filesystem. No vLLM imports.
"""

import json
from pathlib import Path

from tools.tuning.v2.core.overlay import validate_overlay_schema

# Liberal defaults: octave coverage up to mnss × kernel_K at typical
# L-kernel configuration. The orchestrator prunes via memory/feasibility
# checks; an over-wide MNB just OOMs that combo and the sweep continues.
DEFAULT_MAX_NUM_BATCHED_TOKENS: list[int] = [
    8192, 16384, 32768, 65536, 131072, 262144, 524288, 1081344,
]
DEFAULT_MAX_NUM_SEQS: list[int] = [128, 256, 1000]


def service_search_space(
    workload_dir: Path,
    workload_name: str,
) -> dict[str, list[int]]:
    """Build the service-sweep search space for one workload.

    Args:
      workload_dir: Directory containing the `.workload` file and the
                    optional overlay.
      workload_name: Workload stem (e.g. `"prefill_heavy"`).

    Returns:
      Dict mapping axis name -> candidate list. Axes:
      `MAX_NUM_BATCHED_TOKENS`, `MAX_NUM_SEQS`. Per-workload overlay
      `<workload>.service_axes.json` overrides any subset; the rest
      inherit defaults.

    Errors propagate: malformed overlay JSON raises `JSONDecodeError`.
    Missing overlay file is normal (returns defaults).
    """
    space: dict[str, list[int]] = {
        "MAX_NUM_BATCHED_TOKENS": list(DEFAULT_MAX_NUM_BATCHED_TOKENS),
        "MAX_NUM_SEQS":           list(DEFAULT_MAX_NUM_SEQS),
    }
    overlay = _load_overlay(workload_dir, workload_name)
    space.update(overlay)
    return space


def _load_overlay(
    workload_dir: Path,
    workload_name: str,
) -> dict[str, list[int]]:
    """Load `<workload>.service_axes.json` if present, else empty dict.

    Raises `OverlayValidationError` (via `validate_overlay_schema`) if
    the file exists but doesn't match the flat list[positive int]
    schema (fix #7).
    """
    overlay_path = workload_dir / f"{workload_name}.service_axes.json"
    if not overlay_path.exists():
        return {}
    with open(overlay_path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    validate_overlay_schema(doc, overlay_path)
    return doc
