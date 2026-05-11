# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Schema validation for per-workload search-space overlays.

The kernel-tune and service-sweep search-space overlays
(`.kernel_axes.json` / `.service_axes.json`) share a flat schema:

  `{ axis_name: [positive_int, positive_int, ...], ... }`

This module enforces that shape with a single validator, used by both
`tools.tuning.v2.kernel.search_space._load_overlay` and
`tools.tuning.v2.service.search_space._load_overlay`. The point of
the check is to catch malformed overlays at search-space load time —
silently letting `{"bq_sz": "256"}` (string instead of list) through
would balloon into "axis is one character long, so 0, 1, 2, 5, 6
become candidate values" or similar surprises during enumeration
(fix #7).
"""

from pathlib import Path
from typing import Any


class OverlayValidationError(ValueError):
    """Raised when an overlay JSON does not match the required schema."""


def validate_overlay_schema(doc: Any, source_path: Path) -> None:
    """Assert that `doc` is a valid overlay.

    Required shape:
      - Top-level: dict.
      - Each value: a non-empty list.
      - Each list is homogeneous, either:
          * all positive ints (>= 1), excluding bools
            (Python's `bool is int` is a footgun), OR
          * all non-empty strings (used for axes like DATASET that
            sweep over named values such as "sonnet" / "random").

    Mixed-type lists are rejected.

    Args:
      doc: parsed JSON content.
      source_path: file the doc came from. Used in the error message
                   so the operator can find the bad overlay.

    Raises:
      OverlayValidationError with a message that names the offending
      axis and value.
    """
    if not isinstance(doc, dict):
        raise OverlayValidationError(
            f"{source_path}: overlay must be a JSON object at top level "
            f"(got {type(doc).__name__}).",
        )
    for axis, values in doc.items():
        if not isinstance(values, list):
            raise OverlayValidationError(
                f"{source_path}: axis {axis!r} must map to a list "
                f"(got {type(values).__name__}).",
            )
        if not values:
            raise OverlayValidationError(
                f"{source_path}: axis {axis!r} has empty list; "
                f"omit the key to inherit the default candidates.",
            )
        kind = _element_kind(values[0])
        if kind is None:
            raise OverlayValidationError(
                f"{source_path}: axis {axis!r} contains {values[0]!r} "
                f"({type(values[0]).__name__}); each value must be a "
                f"positive int or a non-empty string.",
            )
        for v in values:
            v_kind = _element_kind(v)
            if v_kind != kind:
                raise OverlayValidationError(
                    f"{source_path}: axis {axis!r} mixes types — "
                    f"first element kind {kind}, but {v!r} is "
                    f"{v_kind or type(v).__name__}. Lists must be "
                    f"homogeneous.",
                )
            if kind == "int" and v <= 0:
                raise OverlayValidationError(
                    f"{source_path}: axis {axis!r} contains "
                    f"non-positive value {v}; each candidate must "
                    f"be >= 1.",
                )
            if kind == "str" and not v:
                raise OverlayValidationError(
                    f"{source_path}: axis {axis!r} contains empty "
                    f"string; each candidate must be non-empty.",
                )


def _element_kind(v: Any) -> str | None:
    """Return "int" / "str" for valid element types, else None."""
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return "int"
    if isinstance(v, str):
        return "str"
    return None
