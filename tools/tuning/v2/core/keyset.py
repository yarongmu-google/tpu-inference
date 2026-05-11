# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Canonical-key helpers for skip-set and projection.

Raw rows are plain dicts; comparing them by dict equality is fine but the
skip-set needs a hashable form. This module gives a single canonical
JSON-encoding (sorted keys, compact separators, str-cast values) that
both `build_skip_set` callers and the projection step use.

The canonical encoding is intentionally simple — JSON with `sort_keys=True`
and no whitespace. It is **not** designed to be cross-language stable; it's
just stable across re-reads of the same data within Python. That's enough
for skip-set membership and winners-projection grouping.

Three composable helpers:

- `canonical_json(obj)`: sorted, compact JSON encoding of any
  JSON-serializable dict / list / scalar. Used directly when the caller
  has a flat dict (e.g. a sweep `combo`).
- `combo_key(tuning_key, tunable_params)`: pair of canonical JSONs as a
  tuple. Used by the kernel-tune skip-set (one entry per
  `(tuning_key, tunable_params)` combo).
- `service_combo_key(combo)`: single canonical JSON of a service-sweep
  combo dict. Used by the service-sweep skip-set.
"""

import json
from typing import Any


def canonical_json(obj: Any) -> str:
    """Return a stable JSON encoding (sorted keys, compact)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)


def combo_key(
    tuning_key: dict[str, Any],
    tunable_params: dict[str, Any],
) -> tuple[str, str]:
    """Compose the kernel-tune skip-set key from a (tuning_key, tunable_params) pair.

    Both inputs are canonicalised, then returned as a tuple. The tuple is
    hashable and survives serialization/deserialization round-trips (e.g.
    via the `.raw/<sha>.jsonl` write-then-read path).
    """
    return (canonical_json(tuning_key), canonical_json(tunable_params))


def service_combo_key(combo: dict[str, Any]) -> str:
    """Compose the service-sweep skip-set key from a combo dict.

    A "combo" is one point in the service-sweep search space: a dict of
    `env-var → value`. Same canonical encoding as `canonical_json`.
    """
    return canonical_json(combo)
