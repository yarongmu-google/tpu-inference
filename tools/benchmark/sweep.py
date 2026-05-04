# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Sweep-spec loading + combo enumeration for vLLM-bench sweeps.

This module is the *pure* part of the sweep driver: parse a JSON spec
into a list of (env-var dict) combos, plus deterministic ID / path
helpers. Subprocess execution and CLI live in a follow-up commit so this
file stays unit-testable end-to-end without touching disk or invoking
vLLM.

Spec shape:
    {
      "case_file":   "<path to .env>",
      "sweep_name":  "<short label>",
      "sweep_axes":  {"VAR_A": [a1, a2], "VAR_B": [b1]},      // cartesian
      "coupled_axes":[{"VAR_X": x1, "VAR_Y": y1}, ...],        // each entry = one combo
      "fixed":       {"VAR_C": "c"}                            // applied to every combo
    }

Keys must be **disjoint** across `sweep_axes`, `coupled_axes` (entry
keyset), and `fixed`. Coupled entries must all share the same key set.
"""

import hashlib
import itertools
import json
import os
from pathlib import Path
from typing import Any


class SpecError(ValueError):
    """Raised when a sweep spec is malformed."""


# ----- Loading & validation ------------------------------------------------


def load_spec(path: str | os.PathLike) -> dict[str, Any]:
    """Read + validate a sweep spec from JSON. Raises SpecError on bad shape."""
    with open(path) as f:
        spec = json.load(f)
    if not isinstance(spec, dict):
        raise SpecError("top-level spec must be a JSON object")
    _validate_spec(spec)
    return spec


def _validate_spec(spec: dict[str, Any]) -> None:
    required = {"case_file", "sweep_name"}
    missing = required - spec.keys()
    if missing:
        raise SpecError(f"missing required keys: {sorted(missing)}")

    sweep_axes = spec.get("sweep_axes") or {}
    coupled_axes = spec.get("coupled_axes") or []
    fixed = spec.get("fixed") or {}

    if not isinstance(sweep_axes, dict):
        raise SpecError("sweep_axes must be a JSON object")
    if not isinstance(coupled_axes, list):
        raise SpecError("coupled_axes must be a JSON array")
    if not isinstance(fixed, dict):
        raise SpecError("fixed must be a JSON object")

    for k, v in sweep_axes.items():
        if not isinstance(v, list) or not v:
            raise SpecError(
                f"sweep_axes['{k}'] must be a non-empty list (got {v!r})")

    for i, entry in enumerate(coupled_axes):
        if not isinstance(entry, dict):
            raise SpecError(
                f"coupled_axes[{i}] must be a JSON object (got {type(entry).__name__})")

    if coupled_axes:
        first_keys = set(coupled_axes[0].keys())
        for i, entry in enumerate(coupled_axes):
            if set(entry.keys()) != first_keys:
                raise SpecError(
                    f"coupled_axes[{i}] keys {sorted(entry.keys())} differ "
                    f"from coupled_axes[0] keys {sorted(first_keys)}")

    sweep_keys = set(sweep_axes.keys())
    coupled_keys = set(coupled_axes[0].keys()) if coupled_axes else set()
    fixed_keys = set(fixed.keys())

    overlap = {
        "sweep_axes/coupled_axes": sweep_keys & coupled_keys,
        "sweep_axes/fixed":         sweep_keys & fixed_keys,
        "coupled_axes/fixed":       coupled_keys & fixed_keys,
    }
    for label, ks in overlap.items():
        if ks:
            raise SpecError(
                f"keys must be disjoint; {label} share: {sorted(ks)}")


# ----- Enumeration ---------------------------------------------------------


def enumerate_combos(spec: dict[str, Any]) -> list[dict[str, str]]:
    """Yield the cartesian × coupled product of `spec`, merged with `fixed`.

    Order is deterministic:
      - sweep_axes iterated in insertion order (the dict's key order)
      - cartesian product in the standard itertools.product order
      - coupled_axes entries in their list order

    All values are stringified (vllm flags want strings).
    """
    sweep_axes = spec.get("sweep_axes") or {}
    coupled_axes = spec.get("coupled_axes") or []
    fixed = spec.get("fixed") or {}

    sweep_keys = list(sweep_axes.keys())
    sweep_values = [sweep_axes[k] for k in sweep_keys]
    cartesian: list[tuple] = (
        list(itertools.product(*sweep_values)) if sweep_values else [()])
    coupled: list[dict] = list(coupled_axes) if coupled_axes else [{}]

    combos: list[dict[str, str]] = []
    for cart_tuple in cartesian:
        cart_dict = dict(zip(sweep_keys, cart_tuple))
        for cpl in coupled:
            env: dict[str, Any] = {}
            env.update(cart_dict)
            env.update(cpl)
            env.update(fixed)
            combos.append({k: str(v) for k, v in env.items()})
    return combos


# ----- Naming & paths ------------------------------------------------------


def combo_id(env: dict[str, str]) -> str:
    """Deterministic 12-hex-char ID derived from canonical-JSON of `env`.

    Stable across spec reorderings: the same set of (key, value) pairs
    always hashes to the same ID. That gives us free resumability — if a
    combo's metrics.txt already exists, we skip it on re-run.
    """
    canonical = json.dumps(env, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]


def case_name_from_path(case_file: str | os.PathLike) -> str:
    """Strip dir + .env suffix. e.g. cases/foo.env -> foo."""
    base = os.path.basename(str(case_file))
    if base.endswith(".env"):
        return base[:-len(".env")]
    return base


def result_dir(
    base_dir: str | os.PathLike,
    case_name: str,
    sweep_name: str,
    cid: str,
) -> Path:
    """Path to a single combo's result dir."""
    return Path(base_dir) / f"bench_{case_name}_{sweep_name}" / cid


def sweep_dir(
    base_dir: str | os.PathLike,
    case_name: str,
    sweep_name: str,
) -> Path:
    """Parent directory holding all combos for one sweep."""
    return Path(base_dir) / f"bench_{case_name}_{sweep_name}"


def is_completed(rdir: str | os.PathLike) -> bool:
    """A combo is 'done' if metrics.txt exists in its result dir."""
    return Path(rdir, "metrics.txt").is_file()
