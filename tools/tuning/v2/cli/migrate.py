# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""One-shot migration: rpa3_2's shared `production.kernel` ->
v2 per-workload `<workload>.kernel` files.

The v1 stack stored ALL kernel-tune entries for a model in one
`production.kernel`, regardless of which workload they came from.
v2 partitions by workload — `<workload>.kernel` is the primary
artifact, `production.kernel` is an aggregator (re-derivable).

This script splits the v1 file into v2 per-workload files. The
splitting heuristic: an entry "belongs" to a workload iff the entry's
`tuning_key` matches the workload's model-shape fields
(NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, MAX_MODEL_LEN, dtype,
sliding_window). For our typical setup where multiple workloads share
the same model shape (e.g. prefill_heavy + prefill_heavy_latency
both target Llama 3 8B), each workload receives a full copy. v2
re-tunes will then differentiate the per-workload files as their
raw stores diverge.

Schema conversion per entry:
  v1 row: `{"tuning_key": ..., "tunable_params": ..., "Latency": N,
           "WarmupTime": ..., "CaseId": ...}`
  v2 row: `{"tuning_key": ..., "tunable_params": ...,
           "status": "SUCCESS", "latency_us": float(N)}`

The v1 `production.kernel` and `production.service` are NOT modified
or deleted by this script — operators verify the v2 per-workload
files first, then can run `aggregate.sh` to regenerate the v2-format
`production.kernel` (overwrites v1's file in the same path).
"""

import json
import sys
from pathlib import Path
from typing import Any

from tools.tuning.v2.cli.validate import parse_workload_env


# v1 fields that we drop in the v2 conversion (not load-bearing).
V1_DROP_FIELDS = ("WarmupTime", "CaseId")

# Model-shape keys checked for "belongs to this workload" match.
MODEL_SHAPE_KEYS = (
    "num_q_heads",
    "num_kv_heads",
    "head_dim",
    "max_model_len",
    "q_dtype",
    "kv_dtype",
    "sliding_window",
)


def _convert_v1_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Convert one v1 production.kernel entry to a v2 row.

    Renames `Latency` -> `latency_us` (cast to float), adds
    `status="SUCCESS"`, drops `WarmupTime` / `CaseId`. Keeps
    `tuning_key` and `tunable_params` verbatim.
    """
    # `WarmupTime` and `CaseId` (V1_DROP_FIELDS) are intentionally not
    # copied — they aren't load-bearing for v2's projection.
    new_entry: dict[str, Any] = {
        "tuning_key":     entry["tuning_key"],
        "tunable_params": entry["tunable_params"],
        "status":         "SUCCESS",
    }
    if "Latency" in entry:
        new_entry["latency_us"] = float(entry["Latency"])
    return new_entry


def _model_shape_matches(
    tuning_key: dict[str, Any],
    workload_env: dict[str, str],
) -> bool:
    """True iff `tuning_key`'s model-shape fields all match `workload_env`.

    Each `MODEL_SHAPE_KEYS` field in the tuning_key must equal the
    corresponding value derived from the workload (best-effort cast
    to int for numeric fields). Missing keys in either side are
    treated as match (don't filter on absence — the workload may not
    declare every field).
    """
    # Map tuning_key model-shape key -> .workload env-var name. Has an
    # entry for every key in MODEL_SHAPE_KEYS by contract.
    _ENV_VAR_FOR = {
        "num_q_heads":    "NUM_Q_HEADS",
        "num_kv_heads":   "NUM_KV_HEADS",
        "head_dim":       "HEAD_DIM",
        "max_model_len":  "MAX_MODEL_LEN",
        "q_dtype":        "Q_DTYPE",
        "kv_dtype":       "KV_DTYPE",
        "sliding_window": "SLIDING_WINDOW",
    }

    def _wl_value(key: str) -> Any:
        """Return the workload's value for a model-shape key, or None
        if not declared by the workload."""
        raw = workload_env.get(_ENV_VAR_FOR[key], "")
        return raw if raw else None

    for k in MODEL_SHAPE_KEYS:
        if k not in tuning_key:
            continue
        wl_val = _wl_value(k)
        if wl_val is None:
            continue
        tk_val = tuning_key[k]
        # Compare as strings (workload env is always strings) — robust
        # against int vs str mismatches.
        if str(tk_val) != str(wl_val):
            return False
    return True


def split_v1_production_kernel(
    v1_doc: dict[str, Any],
    workload_env: dict[str, str],
    code_revision: str = "v1_migration",
) -> dict[str, Any]:
    """Build a v2 `<workload>.kernel` doc from a v1 `production.kernel`.

    Args:
      v1_doc: parsed v1 production.kernel content.
      workload_env: parsed `.workload` env (string values).
      code_revision: stamp for the v2 envelope. Defaults to a literal
                     marker so downstream tooling can tell migrated
                     data apart from native v2 tunes.

    Returns:
      Dict matching v2's `.kernel` envelope:
        {"schema_version": 1, "workload": ..., "code_revision": ...,
         "raw_source": "v1_migration", "n_winners": N,
         "winners": [...]}
    """
    results = v1_doc.get("results", {})
    winners: list[dict[str, Any]] = []
    for case_name, entries in results.items():
        for entry in entries:
            tk = entry.get("tuning_key", {})
            if not _model_shape_matches(tk, workload_env):
                continue
            new_row = _convert_v1_entry(entry)
            # Ensure `case` is present (v1 may have it on tuning_key,
            # but in case it's missing, derive from the outer key).
            if "case" not in new_row["tuning_key"]:
                new_row["tuning_key"]["case"] = case_name
            winners.append(new_row)
    return {
        "schema_version": 1,
        "workload":       "<set by caller>",
        "code_revision":  code_revision,
        "raw_source":     "v1_migration",
        "n_winners":      len(winners),
        "winners":        winners,
    }


def migrate_model_dir(model_dir: Path) -> tuple[int, list[Path]]:
    """Migrate all workloads in a single per-model directory.

    Reads `model_dir/production.kernel` (v1 format) if present, then
    for each `<workload>.workload` file in the dir, writes the v2
    `<workload>.kernel` file. Does NOT touch the v1 `production.kernel`.

    Returns:
      `(n_workloads_migrated, written_paths)`.
    """
    v1_path = model_dir / "production.kernel"
    if not v1_path.exists():
        return 0, []
    with open(v1_path, "r", encoding="utf-8") as f:
        v1_doc = json.load(f)

    workload_files = sorted(model_dir.glob("*.workload"))
    written: list[Path] = []
    for wl_path in workload_files:
        workload_name = wl_path.stem
        wl_env = parse_workload_env(wl_path)
        v2_doc = split_v1_production_kernel(v1_doc, wl_env)
        v2_doc["workload"] = workload_name
        out = model_dir / f"{workload_name}.kernel"
        out.write_text(
            json.dumps(v2_doc, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        written.append(out)
    return len(written), written


def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        description="Migrate v1 production.kernel to v2 per-workload files.",
    )
    p.add_argument("model_dir", type=Path,
                   help="Per-model directory containing production.kernel.")
    args = p.parse_args(argv)

    if not args.model_dir.exists() or not args.model_dir.is_dir():
        print(f"model_dir not a directory: {args.model_dir}",
              file=sys.stderr)
        return 1

    n, paths = migrate_model_dir(args.model_dir)
    if n == 0:
        print(
            f"migrate: no production.kernel or no .workload files in "
            f"{args.model_dir}; nothing to do.",
            file=sys.stderr,
        )
        return 1
    print(f"Migrated {n} workload(s):")
    for p in paths:
        print(f"  {p}")
    print(
        f"\nNext steps:\n"
        f"  1. Inspect the per-workload .kernel files.\n"
        f"  2. Run aggregate.sh {args.model_dir} to regenerate the "
        f"v2-format production.kernel.\n"
        f"     (This OVERWRITES the v1 production.kernel — back up "
        f"first if you need it.)\n",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":   # pragma: no cover
    sys.exit(main())
