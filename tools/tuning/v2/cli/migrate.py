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
import logging
import sys
from pathlib import Path
from typing import Any, NamedTuple

logger = logging.getLogger(__spec__.name if __spec__ is not None else __name__)

from tools.tuning.v2.cli.validate import parse_workload_env


# v1 fields that we drop in the v2 conversion (not load-bearing).
V1_DROP_FIELDS = ("WarmupTime", "CaseId")

# Tuning-key fields checked for "belongs to this workload" match.
# Beyond model shape, also includes seq-configuration fields when the
# v1 entry carries them — over-coupling workloads that share model
# shape but differ in MNS / seq config is the regression that fix #8
# closes.
MATCH_KEYS = (
    # Model shape:
    "num_q_heads",
    "num_kv_heads",
    "head_dim",
    "max_model_len",
    "q_dtype",
    "kv_dtype",
    "sliding_window",
    # Seq / deployment config (fix #8): v1 tuning_key may carry these.
    "max_num_seqs",
    "input_len",
    "output_len",
    "tensor_parallel_size",
    # Kernel-derived pin (review followup): two workloads sharing
    # model + seq config but at different page sizes are distinct
    # tunes. Without this, they got identical full copies.
    "page_size",
)

# Effectiveness caveat: the seq/deployment + kernel-derived match
# keys above only filter when the v1 entry's tuning_key actually
# carries the field. Older v1 dumps that predate a particular stamp
# (e.g. tuning_key without page_size or max_num_seqs) fall through
# the conservative "missing on either side = match" rule and land
# in every per-workload .kernel file. That's the safer failure mode
# (some duplication, no lost data); the alternative — rejecting
# unstamped entries — would erase real winners. v1 is a fixed
# historical format and won't regrow stamps, so this caveat is
# unlikely to bite future migrations.


class MigrationResult(NamedTuple):
    """Outcome of `migrate_model_dir`. The `status` field lets callers
    (CLI, integration tests) distinguish empty states that previously
    looked identical at the (0, []) return tuple. (Review followup.)"""

    status: str   # "ok" | "no_v1_file" | "no_workloads"
    n_migrated: int
    paths: list[Path]


def _is_v2_format(doc: dict[str, Any]) -> bool:
    """True iff `doc` looks like a v2 production.kernel (post-aggregate).

    v2 docs have a top-level `by_workload` key; v1 has a top-level
    `results` key. Distinguishing prevents the operator from running
    `migrate` on a v2 file (re-run after aggregate) and silently
    producing empty per-workload `.kernel` files. (fix #3)
    """
    return "by_workload" in doc and "results" not in doc


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
    """True iff `tuning_key` matches `workload_env` on all available
    keys.

    Each `MATCH_KEYS` field present in BOTH tuning_key and workload
    env must equal. Missing on either side is treated as match (don't
    filter on absence — workloads may not declare every field, and
    older v1 entries may not have seq-config fields in their
    tuning_key). The function is conservative: only ACTIVE conflicts
    drop a row.

    fix #8 — widened beyond pure model shape to include MAX_NUM_SEQS,
    INPUT_LEN, OUTPUT_LEN, TENSOR_PARALLEL_SIZE. If a v1 entry's
    tuning_key carries these, two workloads with same model shape but
    different MNS get distinct (correctly-filtered) per-workload
    .kernel files instead of identical full copies.
    """
    # Map tuning_key field -> .workload env-var name. Has an entry for
    # every key in MATCH_KEYS by contract.
    _ENV_VAR_FOR = {
        "num_q_heads":          "NUM_Q_HEADS",
        "num_kv_heads":         "NUM_KV_HEADS",
        "head_dim":             "HEAD_DIM",
        "max_model_len":        "MAX_MODEL_LEN",
        "q_dtype":              "Q_DTYPE",
        "kv_dtype":             "KV_DTYPE",
        "sliding_window":       "SLIDING_WINDOW",
        "max_num_seqs":         "MAX_NUM_SEQS",
        "input_len":            "INPUT_LEN",
        "output_len":           "OUTPUT_LEN",
        "tensor_parallel_size": "TENSOR_PARALLEL_SIZE",
    }

    def _wl_value(key: str) -> Any:
        env_var = _ENV_VAR_FOR.get(key)
        if env_var is None:
            # Kernel-derived match keys (e.g. page_size) have no
            # workload env-var counterpart. Skip the workload side
            # of the comparison — the conservative "missing on
            # either side = match" rule kicks in.
            return None
        raw = workload_env.get(env_var, "")
        return raw if raw else None

    for k in MATCH_KEYS:
        if k not in tuning_key:
            continue
        wl_val = _wl_value(k)
        if wl_val is None:
            continue
        tk_val = tuning_key[k]
        # Compare as strings — robust against int vs str mismatches.
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
    # `schema_version` = ENVELOPE schema version for the .kernel
    # file (matches accumulator.build_production), NOT the per-row
    # ROW_SCHEMA_VERSION stamped inside tuning_key. Same name, two
    # schemas — see core/discriminator.py docstring.
    return {
        "schema_version": 1,
        "workload":       "<set by caller>",
        "code_revision":  code_revision,
        "raw_source":     "v1_migration",
        "n_winners":      len(winners),
        "winners":        winners,
    }


class MigrationRefusedError(RuntimeError):
    """Raised when migrate_model_dir refuses to run — typically
    because the input is already v2-format or because per-workload
    .kernel files would be clobbered without --force."""


def migrate_model_dir(
    model_dir: Path,
    *,
    force: bool = False,
) -> MigrationResult:
    """Migrate all workloads in a single per-model directory.

    Reads `model_dir/production.kernel` (v1 format) if present, then
    for each `<workload>.workload` file in the dir, writes the v2
    `<workload>.kernel` file. Does NOT touch the v1 `production.kernel`.

    Args:
      model_dir: per-model dir containing v1 production.kernel +
                 .workload files.
      force: if False (default), refuse to overwrite any pre-existing
             per-workload `<workload>.kernel`. If True, overwrite.
             Default-safe — prevents the post-aggregate re-run
             scenario from silently destroying per-workload files
             (fix #3).

    Returns:
      MigrationResult(status, n_migrated, paths). status is one of:
        - "no_v1_file":  model_dir had no production.kernel; n=0, paths=[]
        - "no_workloads": v1 file present but no `.workload` files; n=0
        - "ok":          n workload files written

    Raises:
      MigrationRefusedError if the v1 input is in fact a v2 file
      (post-aggregate state), or if `force=False` and at least one
      per-workload `.kernel` already exists.
    """
    v1_path = model_dir / "production.kernel"
    if not v1_path.exists():
        return MigrationResult("no_v1_file", 0, [])
    with open(v1_path, "r", encoding="utf-8") as f:
        v1_doc = json.load(f)

    # Schema check (fix #3): refuse v2 input. A v2 production.kernel
    # has by_workload, no results; running migrate on it would emit
    # empty per-workload files and clobber the real ones.
    if _is_v2_format(v1_doc):
        raise MigrationRefusedError(
            f"{v1_path} is already in v2 format (has 'by_workload', "
            f"no 'results' key). Migration would produce 0 winners "
            f"and clobber the per-workload .kernel files. Aborting.",
        )

    workload_files = sorted(model_dir.glob("*.workload"))
    if not workload_files:
        return MigrationResult("no_workloads", 0, [])

    # Overwrite check (fix #3): refuse if any target exists, unless
    # --force.
    if not force:
        existing = [
            model_dir / f"{wl.stem}.kernel"
            for wl in workload_files
            if (model_dir / f"{wl.stem}.kernel").exists()
        ]
        if existing:
            raise MigrationRefusedError(
                f"refusing to overwrite existing per-workload .kernel "
                f"files: {[str(p) for p in existing]}. Pass --force "
                f"to migrate anyway.",
            )

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
    return MigrationResult("ok", len(written), written)


def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        description="Migrate v1 production.kernel to v2 per-workload files.",
    )
    p.add_argument("model_dir", type=Path,
                   help="Per-model directory containing production.kernel.")
    p.add_argument(
        "--force", action="store_true",
        help="Overwrite pre-existing per-workload .kernel files. "
             "By default, migrate refuses if any target exists "
             "(prevents clobbering after a v2 aggregate run).",
    )
    args = p.parse_args(argv)

    from tools.tuning.v2.core.logs import configure as configure_logging
    configure_logging()

    if not args.model_dir.exists() or not args.model_dir.is_dir():
        logger.error("model_dir not a directory: %s", args.model_dir)
        return 1

    try:
        result = migrate_model_dir(args.model_dir, force=args.force)
    except MigrationRefusedError as e:
        logger.error("migrate refused: %s", e)
        return 1

    # Library now returns a typed result with a status field — distinct
    # exit messages for the three empty states (review followup #18).
    if result.status == "no_v1_file":
        logger.error(
            "no v1 production.kernel in %s; nothing to migrate.",
            args.model_dir,
        )
        return 1
    if result.status == "no_workloads":
        logger.error(
            "production.kernel found, but no .workload files in %s; "
            "nothing to do.", args.model_dir,
        )
        return 1
    logger.info("Migrated %d workload(s).", result.n_migrated)
    # Paths are the machine-parseable result — stdout, no timestamp.
    for p in result.paths:
        print(p)
    logger.info(
        "Next steps:  1. Inspect the per-workload .kernel files.  "
        "2. Run aggregate.sh %s to regenerate the v2-format "
        "production.kernel. (This OVERWRITES the v1 "
        "production.kernel — back up first if you need it.)",
        args.model_dir,
    )
    return 0


if __name__ == "__main__":   # pragma: no cover
    sys.exit(main())
