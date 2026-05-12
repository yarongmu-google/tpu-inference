# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""prev-5: post-tune health check.

The May-11 prefill_heavy incident: tune ran for 3.5 hours producing
45 UNKNOWN_ERROR rows and zero SUCCESS rows, then the Python process
died silently at 22:20:11. The bash orchestrator exited 0 (because
the python's exit code was 0 OR the process was SIGKILLed before
flushing). `set -e` in `run_pipeline.sh` couldn't detect this. The
operator only noticed 10 hours later.

This module is invoked AFTER kernel-tune in `run_pipeline.sh`. It
reads the raw store and fails the pipeline if the tune-step output
is suspicious:

  - **No rows at all** → silent process death OR enumerator yielded
    nothing. Either way, pipeline can't proceed.
  - **No SUCCESS rows** → kernel-tune produced only failure rows.
    Sweep step (next) has no winner to pin against — it will fail
    anyway, but with a less-pointed error.

Either condition exits 1 with a typed message naming the count
breakdown so the operator can diagnose. Healthy = SUCCESS count > 0.

CLI: `python3 -m tools.tuning.v2.cli.tune_step_health <workload>`.
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__spec__.name if __spec__ is not None else __name__)


def _resolve_raw_path(workload: Path) -> Path | None:
    """Locate the most-recent kernel.raw/<sha>.jsonl for this workload.

    Mirrors the projection step's choice — the most-recently-modified
    .jsonl file inside `<workload_stem>.kernel.raw/`. Returns None if
    the directory doesn't exist (tune never ran) or has no .jsonl
    files (tune ran but wrote nothing).
    """
    workload_dir = workload.parent
    raw_dir = workload_dir / f"{workload.stem}.kernel.raw"
    if not raw_dir.is_dir():
        return None
    candidates = sorted(
        raw_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        return None
    return candidates[-1]


def _count_statuses(raw_path: Path) -> Counter:
    """Return Counter of status -> row_count from the JSONL store."""
    counts: Counter = Counter()
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                counts["MALFORMED"] += 1
                continue
            counts[row.get("status", "MISSING_STATUS")] += 1
    return counts


def check(workload: Path) -> tuple[int, str]:
    """Return (exit_code, human_message).

    exit_code 0 = healthy; 1 = silent death or zero-success.
    """
    raw_path = _resolve_raw_path(workload)
    if raw_path is None:
        return 1, (
            "tune-step health: no kernel.raw/*.jsonl found at "
            f"{workload.parent / f'{workload.stem}.kernel.raw'}. "
            "Either the tune step never ran or the Python process "
            "died before writing any row. Check the tune log for the "
            "last line; if there's no error, the process likely got "
            "SIGKILL'd (OOM-killer, XLA hang). Re-run after narrowing "
            "the search-space overlay."
        )
    counts = _count_statuses(raw_path)
    total = sum(counts.values())
    if total == 0:
        return 1, (
            f"tune-step health: {raw_path.name} exists but is empty. "
            "The tune step opened the file but wrote no rows — most "
            "likely the enumerator yielded zero combos (every combo "
            "filtered by static prune). Loosen the .kernel_axes.json "
            "overlay or check the validate-step warnings."
        )
    success = counts.get("SUCCESS", 0)
    if success == 0:
        breakdown = ", ".join(
            f"{k}={v}" for k, v in sorted(counts.items())
        )
        return 1, (
            f"tune-step health: 0 SUCCESS rows in {raw_path.name} "
            f"(breakdown: {breakdown}). The subsequent sweep step "
            "has no winner to pin against and will fail. Inspect "
            "the raw rows' `error` field to see why each combo "
            "failed, then narrow the search space."
        )
    breakdown = ", ".join(
        f"{k}={v}" for k, v in sorted(counts.items())
    )
    return 0, (
        f"tune-step health: OK — {success} SUCCESS rows out of "
        f"{total} total ({breakdown})."
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n", 1)[0],
    )
    p.add_argument("workload", type=Path)
    args = p.parse_args(argv)

    from tools.tuning.v2.core.logs import configure as configure_logging
    configure_logging()

    code, msg = check(args.workload)
    if code == 0:
        logger.info("%s", msg)
    else:
        logger.error("%s", msg)
    return code


if __name__ == "__main__":   # pragma: no cover
    sys.exit(main())
