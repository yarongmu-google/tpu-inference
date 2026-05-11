# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Kernel projection: `.kernel.raw/<sha>.jsonl` → `<workload>.kernel`.

Reads the latest raw store and projects winners per (case, tuning_key),
writing the result as a per-workload JSON. The output is consumed by
the auto-link path in `sweep.py` (workload → kernel winners → env
vars) and the accumulator (`core/accumulator.py`).

Selection of the "latest" `.raw/<sha>.jsonl`:
  - Default: pick the file whose `<sha>` matches the current kernel
    source SHA (via `core/sha.kernel_sha()`). If no matching file,
    falls back to the most-recently-modified file in `.kernel.raw/`.
  - Explicit override: pass `raw_path` directly.

The projection is pure-CPU and idempotent — same input → same output.
Re-running on an unchanged raw store rewrites the same file byte-for-
byte (modulo schema timestamps if added later).
"""

import json
import sys
from pathlib import Path
from typing import Any

from tools.tuning.v2.core.git_atomic import commit_and_push
from tools.tuning.v2.core.projection import project_winners
from tools.tuning.v2.core.raw_store import read_rows
from tools.tuning.v2.core.sha import kernel_sha


def _resolve_raw_path(
    workload_dir: Path,
    workload_name: str,
    code_revision: str | None,
) -> Path | None:
    """Pick the `.kernel.raw/<sha>.jsonl` to project.

    If `code_revision` is given and the matching file exists, use it.
    Otherwise pick the most-recently-modified `.jsonl` under the
    `.kernel.raw/` directory. Returns None if no `.raw/` dir or no
    files inside.
    """
    raw_dir = workload_dir / f"{workload_name}.kernel.raw"
    if not raw_dir.exists():
        return None
    if code_revision is not None:
        candidate = raw_dir / f"{code_revision}.jsonl"
        if candidate.exists():
            return candidate
    files = sorted(
        raw_dir.glob("*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not files:
        return None
    return files[0]


def _kernel_group_key(row: dict[str, Any]) -> tuple:
    """Group key for a kernel-tune row: (case, tuning_key)."""
    tk = row["tuning_key"]
    # Use a hashable form of the tuning_key (canonical_json would
    # work; tuple of items in sorted order also fine).
    return (tk.get("case"),) + tuple(
        sorted((k, _hashable(v)) for k, v in tk.items() if k != "case")
    )


def _hashable(v: Any) -> Any:
    """Convert v to a hashable form (dicts -> sorted tuples of items)."""
    if isinstance(v, dict):
        return tuple(sorted((k, _hashable(x)) for k, x in v.items()))
    if isinstance(v, list):
        return tuple(_hashable(x) for x in v)
    return v


def _kernel_objective(row: dict[str, Any]) -> float:
    """Objective for kernel-tune: latency (lower wins)."""
    return float(row["latency_us"])


def _kernel_skip_if(row: dict[str, Any]) -> bool:
    """Skip rows that aren't a valid measurement (missing latency,
    or status != SUCCESS)."""
    if row.get("status") != "SUCCESS":
        return True
    if "latency_us" not in row:
        return True
    return False


def project_kernel(
    workload_dir: Path,
    workload_name: str,
    *,
    code_revision: str | None = None,
    raw_path: Path | None = None,
) -> Path | None:
    """Project a workload's kernel raw store into its `.kernel` file.

    Args:
      workload_dir: directory containing the workload's files.
      workload_name: workload stem.
      code_revision: if given, prefer the matching `.raw/<sha>.jsonl`.
                     Defaults to `kernel_sha()`.
      raw_path: explicit raw-file override. If given, bypasses the
                discovery in `_resolve_raw_path`.

    Returns:
      Path to the written `.kernel` file, or None if no raw data
      exists yet (e.g., tune never ran).
    """
    if code_revision is None:
        code_revision = kernel_sha()
    if raw_path is None:
        raw_path = _resolve_raw_path(
            workload_dir, workload_name, code_revision,
        )
        if raw_path is None:
            return None

    rows = list(read_rows(raw_path))
    winners = project_winners(
        rows,
        group_fn=_kernel_group_key,
        objective_fn=_kernel_objective,
        descending=False,
        skip_if=_kernel_skip_if,
    )

    out_path = workload_dir / f"{workload_name}.kernel"
    doc = {
        "schema_version":  1,
        "workload":        workload_name,
        "code_revision":   code_revision,
        "raw_source":      str(raw_path.name),
        "n_winners":       len(winners),
        "winners":         winners,
    }
    out_path.write_text(
        json.dumps(doc, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out_path


def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        description="Project a workload's kernel raw store to .kernel.",
    )
    p.add_argument("workload", type=Path,
                   help="Path to the `.workload` file.")
    p.add_argument("--code-revision", default=None,
                   help="Pin to a specific .raw/<sha>.jsonl.")
    p.add_argument("--no-commit", action="store_true",
                   help="Skip git commit + push.")
    args = p.parse_args(argv)

    if not args.workload.exists():
        print(f"workload not found: {args.workload}", file=sys.stderr)
        return 1
    workload_dir = args.workload.parent
    workload_name = args.workload.stem
    out = project_kernel(
        workload_dir, workload_name,
        code_revision=args.code_revision,
    )
    if out is None:
        print(
            f"project_kernel: no raw data found for "
            f"{workload_dir}/{workload_name}.kernel.raw/. Run the "
            f"kernel tune first.",
            file=sys.stderr,
        )
        return 1
    print(out)
    if not args.no_commit:
        commit_and_push(
            [out],
            f"[Tune-v2] Update {out.name} from kernel projection",
        )
    return 0


if __name__ == "__main__":   # pragma: no cover
    sys.exit(main())
