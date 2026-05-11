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


class CodeRevisionMismatchError(RuntimeError):
    """Raised when a raw row's tuning_key.code_revision doesn't match
    the .jsonl filename's SHA. Indicates files were manually
    concatenated or a buggy tuner stamped the wrong SHA."""


def _resolve_raw_path(
    workload_dir: Path,
    workload_name: str,
    code_revision: str | None,
) -> Path | None:
    """Pick the `.kernel.raw/<sha>.jsonl` to project.

    Two-mode contract:
      - **code_revision given explicitly**: file MUST exist at
        `<workload>.kernel.raw/<code_revision>.jsonl`. No mtime
        fallback — when the operator pins the revision, missing
        file is a fail-loud condition (fix #2).
      - **code_revision is None**: discovery mode. Picks the
        most-recently-modified `.jsonl` under `.kernel.raw/`.
        Returns None if no `.raw/` dir or no files inside.
    """
    raw_dir = workload_dir / f"{workload_name}.kernel.raw"
    if not raw_dir.exists():
        return None
    files = sorted(
        raw_dir.glob("*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not files:
        # No raw data is a valid state (tune never ran). Return None
        # even if code_revision was specified — empty-dir isn't a
        # "wrong SHA" condition.
        return None
    if code_revision is not None:
        candidate = raw_dir / f"{code_revision}.jsonl"
        if not candidate.exists():
            raise FileNotFoundError(
                f"explicit code_revision {code_revision!r} but "
                f"{candidate} does not exist. {raw_dir} contains: "
                f"{[p.name for p in files]}. Either drop the "
                f"--code-revision arg (fall back to most-recent) or "
                f"pick from the list.",
            )
        return candidate
    return files[0]


def _kernel_group_key(row: dict[str, Any]) -> tuple:
    """Group key for a kernel-tune row.

    Two raw rows belong to the same group iff their tuning_key matches
    on every field. The grouped winner is the lowest-latency row
    within each group — i.e., one winner per unique tuning identity
    regardless of how many times that identity was measured (re-runs,
    auto-resume picking up where a crash left off, etc.).

    The key is built as `(case, *sorted_non_case_items)` so `case`
    sorts first in the projection output — useful when scanning
    `<workload>.kernel` by case manually. The rest are sorted on key
    name for hash stability across Python versions / dict iteration
    orders.
    """
    tk = row["tuning_key"]
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
      code_revision: optional pin. If given, the matching `.raw/<sha>.jsonl`
                     must exist (fail-loud, no mtime fallback). If
                     None, discovery picks the most-recently-modified
                     `.jsonl` and the output is stamped with the SHA
                     extracted from that filename (fix #2).
      raw_path: explicit raw-file override. Bypasses both discovery
                and the explicit-revision-must-exist check; the
                output's `code_revision` is stamped from
                `raw_path.stem`.

    Returns:
      Path to the written `.kernel` file, or None if no raw data
      exists yet (e.g., tune never ran).

    Raises:
      FileNotFoundError: explicit `code_revision` was given but the
                         matching file doesn't exist.
      CodeRevisionMismatchError: a row in the raw store carries a
                                 `tuning_key.code_revision` that
                                 doesn't match the file's SHA. Defends
                                 against manual concatenation /
                                 wrong-stamp regressions (fix #3).
    """
    if raw_path is None:
        raw_path = _resolve_raw_path(
            workload_dir, workload_name, code_revision,
        )
        if raw_path is None:
            return None

    # The SHA in the output is whatever the file's name says — not the
    # caller's request. If they passed None we discovered; if they
    # passed an explicit SHA we resolved to the matching file (above).
    file_sha = raw_path.stem

    rows = list(read_rows(raw_path))

    # Cross-validate: every row in <sha>.jsonl must carry tuning_key.
    # code_revision == sha. A mismatch means files were manually
    # concatenated or a buggy run stamped wrong SHAs into rows
    # (fix #3). NOTE: `i` is the post-parse row index — read_rows
    # silently skips malformed lines (logs a warning), so this index
    # does NOT correspond to raw JSONL line numbers when the file is
    # partially corrupt. Operators correlating against grep / sed
    # output should re-derive line numbers from the row content.
    for i, row in enumerate(rows):
        row_sha = row.get("tuning_key", {}).get("code_revision")
        if row_sha is not None and row_sha != file_sha:
            raise CodeRevisionMismatchError(
                f"parsed-row index {i} in {raw_path} has "
                f"tuning_key.code_revision={row_sha!r} but the file "
                f"is named {file_sha!r}. Files are not concatenable "
                f"across kernel SHAs — re-run the tune.",
            )

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
        "code_revision":   file_sha,
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
