# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Service projection: `.service.raw/<sha>.jsonl` -> `<workload>.service`.

Multi-objective projection: for each named objective, pick one winner.
Default objectives:

  throughput_max  -> argmax(metrics.req_per_sec)
  ttft_min        -> argmin(metrics.ttft_mean_ms)
  p99_min         -> argmin(metrics.ttft_p99_ms)

The `<workload>.service` file carries one winner per objective so the
deploy-time lookup can pick the right operating point (chat: ttft_min;
batch: throughput_max; SLA-bounded: p99_min). Callers can extend the
objective set by passing a custom dict to `project_service`.

Symmetric to `tools.tuning.v2.kernel.project`: idempotent, pure-CPU,
reads the latest `.raw/<sha>.jsonl` (preferring SHA match, falling
back to most-recent mtime).
"""

import json
import sys
from pathlib import Path
from typing import Any

from tools.tuning.v2.core.git_atomic import commit_and_push
from tools.tuning.v2.core.projection import project_winners
from tools.tuning.v2.core.raw_store import read_rows
from tools.tuning.v2.core.sha import service_sha


# (objective_name -> (metric_field_path_as_tuple, descending))
# metric_field_path is the chain of dict keys to follow inside each
# row's `metrics` sub-dict.
DEFAULT_OBJECTIVES: dict[str, tuple[tuple[str, ...], bool]] = {
    "throughput_max": (("metrics", "req_per_sec"),  True),
    "ttft_min":       (("metrics", "ttft_mean_ms"), False),
    "p99_min":        (("metrics", "ttft_p99_ms"),  False),
}


def _resolve_raw_path(
    workload_dir: Path,
    workload_name: str,
    service_revision: str | None,
) -> Path | None:
    """Pick the `.service.raw/<sha>.jsonl` to project.

    Prefers SHA match; falls back to most-recent-mtime. Returns None
    if no raw directory or empty.
    """
    raw_dir = workload_dir / f"{workload_name}.service.raw"
    if not raw_dir.exists():
        return None
    if service_revision is not None:
        candidate = raw_dir / f"{service_revision}.jsonl"
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


def _get_path(row: dict[str, Any], path: tuple[str, ...]) -> Any:
    """Walk a dict-path through nested dicts. Raises KeyError on miss."""
    cur: Any = row
    for k in path:
        cur = cur[k]
    return cur


def _row_has_objective(row: dict[str, Any], path: tuple[str, ...]) -> bool:
    """True iff the row carries the full key-path."""
    cur: Any = row
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return False
        cur = cur[k]
    # Metric must be a real number, not None.
    if cur is None:
        return False
    try:
        float(cur)
    except (TypeError, ValueError):
        return False
    return True


def project_service(
    workload_dir: Path,
    workload_name: str,
    *,
    service_revision: str | None = None,
    raw_path: Path | None = None,
    objectives: dict[str, tuple[tuple[str, ...], bool]] | None = None,
) -> Path | None:
    """Project a workload's service raw store into its `.service` file.

    Args:
      workload_dir: directory containing the workload's files.
      workload_name: workload stem.
      service_revision: SHA pair `<tpu_inference>-<vllm>`. Defaults to
                        `core.sha.service_sha()`.
      raw_path: explicit raw-file override (bypasses discovery).
      objectives: mapping from objective name to `(path, descending)`.
                  Defaults to DEFAULT_OBJECTIVES.

    Returns:
      Path to the written `.service` file, or None if no raw data.
    """
    if service_revision is None:
        service_revision = service_sha()
    if objectives is None:
        objectives = DEFAULT_OBJECTIVES

    if raw_path is None:
        raw_path = _resolve_raw_path(
            workload_dir, workload_name, service_revision,
        )
        if raw_path is None:
            return None

    rows = list(read_rows(raw_path))

    winners_by_objective: dict[str, dict[str, Any] | None] = {}
    for objective_name, (metric_path, descending) in objectives.items():
        candidates = project_winners(
            rows,
            group_fn=lambda _r: "workload",
            objective_fn=lambda r, p=metric_path: float(_get_path(r, p)),
            descending=descending,
            skip_if=lambda r, p=metric_path: (
                r.get("status") != "SUCCESS"
                or not _row_has_objective(r, p)
            ),
        )
        winners_by_objective[objective_name] = (
            candidates[0] if candidates else None
        )

    out_path = workload_dir / f"{workload_name}.service"
    doc = {
        "schema_version":   1,
        "workload":         workload_name,
        "service_revision": service_revision,
        "raw_source":       str(raw_path.name),
        "winners":          winners_by_objective,
    }
    out_path.write_text(
        json.dumps(doc, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out_path


def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        description="Project a workload's service raw store to .service.",
    )
    p.add_argument("workload", type=Path)
    p.add_argument("--service-revision", default=None,
                   help="Pin to a specific .raw/<sha>.jsonl.")
    p.add_argument("--no-commit", action="store_true")
    args = p.parse_args(argv)

    if not args.workload.exists():
        print(f"workload not found: {args.workload}", file=sys.stderr)
        return 1
    workload_dir = args.workload.parent
    workload_name = args.workload.stem
    out = project_service(
        workload_dir, workload_name,
        service_revision=args.service_revision,
    )
    if out is None:
        print(
            f"project_service: no raw data found for "
            f"{workload_dir}/{workload_name}.service.raw/. Run the "
            f"service sweep first.",
            file=sys.stderr,
        )
        return 1
    print(out)
    if not args.no_commit:
        commit_and_push(
            [out],
            f"[Sweep-v2] Update {out.name} from service projection",
        )
    return 0


if __name__ == "__main__":   # pragma: no cover
    sys.exit(main())
