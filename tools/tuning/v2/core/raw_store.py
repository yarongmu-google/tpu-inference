# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Append-only JSONL store for raw kernel-tune / service-sweep measurements.

Each `.kernel.raw/<kernel_sha>.jsonl` and `.service.raw/<service_sha>.jsonl`
is an append-only JSONL file. Rows are written one-at-a-time as the tuner /
sweeper produces measurements, with crash-atomicity (a kill / OOM mid-run
leaves a fully-formed file up to the last completed row) and resume support
(restart reads the file, builds a skip-set, continues from the next unseen
work unit).

Design choices:

- **O_APPEND + single write() per row.** POSIX guarantees that an O_APPEND
  write atomically seeks to end-of-file before writing — no interleaving
  even with concurrent appenders (we don't currently use that, but it's
  the safety property). One JSON line per write() call, with `os.fsync`
  after each, so a crash leaves the file durable up to the last row.
- **Per-call open/close.** Each append re-opens the file. Slower than a
  long-lived handle, but means the offset is always end-of-file and we
  don't lose data on file-handle leak. Throughput here is one row per
  kernel-tune case (~seconds to minutes apart), so the cost is irrelevant.
- **Tolerant read.** A crashed write may leave a partial last line.
  `read_rows` skips malformed JSON with a stderr warning instead of
  aborting the projection step. Production rows are well-formed; this
  is purely for crash-recovery.
- **Skip-set with status filter.** Resume policy distinguishes permanent
  outcomes (SUCCESS / FAILED_OOM / SKIPPED — re-running wastes time or
  re-fails identically) from retryable ones (UNKNOWN_ERROR — possibly
  caused by a code bug that's since been fixed; should re-tune).
  Callers pass `status_filter={"SUCCESS", "FAILED_OOM", "SKIPPED"}`
  to skip those rows and retry UNKNOWN_ERROR.

Not a database. No indexes, no queries, no concurrent reads + writes from
the same process. The projection step reads the whole file once and emits
a winners JSON; that's the only consumer.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Iterator


def append_row(raw_path: Path, row: dict[str, Any]) -> None:
    """Append one row to a JSONL file, crash-atomically.

    Args:
      raw_path: Path to the `.raw/<sha>.jsonl` file. Parent dirs are
                created if missing.
      row: A JSON-serialisable dict. Written compactly (no spaces),
           terminated with a newline.

    Atomicity: opened with O_APPEND so the kernel seeks-to-end + writes
    in one operation; the single write() of a one-line JSON record is
    atomic on POSIX for small writes. `os.fsync` after the write
    flushes to disk so a kill / crash leaves the row durable.
    """
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(row, separators=(",", ":"), ensure_ascii=False)
            + "\n").encode("utf-8")
    fd = os.open(raw_path,
                 os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, line)
        os.fsync(fd)
    finally:
        os.close(fd)


def read_rows(raw_path: Path) -> Iterator[dict[str, Any]]:
    """Yield each row from a JSONL file. Missing/empty file → no rows.

    Tolerates a malformed final line (the partial-write-on-crash case):
    logs a stderr warning and skips, rather than aborting. Earlier
    well-formed rows are still emitted.
    """
    if not raw_path.exists():
        return
    with open(raw_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as e:
                print(
                    f"raw_store: skipping malformed row "
                    f"{raw_path}:{line_no}: {e}",
                    file=sys.stderr,
                )


def prune_raw_ttl(raw_dir: Path, keep: int = 2) -> list[Path]:
    """Delete all but the `keep` most-recent `.jsonl` files in `raw_dir`.

    Architecture doc §8 line 241: "TTL = 2. Keep the 2 most-recent
    kernel (or service) SHA's .raw files. Older ones are pruned.
    This survives one rollback (current → previous) without losing
    data."

    Recency is by mtime — the most-recent activity wins, not the
    SHA's git history. (The two usually agree, but a touch /
    re-tune of an older partition would refresh its mtime; that's
    intended.) Callers typically run this after a successful
    projection so a successful round leaves disk in steady state.

    Args:
      raw_dir: directory holding `<sha>.jsonl` files (e.g.
               `<workload>.kernel.raw/`). Missing dir → no-op.
      keep: number of most-recent files to retain. Default 2 per
            §8. `keep <= 0` is treated as "keep none" (deletes
            everything).

    Returns:
      List of paths that were deleted. Empty if nothing to prune.
    """
    if not raw_dir.exists():
        return []
    files = sorted(
        raw_dir.glob("*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    to_delete = files[max(keep, 0):]
    deleted: list[Path] = []
    for p in to_delete:
        try:
            p.unlink()
            deleted.append(p)
        except OSError as e:
            # Don't take down a tune just because an old prune
            # failed. Log and continue — next run retries.
            print(
                f"raw_store: prune failed for {p}: {e}",
                file=sys.stderr,
            )
    return deleted


def build_skip_set(
    raw_path: Path,
    key_fn: Callable[[dict[str, Any]], Any],
    status_filter: set[str] | None = None,
) -> set:
    """Build a set of keys from rows that should be skipped on resume.

    Args:
      raw_path: Path to the `.raw/<sha>.jsonl` file. Missing OK
                (returns empty set).
      key_fn: Pure function (row -> hashable key). The caller picks
              what identifies a "work unit" — typically a tuple of
              (tuning_key, tunable_params) or a frozen combo-dict.
      status_filter: Optional set of `row["status"]` values to treat
                     as permanent. Rows with other statuses are NOT
                     added to the skip-set (so they will be retried).
                     If None, every row contributes.

    Returns:
      A set of keys. Calling code: `if key_fn(combo) in skip_set: continue`.

    Reference: 35b570d7 fixed a bug where get_already_processed_ids
    returned every row regardless of status, so UNKNOWN_ERROR rows
    (transient code bugs) wedged the resume — kernel fixes never took
    effect because the broken case was "already processed". The
    status_filter param prevents that class of regression here.
    """
    out: set = set()
    for row in read_rows(raw_path):
        if status_filter is not None:
            status = row.get("status")
            if status not in status_filter:
                continue
        out.add(key_fn(row))
    return out
