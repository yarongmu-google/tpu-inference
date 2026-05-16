"""Render a `production.kernel` JSON file as per-case tables.

Usage:
    python3 -m tools.benchmark.show_production_kernel <path> [options]
    python3 -m tools.benchmark.show_production_kernel \
        tools/benchmark/cases/v7x/llama3_8b/production.kernel
    python3 -m tools.benchmark.show_production_kernel <path> --case logical
    python3 -m tools.benchmark.show_production_kernel <path> --diff-only

The file accumulates one entry per (tuning_key) — typically one per
code_revision, since the rest of the tuning_key is workload-fixed.
Multiple entries for the same case show what changed across revisions:
which tunable_params the tuner converged on (often the same shape
across revs = strong "this is the sweet spot" signal), how latency
moved, etc. `--diff-only` hides columns where every row carries the
same value, so the eye lands on what actually varies.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


# Column order: tuning_key fields first (workload signature), then
# tunable_params (what the tuner picked), then result fields.
# Within each group, ordered by what changes most across entries
# (rough heuristic; --diff-only lets the user re-sort by inspection).
_TUNING_KEY_COLS = [
    "page_size", "q_dtype", "kv_dtype",
    "num_q_heads", "num_kv_heads", "head_dim",
    "max_model_len", "sliding_window",
    "case", "chunk_prefill_size", "code_revision",
]
_TUNABLE_COLS = ["bq_sz", "bkv_sz", "bq_csz", "bkv_csz", "max_num_subseqs"]
_RESULT_COLS = ["Latency", "WarmupTime", "CaseId"]


def _flatten(entry: dict) -> dict:
    """Flatten {tuning_key:{}, tunable_params:{}, Latency:N, ...} → one dict."""
    flat = {}
    flat.update(entry.get("tuning_key", {}))
    flat.update(entry.get("tunable_params", {}))
    for k in _RESULT_COLS:
        if k in entry:
            flat[k] = entry[k]
    return flat


def _commit_date(rev: str | None) -> str:
    """Return YYYY-MM-DD HH:MM for a git commit, or '?' if not available."""
    if not rev:
        return "?"
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%ai", rev],
            capture_output=True, text=True, check=True, timeout=5)
        # %ai → "2026-05-10 17:19:36 -0700"; trim to date+time, drop tz.
        return out.stdout.strip()[:16]
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            FileNotFoundError):
        return "?"


def _sort_and_truncate(entries: list, *, sort_by_recency: bool,
                       ttl: int | None) -> tuple[list, dict]:
    """Optionally sort entries newest-first (by code_revision commit date)
    and truncate to the most recent TTL.

    Returns the (possibly reordered/truncated) list, and a dict mapping
    each surviving entry's id() to its commit date string (so the caller
    can render a commit_date column without re-querying git).
    """
    date_map = {}
    if sort_by_recency or ttl is not None:
        # Fetch commit dates for sorting / display.
        for e in entries:
            rev = e.get("tuning_key", {}).get("code_revision")
            date_map[id(e)] = _commit_date(rev)
        # Sort newest-first. Entries with no rev or '?' date sort last
        # (treat as oldest / unknown).
        entries = sorted(
            entries,
            key=lambda e: date_map.get(id(e), "?"),
            reverse=True,
        )
    if ttl is not None and ttl > 0:
        entries = entries[:ttl]
    return entries, date_map


def _shorten(val, *, col: str | None = None) -> str:
    """Stringify a column value for table display.

    Only `code_revision` is truncated (to 12 chars — the conventional
    short hash length). Other fields (including `commit_date` which
    is 16 chars) print in full so diff-only comparisons see the real
    value, not a truncated prefix that happens to match across rows.
    """
    if val is None:
        return "-"
    s = str(val)
    if col == "code_revision" and len(s) > 12:
        return s[:12]
    return s


def _print_table(case: str, entries: list, *, diff_only: bool,
                 sort_by_recency: bool, ttl: int | None) -> None:
    # Optional sort + TTL truncate. date_map maps id(entry) → commit date.
    total = len(entries)
    entries, date_map = _sort_and_truncate(
        entries, sort_by_recency=sort_by_recency, ttl=ttl)

    rows = []
    for e in entries:
        r = _flatten(e)
        if date_map:
            r["commit_date"] = date_map.get(id(e), "?")
        rows.append(r)

    all_cols = _TUNING_KEY_COLS + _TUNABLE_COLS + _RESULT_COLS
    if date_map:
        # Insert commit_date right after code_revision for readability.
        idx = all_cols.index("code_revision") + 1
        all_cols = all_cols[:idx] + ["commit_date"] + all_cols[idx:]

    # Which columns vary across rows?
    if diff_only:
        varying = [
            c for c in all_cols
            if len({_shorten(r.get(c), col=c) for r in rows}) > 1
        ]
        cols = varying or all_cols  # avoid printing nothing
    else:
        # Skip columns entirely missing across all rows.
        cols = [c for c in all_cols if any(c in r for r in rows)]

    # Compute widths.
    widths = {}
    for c in cols:
        widths[c] = max(len(c), max((len(_shorten(r.get(c), col=c)) for r in rows),
                                    default=0))

    sep_line = "  " + "  ".join("-" * widths[c] for c in cols)

    title_extras = []
    if sort_by_recency:
        title_extras.append("sorted newest→oldest")
    if ttl is not None:
        title_extras.append(f"ttl={ttl} of {total}")
    if diff_only:
        title_extras.append("diff-only")
    suffix = f" — {', '.join(title_extras)}" if title_extras else ""

    print(f"\n=== {case.upper()} ({len(entries)} entries{suffix}) ===")
    if diff_only and len(cols) < len(all_cols):
        hidden = [c for c in all_cols if c not in cols
                  and any(c in r for r in rows)]
        print(f"  (hidden: identical across all rows → {', '.join(hidden)})")
    if ttl is not None and total > len(entries):
        print(f"  (truncated: {total - len(entries)} older entries hidden by --ttl)")
    print()
    print("  " + "  ".join(f"{c:<{widths[c]}}" for c in cols))
    print(sep_line)
    for r in rows:
        print("  " + "  ".join(
            f"{_shorten(r.get(c), col=c):<{widths[c]}}" for c in cols))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", help="Path to production.kernel JSON")
    p.add_argument("--case",
                   help="Filter to one case (decode/mixed/prefill/logical). "
                        "Default: all cases.")
    p.add_argument("--diff-only", action="store_true",
                   help="Hide columns where every row carries the same value.")
    p.add_argument("--sort-by-recency", action="store_true",
                   help="Sort entries newest→oldest using each entry's "
                        "code_revision commit date (queried via `git log`). "
                        "Adds a `commit_date` column. Implied by --ttl.")
    p.add_argument("--ttl", type=int, default=None, metavar="N",
                   help="Show only the N most recent entries per case "
                        "(by code_revision commit date). Display-only; "
                        "does NOT modify the file. Use to preview what a "
                        "TTL-based prune would keep.")
    args = p.parse_args(argv)

    path = Path(args.path)
    with path.open() as f:
        data = json.load(f)

    meta = data.get("metadata", {})
    print(f"File:               {path}")
    print(f"last_updated_at:    {meta.get('last_updated_at', '?')}")
    print(f"source runlog:      {meta.get('last_updated_from_runlog', '?')}")

    results = data.get("results", {})
    cases = [args.case] if args.case else list(results.keys())
    for case in cases:
        if case not in results:
            print(f"\n(no entries for case={case})", file=sys.stderr)
            continue
        sort_by_recency = args.sort_by_recency or args.ttl is not None
        _print_table(case, results[case], diff_only=args.diff_only,
                     sort_by_recency=sort_by_recency, ttl=args.ttl)

    return 0


if __name__ == "__main__":
    sys.exit(main())
