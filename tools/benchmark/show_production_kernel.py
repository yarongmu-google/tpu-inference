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


def _shorten(val) -> str:
    """Truncate long values (code_revision) for table display."""
    if val is None:
        return "-"
    s = str(val)
    if len(s) > 12:
        return s[:12]
    return s


def _print_table(case: str, entries: list, *, diff_only: bool) -> None:
    rows = [_flatten(e) for e in entries]
    all_cols = _TUNING_KEY_COLS + _TUNABLE_COLS + _RESULT_COLS

    # Which columns vary across rows?
    if diff_only:
        varying = [
            c for c in all_cols
            if len({_shorten(r.get(c)) for r in rows}) > 1
        ]
        cols = varying or all_cols  # avoid printing nothing
    else:
        # Skip columns entirely missing across all rows.
        cols = [c for c in all_cols if any(c in r for r in rows)]

    # Compute widths.
    widths = {}
    for c in cols:
        widths[c] = max(len(c), max((len(_shorten(r.get(c))) for r in rows),
                                    default=0))

    sep_line = "  " + "  ".join("-" * widths[c] for c in cols)

    print(f"\n=== {case.upper()} ({len(entries)} entries"
          f"{' — diff-only' if diff_only else ''}) ===")
    if diff_only and len(cols) < len(all_cols):
        hidden = [c for c in all_cols if c not in cols and any(c in r for r in rows)]
        print(f"  (hidden: identical across all rows → {', '.join(hidden)})")
    print()
    print("  " + "  ".join(f"{c:<{widths[c]}}" for c in cols))
    print(sep_line)
    for r in rows:
        print("  " + "  ".join(
            f"{_shorten(r.get(c)):<{widths[c]}}" for c in cols))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", help="Path to production.kernel JSON")
    p.add_argument("--case",
                   help="Filter to one case (decode/mixed/prefill/logical). "
                        "Default: all cases.")
    p.add_argument("--diff-only", action="store_true",
                   help="Hide columns where every row carries the same value.")
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
        _print_table(case, results[case], diff_only=args.diff_only)

    return 0


if __name__ == "__main__":
    sys.exit(main())
