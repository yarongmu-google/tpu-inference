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
"""Aggregate and rank results from a vLLM-bench sweep directory.

Walks `<sweep_dir>/<combo_id>/{metrics.txt, meta.txt}` produced by
sweep.py + run_benchmark.sh, parses the flat KEY=VAL files, and renders
a Markdown table. Supports ranking by a chosen metric and `--best-per`
to show the winner per unique value of a meta-field axis (e.g., per K).
"""

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any


DEFAULT_METRIC = "RequestThroughput"

# Default columns for the rendered table. Each entry is
# ('<section>.<field>' or 'combo_id'/'result_dir', <header label>).
DEFAULT_COLUMNS: list[tuple[str, str]] = [
    ("meta.case_name", "case"),
    ("meta.max_num_batched_tokens", "MNB"),
    ("meta.max_num_seqs", "MNS"),
    ("meta.max_model_len", "MML"),
    ("meta.block_size", "page"),
    ("meta.long_prefill_token_threshold", "K"),
    ("meta.rpa_p_block_sizes", "p_blk"),
    ("metrics.RequestThroughput", "req/s"),
    ("metrics.OutputTokenThroughput", "out_tok/s"),
    ("metrics.MeanTTFT", "ttft_mean"),
    ("metrics.P99TTFT", "ttft_p99"),
    ("metrics.MeanITL", "itl_mean"),
    ("metrics.P99ITL", "itl_p99"),
    ("combo_id", "id"),
]


# ----- File parsing --------------------------------------------------------


def parse_kv_file(path: str | os.PathLike) -> dict[str, str]:
    """Read KEY=VALUE lines into a dict. Skip blanks and lines without '='.

    Used for both metrics.txt (parse_bench_log output) and meta.txt
    (run_benchmark.sh's run-config dump).
    """
    out: dict[str, str] = {}
    with open(path) as f:
        for raw in f:
            line = raw.rstrip("\n").rstrip("\r")
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k] = v
    return out


def collect_results(sweep_dir: str | os.PathLike) -> list[dict[str, Any]]:
    """Walk `sweep_dir/<combo_id>/` and parse metrics.txt + meta.txt.

    A combo with no metrics.txt is skipped (treated as not yet completed).
    Missing meta.txt is tolerated (its dict is empty).
    """
    sdir = Path(sweep_dir)
    out: list[dict[str, Any]] = []
    if not sdir.is_dir():
        return out
    for combo_dir in sorted(sdir.iterdir()):
        if not combo_dir.is_dir():
            continue
        metrics_path = combo_dir / "metrics.txt"
        meta_path = combo_dir / "meta.txt"
        if not metrics_path.is_file():
            continue
        out.append({
            "combo_id": combo_dir.name,
            "result_dir": combo_dir,
            "metrics": parse_kv_file(metrics_path),
            "meta": parse_kv_file(meta_path) if meta_path.is_file() else {},
        })
    return out


# ----- Ranking -------------------------------------------------------------


def _to_float(s: Any) -> float | None:
    """Parse a string-like value as a finite float, else None.

    Treats NaN and ±inf as 'not a usable number':
      - NaN comparisons return False for <, >, ==, breaking the
        strict-weak-ordering contract sorted() requires; with a NaN
        key, sort order is unspecified and may differ between calls.
      - inf is well-ordered, but a metric value of inf is more likely
        a bug (division by zero in the bench) than a real datapoint;
        treating it as missing is safer than letting it sort to top.
    """
    if s is None or s == "":
        return None
    try:
        v = float(s)
    except (ValueError, TypeError):
        return None
    if not math.isfinite(v):
        return None
    return v


def rank_by(
    results: list[dict[str, Any]],
    metric: str,
    *,
    descending: bool = True,
) -> list[dict[str, Any]]:
    """Sort results by metrics[metric]. Missing/non-numeric go last."""

    def sort_key(r):
        v = _to_float(r["metrics"].get(metric))
        if v is None:
            return (1, 0.0)
        return (0, -v if descending else v)

    return sorted(results, key=sort_key)


def best_per_axis(
    results: list[dict[str, Any]],
    axis_key: str,
    metric: str,
    *,
    descending: bool = True,
) -> dict[str, dict[str, Any]]:
    """For each unique meta[axis_key], return the best-by-metric result.

    Combos missing meta[axis_key] are dropped. Within each group, combos
    with a non-numeric / missing metric are dropped from contention; if
    the whole group has none, that key is omitted from the output.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        v = r["meta"].get(axis_key)
        if v is None or v == "":
            continue
        groups.setdefault(v, []).append(r)
    best: dict[str, dict[str, Any]] = {}
    for k, group in groups.items():
        ranked = rank_by(group, metric, descending=descending)
        if ranked and _to_float(ranked[0]["metrics"].get(metric)) is not None:
            best[k] = ranked[0]
    return best


# ----- Rendering -----------------------------------------------------------


def _extract_field(result: dict[str, Any], key: str) -> str:
    """Look up a field path in a result dict, return as a Markdown-safe str.

    Pipe characters in cell values are escaped (`|` -> `\\|`) so a value
    containing a pipe doesn't break the table layout. Pure-data values
    in our pipeline don't contain pipes today, but defending is cheap.
    """
    raw: Any
    if key == "combo_id":
        raw = result.get("combo_id", "")
    elif key == "result_dir":
        raw = result.get("result_dir", "")
    elif "." in key:
        section, field = key.split(".", 1)
        section_dict = result.get(section, {}) or {}
        raw = section_dict.get(field, "")
    else:
        raw = result.get(key, "")
    return str(raw).replace("|", r"\|")


def format_markdown_table(
    results: list[dict[str, Any]],
    columns: list[tuple[str, str]],
) -> str:
    """Render `results` as a left-aligned Markdown table with the columns."""
    if not results:
        return "(no results)"

    headers = [label for _, label in columns]
    rows: list[list[str]] = []
    for r in results:
        rows.append([_extract_field(r, key) for key, _ in columns])

    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]

    def render(row):
        return "| " + " | ".join(row[i].ljust(widths[i])
                                 for i in range(len(row))) + " |"

    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    return "\n".join([render(headers), sep, *(render(r) for r in rows)])


# ----- CLI -----------------------------------------------------------------


def _columns_from_arg(arg: str | None) -> list[tuple[str, str]]:
    if not arg:
        return DEFAULT_COLUMNS
    cols: list[tuple[str, str]] = []
    for piece in arg.split(","):
        key = piece.strip()
        if not key:
            continue
        label = key.split(".", 1)[-1] if "." in key else key
        cols.append((key, label))
    return cols


def _sort_axis_keys(
    keys: list[str],
) -> list[str]:
    """Sort axis values: numeric ones first (ascending), then string-sorted rest.

    Avoids the `or 0` falsy-mask pattern: `_to_float("-0.0")` returns
    `-0.0` which is falsy in Python, so `(... or 0)` would map negative
    zero to positive zero — harmless for ordering but inconsistent with
    the spec/runner code paths where we replaced this idiom. Use an
    explicit conditional expression instead.
    """
    def axis_key(x: str) -> tuple[bool, float, str]:
        n = _to_float(x)
        return (n is None, n if n is not None else 0.0, x)
    return sorted(keys, key=axis_key)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Aggregate + rank results of a vLLM-bench sweep.")
    p.add_argument("sweep_dir",
                   help="Sweep dir, e.g. tmp/bench_<case>_<sweep>/.")
    p.add_argument("--metric", default=DEFAULT_METRIC,
                   help=f"Metric to rank by (default: {DEFAULT_METRIC}).")
    p.add_argument("--ascending", action="store_true",
                   help="Sort ascending instead of descending.")
    p.add_argument("--best-per", default=None,
                   help="Show only the best result per unique value of "
                        "meta[<axis>] (e.g. long_prefill_token_threshold).")
    p.add_argument("--columns", default=None,
                   help="Comma-separated column keys to display. Each key is "
                        "'<section>.<field>' (section in {meta, metrics}) or "
                        "'combo_id' / 'result_dir'. Defaults to a sensible set.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    results = collect_results(args.sweep_dir)
    if not results:
        print(f"No results in {args.sweep_dir}", file=sys.stderr)
        return 1

    columns = _columns_from_arg(args.columns)
    descending = not args.ascending

    if args.best_per is not None:
        best = best_per_axis(results, axis_key=args.best_per,
                             metric=args.metric, descending=descending)
        ordered = [best[k] for k in _sort_axis_keys(list(best.keys()))]
        results_to_print = ordered
    else:
        results_to_print = rank_by(results, args.metric, descending=descending)

    print(format_markdown_table(results_to_print, columns))
    return 0


if __name__ == "__main__":
    sys.exit(main())
