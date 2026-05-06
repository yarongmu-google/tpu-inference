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
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from tools.benchmark._schema import (
    META_FILENAME, METRICS_FILENAME, THROUGHPUT_METRIC,
)


DEFAULT_METRIC = THROUGHPUT_METRIC

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
    ("meta.request_rate", "rate"),
    ("metrics.RequestThroughput", "req/s"),
    ("metrics.OutputTokenThroughput", "out_tok/s"),
    ("metrics.MeanTTFT", "ttft_mean"),
    ("metrics.P99TTFT", "ttft_p99"),
    ("metrics.MeanITL", "itl_mean"),
    ("metrics.P99ITL", "itl_p99"),
    ("meta.bench_duration_seconds", "dur_s"),
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

    Note the deliberate asymmetry vs sweep.is_completed: this function
    includes combos whose metrics.txt is present but blank (RequestThroughput
    empty), so failed runs show up in compare's table — that's where you'd
    want to see them. sweep.is_completed is the stricter check that drives
    skip-resumability; it requires non-empty RequestThroughput so a sweep
    re-runs combos that died mid-write.

    Race window: this function reads metrics.txt files while a sweep may
    still be writing them. run_benchmark.sh redirects via `>` (atomic-ish
    on most filesystems for small files); the worst case is one combo's
    row showing blanks in the table because we caught its metrics.txt
    mid-write. Compare is intended to run *after* a sweep finishes, not
    concurrently.
    """
    sdir = Path(sweep_dir)
    out: list[dict[str, Any]] = []
    if not sdir.is_dir():
        return out
    for combo_dir in sorted(sdir.iterdir()):
        if not combo_dir.is_dir():
            continue
        metrics_path = combo_dir / METRICS_FILENAME
        meta_path = combo_dir / META_FILENAME
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
    """Render `results` as a left-aligned Markdown table with the columns.

    Uses ``:---`` separators (rather than bare ``---``) so renderers
    that distinguish default vs. explicit alignment (Pandoc, VS Code
    preview) keep cells left-aligned to match the ljust padding the
    function applies to data cells.
    """
    if not results:
        return "(no results)"

    # Escape pipes in headers as well as data — _columns_from_arg can
    # derive labels from user-supplied --columns input (e.g. a key with
    # a literal '|' in it), so trusting the label is wrong even though
    # DEFAULT_COLUMNS and current callers happen to be pipe-free.
    headers = [label.replace("|", r"\|") for _, label in columns]
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

    # `:---` (with the colon on the left) is the explicit left-alignment
    # spec in GFM. Width is (w + 2) total, so we want ":" + "-" * (w + 1).
    sep = "|" + "|".join(":" + "-" * (w + 1) for w in widths) + "|"
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
    p.add_argument("--export-production", default=None,
                   help="Path to export the absolute best configuration as a .service JSON file "
                        "(e.g., tools/benchmark/cases/v7x/llama3_8b/production.service). "
                        "If the file exists, it will accumulate the new workload shape.")
    p.add_argument("--kernel-id", default="unknown",
                   help="Kernel identifier baked into the workload_key so "
                        "results from different kernels do not collide in "
                        "the production.service file (default: unknown).")
    p.add_argument("--service-id", default="unknown",
                   help="Service identifier baked into the workload_key "
                        "(default: unknown).")
    return p


def _make_workload_key(meta: dict[str, str],
                       kernel_id: str = "unknown",
                       service_id: str = "unknown") -> str:
    """Compose a workload-shape key that does not collide across kernels,
    services, models, or TP sizes.

    Earlier this was just `f"{input_len}_in_{output_len}_out"`. The
    moment a second kernel (e.g. RPA v4) or a second service (SGLang)
    landed, results from the second sweep silently overwrote the first
    because their (in,out) shapes happened to match. Key on the full
    shape now so each (kernel, service, model, TP, in, out) tuple gets
    its own slot.
    """
    return (
        f"{kernel_id}__{service_id}__"
        f"{meta.get('model', 'unknown')}__"
        f"tp{meta.get('tensor_parallel_size', '1')}__"
        f"{meta.get('input_len', 'unknown')}_in_"
        f"{meta.get('output_len', 'unknown')}_out"
    )


def export_production_registry(
    best_result: dict[str, Any],
    output_path: str | os.PathLike,
    kernel_id: str = "unknown",
    service_id: str = "unknown",
) -> None:
    """Export the best combo into an accumulated production .service JSON.

    Three data-loss hazards the earlier implementation tripped on:

      1. `except Exception: pass` on the existing-file load silently
         dropped weeks of accumulated best-configs whenever the prior
         file was corrupt or partial. We now log the failure loudly
         AND refuse to overwrite a non-empty existing file we could
         not parse — better to fail than to nuke history.
      2. `open(out_path, "w")` truncates the file immediately; a crash
         mid-`json.dump` left a corrupt artifact. Now: write to a
         sibling tmp file, fsync, and `os.replace` (atomic on POSIX).
      3. `_to_float(...) or 0.0` treated a legitimate 0.0 as "missing"
         (`0.0 or 0.0` -> 0.0; same as None or non-numeric). The
         comparison `new_tp > 0.0` then over-eagerly accepted any
         positive value over a true 0.0 baseline. Now: explicit
         `is None` check, fall back to "no-existing-data so accept new"
         only when the existing entry has no parseable metric.
    """
    out_path = Path(output_path)
    data: dict[str, Any] = {}
    if out_path.is_file():
        try:
            with open(out_path) as f:
                data = json.load(f)
        except Exception as e:
            print(
                f"ERROR: existing {out_path} is unreadable ({e}). "
                "Refusing to overwrite — back up the file and re-run, "
                "or delete it manually if the loss is acceptable.",
                file=sys.stderr)
            raise

    meta = best_result.get("meta", {})
    metrics = best_result.get("metrics", {})

    workload_key = _make_workload_key(meta, kernel_id, service_id)

    if "best_configs_by_workload" not in data:
        data["best_configs_by_workload"] = {}

    entry = {
        "kernel_id": kernel_id,
        "service_id": service_id,
        "model": meta.get("model"),
        "tensor_parallel_size": meta.get("tensor_parallel_size"),
        "MAX_NUM_SEQS": meta.get("max_num_seqs"),
        "MAX_NUM_BATCHED_TOKENS": meta.get("max_num_batched_tokens"),
        "MAX_MODEL_LEN": meta.get("max_model_len"),
        "BLOCK_SIZE": meta.get("block_size"),
        "LONG_PREFILL_TOKEN_THRESHOLD": meta.get("long_prefill_token_threshold"),
        "RPA_P_BLOCK_SIZES": meta.get("rpa_p_block_sizes"),
        "RPA_D_BLOCK_SIZES": meta.get("rpa_d_block_sizes"),
        "RPA_M_BLOCK_SIZES": meta.get("rpa_m_block_sizes"),
        "metrics": metrics,
    }

    # Only overwrite if new throughput is strictly better. Use explicit
    # `is None` instead of `or 0.0` so a real 0.0 doesnt round-trip
    # through the falsy-mask and look like missing data.
    existing = data["best_configs_by_workload"].get(workload_key)
    should_write = True
    if existing is not None:
        old_tp = _to_float(existing.get("metrics", {}).get(DEFAULT_METRIC))
        new_tp = _to_float(metrics.get(DEFAULT_METRIC))
        if new_tp is None:
            # New result has no parseable metric — never overwrite a
            # known-good entry with a missing one.
            should_write = False
        elif old_tp is not None and new_tp <= old_tp:
            should_write = False
    if should_write:
        data["best_configs_by_workload"][workload_key] = entry

    # Atomic write: stage to a sibling tmp file, then os.replace.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=out_path.name + ".",
        suffix=".tmp",
        dir=str(out_path.parent),
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, out_path)
    except Exception:
        # Best-effort cleanup of the staging file; reraise the original.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    print(f"\nExported absolute best configuration to: {out_path}")


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

    if args.export_production and results_to_print:
        export_production_registry(
            results_to_print[0],
            args.export_production,
            kernel_id=args.kernel_id,
            service_id=args.service_id,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
