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
"""Parse `vllm bench serve` output into a flat key=value summary.

Used by run_benchmark.sh to write metrics.txt. Kept pure-Python so it
gets unit-tested independently of the surrounding shell driver.
"""

import argparse
import re
import sys
from typing import Iterable, Mapping

# Latency sections that vllm bench reports under "X (ms):" with rows like
# "Mean X (ms): ...", "Median X (ms): ...", "P99 X (ms): ...".
LATENCY_SECTIONS = ("TTFT", "TPOT", "ITL", "E2EL")
LATENCY_STATS = ("Mean", "Median", "P99")

# Single-line metrics that appear once each.
THROUGHPUT_METRICS = (
    ("RequestThroughput", "Request throughput (req/s):"),
    ("OutputTokenThroughput", "Output token throughput (tok/s):"),
    ("TotalTokenThroughput", "Total Token throughput (tok/s):"),
)


def _last_token(line: str) -> str:
    """Return the last whitespace-separated token of a line, or empty.

    Honest about the contract: this is *not* a numeric extractor. For
    well-formed vllm output the last token happens to be the metric value,
    but the helper does no numeric validation — a trailing parenthetical
    or unit would be returned verbatim. Callers either trust the input
    shape or validate downstream.
    """
    parts = line.strip().split()
    return parts[-1] if parts else ""


def parse_latency_metrics(
    lines: Iterable[str],
    sections: Iterable[str] = LATENCY_SECTIONS,
    stats: Iterable[str] = LATENCY_STATS,
) -> dict[str, str]:
    """Extract per-(stat, section) numbers from a list of bench-log lines.

    For each (stat, section), find a line that:
      - contains the substring ``f"{section} (ms):"``, AND
      - matches the regex ``\\b{stat}\\b.*\\b{section}\\b`` (word-boundary
        on both ends, so ``(stat, section)`` may be separated by any
        intervening characters; section-header lines like ``"TTFT (ms):"``
        with no preceding stat token don't match).
    Take the last whitespace-separated token of that line as the value.

    Returns a dict like {"MeanTTFT": "12.3", "P99E2EL": "999.0", ...}.
    Missing values are recorded as empty string. Key order is
    section-major (see parse_bench_log docstring).
    """
    sections = tuple(sections)
    stats = tuple(stats)
    out: dict[str, str] = {f"{stat}{sec}": "" for sec in sections for stat in stats}
    for line in lines:
        for sec in sections:
            section_marker = f"{sec} (ms):"
            if section_marker not in line:
                continue
            for stat in stats:
                if re.search(rf"\b{re.escape(stat)}\b.*\b{re.escape(sec)}\b", line):
                    out[f"{stat}{sec}"] = _last_token(line)
    return out


def parse_throughput_metrics(
    lines: Iterable[str],
    metrics: Iterable[tuple[str, str]] = THROUGHPUT_METRICS,
) -> dict[str, str]:
    """Extract single-line throughput metrics keyed by name.

    `metrics` is an iterable of (output_key, search_substring). For each
    metric, the first line whose lstrip()ped form *starts* with the
    substring contributes its last token as the value. The start-anchor
    guards against a stray diagnostic line (e.g. "WARNING: Request
    throughput (req/s): too few samples — unreliable") swallowing the
    real metric.
    """
    out: dict[str, str] = {key: "" for key, _ in metrics}
    for line in lines:
        stripped = line.lstrip()
        for key, marker in metrics:
            if out[key] == "" and stripped.startswith(marker):
                out[key] = _last_token(line)
    return out


def parse_bench_log(text: str) -> dict[str, str]:
    """Top-level: parse a vllm-bench-serve stdout blob into a flat dict.

    Combines latency + throughput extraction. Key order is deterministic
    (and pinned by tests): for each section in (TTFT, TPOT, ITL, E2EL)
    we emit (Mean, Median, P99) — section-major — followed by the
    throughput keys in their declared order.
    """
    lines = text.splitlines()
    out: dict[str, str] = {}
    out.update(parse_latency_metrics(lines))
    out.update(parse_throughput_metrics(lines))
    return out


def format_metrics(metrics: Mapping[str, str]) -> str:
    """Render a metrics dict as deterministic `KEY=VALUE\n` lines."""
    return "".join(f"{k}={v}\n" for k, v in metrics.items())


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Parse a vllm bench serve log into KEY=VALUE lines.")
    p.add_argument("bench_log", type=str,
                   help="Path to bench.log (the captured stdout of "
                        "`vllm bench serve`). Use - for stdin.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.bench_log == "-":
        text = sys.stdin.read()
    else:
        with open(args.bench_log) as f:
            text = f.read()
    sys.stdout.write(format_metrics(parse_bench_log(text)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
