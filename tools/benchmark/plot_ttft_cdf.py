#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Plot per-request TTFT CDF(s) from vllm bench --save-detailed runs.

Usage:
    python tools/benchmark/plot_ttft_cdf.py <input1> [<input2> ...] \\
        [--labels LABEL1 [LABEL2 ...]] \\
        [--out ttft_cdf.png] \\
        [--title "..."] \\
        [--linear]

Each <input> is either:
  - a path to a vllm `detailed.json` file (--save-result --save-detailed), or
  - a path to a bench directory containing `detailed.json` at top level.

Output is a PNG (or any matplotlib-supported format inferred from
the --out extension) with one CDF curve per input, log-x by default.

Stdout/logging emits a percentile table per input alongside the plot.
"""
import argparse
import json
import logging
import pathlib
import sys

logger = logging.getLogger("plot_ttft_cdf")


def _resolve_detailed_json(path: pathlib.Path) -> pathlib.Path:
    """Accept a JSON path or a bench-result dir; return the JSON path.

    The vllm bench harness writes `<result-dir>/<result-filename>`.
    Our run_benchmark.sh fixes the filename to `detailed.json`. If a
    user points us at the dir, find that file; if they point us at the
    file, use it directly.
    """
    if path.is_file():
        return path
    if path.is_dir():
        candidate = path / "detailed.json"
        if candidate.is_file():
            return candidate
        # Fall back to any *detailed*.json so users who picked a
        # different --result-filename aren't blocked.
        matches = sorted(path.glob("*detailed*.json"))
        if matches:
            return matches[0]
        raise FileNotFoundError(
            f"No detailed.json in {path}. Re-run the bench with "
            "`--save-result --save-detailed` (run_benchmark.sh now "
            "does this automatically since 2026-05-11)."
        )
    raise FileNotFoundError(f"{path} is neither file nor directory")


def load_ttfts_seconds(path: pathlib.Path) -> list[float]:
    """Return per-request TTFT in seconds for a vllm detailed.json.

    vllm dumps ttfts in seconds already (see
    `vllm/benchmarks/serve.py` line 982: `[output.ttft for output in
    outputs]`, where `output.ttft` is the wall-clock difference
    captured by the async client). We preserve that unit and let the
    caller decide what to plot.
    """
    resolved = _resolve_detailed_json(path)
    with open(resolved) as f:
        blob = json.load(f)
    if "ttfts" not in blob:
        raise ValueError(
            f"{resolved} has no 'ttfts' key — was --save-detailed "
            "passed to vllm bench? --save-result alone is insufficient."
        )
    ttfts = list(blob["ttfts"])
    if not ttfts:
        raise ValueError(f"{resolved} has empty 'ttfts' list")
    return ttfts


def percentile_table(ttfts_s: list[float]) -> list[tuple[int, float]]:
    """Return [(percentile, value_seconds), ...] for canonical pcts.

    Returned in ascending-percentile order. Uses numpy for the
    percentile math because numpy is a hard project dep and the linear-
    interpolation contract for percentile() is the conventional one
    most users expect; a hand-rolled version would be a source of
    subtle off-by-one disagreements.
    """
    try:
        import numpy as np
    except ImportError as e:
        raise ImportError(
            "plot_ttft_cdf needs numpy (already a project dep). "
            "Activate the project venv or `pip install numpy`."
        ) from e
    pcts = [10, 25, 50, 75, 90, 99]
    return [(p, float(np.percentile(ttfts_s, p))) for p in pcts]


def _default_label(path: pathlib.Path) -> str:
    if path.is_file():
        # Bench dir name is more informative than 'detailed' (the
        # filename). Walk up to the dir that contained the json.
        return path.parent.name or path.stem
    return path.name


def plot_cdfs(
    inputs: list[pathlib.Path],
    out_path: pathlib.Path,
    labels: list[str] | None,
    title: str,
    log_x: bool,
) -> None:
    """Render the CDF figure and write it to out_path.

    Side effects: log per-input percentile tables at INFO. Returns
    nothing — the artifact is the file at out_path.
    """
    # Lazy-import matplotlib so the loader/percentile functions stay
    # importable in environments without it (e.g. CI containers that
    # only run the no-dep unit tests). If a user actually invokes
    # plot_cdfs without matplotlib installed, raise with the install
    # hint rather than the bare ModuleNotFoundError.
    try:
        import matplotlib
    except ImportError as e:
        raise ImportError(
            "plot_ttft_cdf needs matplotlib to render plots. "
            "Install with: pip install matplotlib"
        ) from e
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, path in enumerate(inputs):
        ttfts = load_ttfts_seconds(path)
        ttfts_sorted = sorted(ttfts)
        n = len(ttfts_sorted)
        ys = [(j + 1) / n for j in range(n)]
        label = (
            labels[i] if labels else _default_label(path)
        )
        ax.plot(ttfts_sorted, ys, label=label, linewidth=2)
        logger.info("%s: n=%d reqs", label, n)
        for p, v in percentile_table(ttfts_sorted):
            logger.info("  P%-2d  %8.3fs", p, v)

    ax.set_xlabel("TTFT (seconds)")
    ax.set_ylabel("CDF (fraction of requests ≤ x)")
    ax.set_title(title)
    if log_x:
        ax.set_xscale("log")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    logger.info("wrote %s", out_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Plot per-request TTFT CDFs from vllm bench detailed.json"
        ),
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=pathlib.Path,
        help=(
            "Paths to detailed.json files OR bench result dirs "
            "containing detailed.json at the top level."
        ),
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help=(
            "Label per input (default: parent dir name). Must match "
            "the number of inputs if provided."
        ),
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("ttft_cdf.png"),
        help="Output image path (default: ttft_cdf.png).",
    )
    parser.add_argument(
        "--title",
        default="TTFT CDF — per-request",
        help="Plot title.",
    )
    parser.add_argument(
        "--linear",
        action="store_true",
        help=(
            "Use a linear x-axis. Default is log (better for "
            "saturated-load runs where the lucky 10%% finish in <5s "
            "and the unlucky 1%% wait 200+s)."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stderr,
    )

    if args.labels and len(args.labels) != len(args.inputs):
        parser.error(
            f"--labels has {len(args.labels)} entries but there are "
            f"{len(args.inputs)} inputs"
        )

    plot_cdfs(
        inputs=args.inputs,
        out_path=args.out,
        labels=args.labels,
        title=args.title,
        log_x=not args.linear,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
