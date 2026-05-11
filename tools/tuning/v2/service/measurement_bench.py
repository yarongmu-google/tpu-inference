# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Bench-side `measurement_fn` adapter for the service sweep.

Architecture doc §13.2: each kernel-plugin's hardware-bound adapter
satisfies the v2 `measurement_fn(combo) -> result_dict` contract.
For the service layer, the adapter drives `tools/benchmark/run_benchmark.sh`
(which wraps `vllm bench serve`), parses the bench output, and
returns the v2 metrics dict the projection step expects.

Why an adapter instead of a port: `run_benchmark.sh` already handles
the vLLM-server lifecycle (start, warmup, bench, kill), result-dir
management, and metrics persistence. Re-implementing that in Python
v2 would duplicate ~400 lines of bash logic plus the implicit
contract with `vllm bench serve` flag evolution. The adapter is the
thin shim — set env vars, run the shell, parse output.

Output mapping (v1 bench → v2 metric names, matches
service/project.DEFAULT_OBJECTIVES):
  RequestThroughput → metrics.req_per_sec     (throughput_max)
  MeanTTFT          → metrics.ttft_mean_ms    (ttft_min)
  P99TTFT           → metrics.ttft_p99_ms     (p99_min)

Status mapping:
  rc=0 + metrics parse → SUCCESS
  rc != 0              → FAILED (bench-tool failure)
  timeout              → FAILED_OOM (timeout almost always = OOM
                                     downstream of the kernel)
  subprocess raise     → UNKNOWN_ERROR (caller may retry)
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable


# Default timeout per combo. Long enough to cover a full bench
# (warmup + measurement) at the high-MNB end of the sweep; short
# enough that a wedged server doesn't burn the whole sweep window.
DEFAULT_TIMEOUT_SECONDS = 1800   # 30 min


def _parse_kv_metrics(text: str) -> dict[str, str]:
    """Parse a metrics.txt blob (KEY=VALUE per line) into a dict.

    The persisted form `tools/benchmark/run_benchmark.sh` writes
    after piping vllm-bench-serve through parse_bench_log. Lines
    without `=` are skipped (defensive against stray blank lines).
    """
    out: dict[str, str] = {}
    for line in text.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip()
    return out


def _v1_metrics_to_v2(parsed: dict[str, str]) -> dict[str, float]:
    """Map v1 bench-output metric names to v2 schema names + cast to float.

    Drops anything not in the v2 objective set (today's three
    metrics). A future objective addition (e.g. tail-latency
    composite) extends both this map and
    service/project.DEFAULT_OBJECTIVES in the same commit.
    """
    name_map = {
        "req_per_sec":  "RequestThroughput",
        "ttft_mean_ms": "MeanTTFT",
        "ttft_p99_ms":  "P99TTFT",
    }
    out: dict[str, float] = {}
    for v2_name, v1_name in name_map.items():
        raw = parsed.get(v1_name, "")
        try:
            out[v2_name] = float(raw)
        except (ValueError, TypeError):
            # Empty / non-numeric → omit. The projection's
            # `_row_has_objective` check drops rows missing a
            # required metric, so the sweep continues without a
            # spurious win.
            pass
    return out


def make_measurement_fn(
    workload_path: Path,
    *,
    bench_script: Path | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    base_environ: dict[str, str] | None = None,
    run_subprocess: Callable = subprocess.run,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build a v2-shaped measurement_fn that drives `run_benchmark.sh`.

    Args:
      workload_path: Path to the .workload file. Forwarded to
                     run_benchmark.sh as its positional case-file arg
                     (same convention as v1 sweep).
      bench_script: Override path to `run_benchmark.sh`. Defaults to
                    `tools/benchmark/run_benchmark.sh` relative to the
                    repo root inferred from this module's location.
      timeout_seconds: Per-combo wall-clock cap. Past this we record
                       FAILED_OOM (timeout almost always tracks back
                       to OOM in the kernel; the v2 status is
                       permanent so resume doesn't re-attempt).
      base_environ: Base environment for the subprocess. Defaults to
                    `os.environ`. The combo's vars are merged in on
                    top per combo.
      run_subprocess: Injected for testability. Defaults to
                      `subprocess.run`.

    Returns:
      `measurement_fn(combo_dict) -> result_dict`. The result is
      v2-shaped: status + (on success) metrics sub-dict.

    The combo dict's keys are bench env-var names directly — the
    service-sweep search-space module emits MAX_NUM_BATCHED_TOKENS,
    MAX_NUM_SEQS, etc. with their final v1 names, so no translation
    is needed at this layer.
    """
    if bench_script is None:
        # Repo root: this file is at <repo>/tools/tuning/v2/service/...
        # so up four levels.
        repo_root = Path(__file__).resolve().parents[4]
        bench_script = repo_root / "tools" / "benchmark" / "run_benchmark.sh"

    def measurement(combo: dict[str, Any]) -> dict[str, Any]:
        # Smoke-run escape hatch: bypass the real bench subprocess
        # and return deterministic synthetic metrics. Lets the sweep
        # flow run end-to-end (raw_store write, projection, commit
        # hooks) without standing up a vLLM server. Values are
        # non-zero so projection's "is there a metric?" check passes
        # — they are NOT real measurements; do not compare to past
        # winners.
        if os.environ.get("MOCK_BENCH") == "1":
            return {
                "status": "SUCCESS",
                "metrics": {
                    "req_per_sec":  1.0,
                    "ttft_mean_ms": 100.0,
                    "ttft_p99_ms":  200.0,
                },
                "mock":   True,   # downstream can flag-filter mocks
            }

        # Lazy import so this module is import-able without
        # tools.benchmark on the path (tests stub the path; the
        # parser doesn't pull vLLM).
        from tools.benchmark.parse_bench_log import parse_bench_log
        from tools.benchmark._schema import METRICS_FILENAME

        # One temp dir per combo so concurrent sweeps (a future
        # feature) don't collide; cleaned up by Python on exit.
        rdir = Path(tempfile.mkdtemp(prefix="v2_bench_"))

        env = dict(os.environ if base_environ is None else base_environ)
        env.update({str(k): str(v) for k, v in combo.items()})
        env["RESULT_DIR"] = str(rdir)

        cmd = [str(bench_script), str(workload_path)]
        try:
            proc = run_subprocess(
                cmd, env=env, check=False, timeout=timeout_seconds,
                capture_output=True, text=True,
            )
        except subprocess.TimeoutExpired:
            return {
                "status": "FAILED_OOM",
                "error":  f"bench timed out after {timeout_seconds}s "
                          f"(typically OOM downstream of the kernel)",
            }
        except (OSError, subprocess.SubprocessError) as e:
            return {
                "status": "UNKNOWN_ERROR",
                "error":  f"bench subprocess raised: "
                          f"{type(e).__name__}: {e}",
            }

        rc = getattr(proc, "returncode", 1)
        stdout = getattr(proc, "stdout", "") or ""

        if rc != 0:
            return {
                "status":      "FAILED",
                "error":       f"bench exited rc={rc}",
                "return_code": rc,
            }

        # Prefer parsing the persisted metrics.txt (canonical KEY=VALUE
        # lines per parse_bench_log.format_metrics) and fall back to
        # the captured stdout (vllm-bench-serve native format) if
        # the file is absent. Two different parsers because
        # run_benchmark.sh pipes the bench output through
        # parse_bench_log to normalise it.
        metrics_path = rdir / METRICS_FILENAME
        parsed: dict[str, str] = {}
        if metrics_path.exists() and metrics_path.read_text().strip():
            parsed = _parse_kv_metrics(metrics_path.read_text())
        if not parsed:
            parsed = parse_bench_log(stdout)
        v2_metrics = _v1_metrics_to_v2(parsed)

        # Sanity: at least req_per_sec must parse. If it didn't, the
        # bench probably wrote a partial / corrupt file (the v1
        # sweep's `is_completed` check captures the same condition).
        if "req_per_sec" not in v2_metrics:
            return {
                "status": "FAILED",
                "error":  f"bench exited rc=0 but RequestThroughput "
                          f"missing from output ({metrics_path}).",
            }

        return {
            "status":  "SUCCESS",
            "metrics": v2_metrics,
        }

    return measurement


def main(argv: list[str] | None = None) -> int:
    """Smoke entry. Builds the adapter, runs ONE bench, prints result.

    Useful for verifying the vLLM bench env is alive end-to-end
    without going through the full service sweep.
    """
    import argparse
    import json
    p = argparse.ArgumentParser(
        description="Run one vllm-bench measurement and print the result.",
    )
    p.add_argument("workload", type=Path,
                   help="Path to the .workload file.")
    p.add_argument("--mnb", type=int, default=8192,
                   help="MAX_NUM_BATCHED_TOKENS override.")
    p.add_argument("--mns", type=int, default=128,
                   help="MAX_NUM_SEQS override.")
    p.add_argument("--timeout", type=int,
                   default=DEFAULT_TIMEOUT_SECONDS,
                   help="Per-bench timeout in seconds.")
    args = p.parse_args(argv)

    if not args.workload.exists():
        print(f"workload not found: {args.workload}", file=sys.stderr)
        return 1

    measure = make_measurement_fn(
        args.workload, timeout_seconds=args.timeout,
    )
    result = measure({
        "MAX_NUM_BATCHED_TOKENS": args.mnb,
        "MAX_NUM_SEQS":           args.mns,
    })
    print(json.dumps(result, indent=2))
    return 0 if result.get("status") == "SUCCESS" else 1


if __name__ == "__main__":   # pragma: no cover
    sys.exit(main())
