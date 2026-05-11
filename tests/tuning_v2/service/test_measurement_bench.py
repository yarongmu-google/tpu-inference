# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for tools.tuning.v2.service.measurement_bench.

The adapter shells out to tools/benchmark/run_benchmark.sh, parses
the resulting metrics, returns a v2-shaped result dict. We mock
subprocess.run so these tests never actually run a bench.
"""

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.tuning.v2.service.measurement_bench import (
    DEFAULT_TIMEOUT_SECONDS,
    _parse_kv_metrics,
    _v1_metrics_to_v2,
    make_measurement_fn,
)


# Sample bench-output text in v1 format. parse_bench_log extracts
# RequestThroughput + Mean/P99 TTFT from this shape.
SAMPLE_BENCH_STDOUT = """\
Benchmark...
Request throughput (req/s):  4.79
Output token throughput (tok/s):  1234.5
Total Token throughput (tok/s):  5678.9
TTFT (ms):
Mean TTFT (ms):  102.3
Median TTFT (ms):  98.1
P99 TTFT (ms):  205.8
"""


class TestMetricsMapping(unittest.TestCase):

    def test_maps_three_v2_metrics(self):
        parsed = {
            "RequestThroughput": "4.79",
            "MeanTTFT":          "102.3",
            "P99TTFT":           "205.8",
            "OutputTokenThroughput": "1234.5",   # dropped (not v2)
        }
        out = _v1_metrics_to_v2(parsed)
        self.assertEqual(out["req_per_sec"], 4.79)
        self.assertEqual(out["ttft_mean_ms"], 102.3)
        self.assertEqual(out["ttft_p99_ms"], 205.8)
        self.assertEqual(set(out.keys()),
                         {"req_per_sec", "ttft_mean_ms", "ttft_p99_ms"})

    def test_missing_metric_dropped_not_zero(self):
        """Empty string in v1 output -> the key is OMITTED in v2.
        Projection's _row_has_objective then drops the row from
        comparison; better than seeding 0.0 which would silently
        win on min objectives."""
        parsed = {"RequestThroughput": "4.79", "MeanTTFT": ""}
        out = _v1_metrics_to_v2(parsed)
        self.assertIn("req_per_sec", out)
        self.assertNotIn("ttft_mean_ms", out)

    def test_non_numeric_dropped(self):
        parsed = {"RequestThroughput": "n/a"}
        out = _v1_metrics_to_v2(parsed)
        self.assertEqual(out, {})


class TestParseKvMetrics(unittest.TestCase):

    def test_skips_lines_without_equals(self):
        """metrics.txt is mostly KEY=VALUE, but a blank line or banner
        comment shouldn't crash the parse."""
        text = "RequestThroughput=4.79\n\nbanner line\nMeanTTFT=100\n"
        out = _parse_kv_metrics(text)
        self.assertEqual(out, {"RequestThroughput": "4.79",
                               "MeanTTFT": "100"})


class TestMakeMeasurementFn(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.workload = self.dir / "x.workload"
        self.workload.write_text("MAX_NUM_SEQS=128\n")

    def tearDown(self):
        self.tmp.cleanup()

    def _make_proc(self, *, rc: int = 0,
                   stdout: str = SAMPLE_BENCH_STDOUT) -> mock.Mock:
        proc = mock.Mock()
        proc.returncode = rc
        proc.stdout = stdout
        proc.stderr = ""
        return proc

    def test_success_returns_v2_metrics(self):
        run_mock = mock.Mock(return_value=self._make_proc())
        fn = make_measurement_fn(
            self.workload,
            bench_script=Path("/fake/run_benchmark.sh"),
            run_subprocess=run_mock,
        )
        result = fn({"MAX_NUM_BATCHED_TOKENS": 8192,
                     "MAX_NUM_SEQS":           128})
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["metrics"]["req_per_sec"], 4.79)
        self.assertEqual(result["metrics"]["ttft_mean_ms"], 102.3)
        self.assertEqual(result["metrics"]["ttft_p99_ms"], 205.8)

    def test_combo_vars_passed_to_subprocess_env(self):
        captured = {}
        def fake_run(*args, **kwargs):
            captured["env"] = kwargs.get("env", {})
            return self._make_proc()
        fn = make_measurement_fn(
            self.workload,
            bench_script=Path("/fake/run_benchmark.sh"),
            run_subprocess=fake_run,
        )
        fn({"MAX_NUM_BATCHED_TOKENS": 16384,
            "MAX_NUM_SEQS":           1000})
        env = captured["env"]
        self.assertEqual(env["MAX_NUM_BATCHED_TOKENS"], "16384")
        self.assertEqual(env["MAX_NUM_SEQS"], "1000")
        # RESULT_DIR is also set so the bench writes metrics.txt
        # somewhere we can pick it up later.
        self.assertIn("RESULT_DIR", env)

    def test_workload_passed_as_positional_arg(self):
        captured = {}
        def fake_run(cmd, *args, **kwargs):
            captured["cmd"] = cmd
            return self._make_proc()
        fn = make_measurement_fn(
            self.workload,
            bench_script=Path("/fake/run_benchmark.sh"),
            run_subprocess=fake_run,
        )
        fn({"MAX_NUM_BATCHED_TOKENS": 8192})
        # cmd is [<bench_script>, <workload_path>].
        self.assertEqual(captured["cmd"][1], str(self.workload))

    def test_nonzero_returncode_is_failed(self):
        run_mock = mock.Mock(return_value=self._make_proc(rc=1, stdout=""))
        fn = make_measurement_fn(
            self.workload,
            bench_script=Path("/fake/run_benchmark.sh"),
            run_subprocess=run_mock,
        )
        result = fn({"MAX_NUM_BATCHED_TOKENS": 8192})
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["return_code"], 1)

    def test_timeout_is_failed_oom(self):
        """Wall-clock timeout almost always tracks back to OOM in the
        kernel — the v1 sweep saw the same pattern. Mark FAILED_OOM
        so resume doesn't waste cycles re-attempting."""
        def fake_run(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd="bench", timeout=10)
        fn = make_measurement_fn(
            self.workload,
            bench_script=Path("/fake/run_benchmark.sh"),
            run_subprocess=fake_run,
            timeout_seconds=10,
        )
        result = fn({"MAX_NUM_BATCHED_TOKENS": 8192})
        self.assertEqual(result["status"], "FAILED_OOM")
        self.assertIn("10s", result["error"])

    def test_oserror_is_unknown_error(self):
        """Bench script missing / not executable — surface as
        UNKNOWN_ERROR (retryable; operator fixes the script, re-runs)."""
        def fake_run(*args, **kwargs):
            raise OSError(2, "No such file or directory")
        fn = make_measurement_fn(
            self.workload,
            bench_script=Path("/fake/missing.sh"),
            run_subprocess=fake_run,
        )
        result = fn({"MAX_NUM_BATCHED_TOKENS": 8192})
        self.assertEqual(result["status"], "UNKNOWN_ERROR")
        # FileNotFoundError is an OSError subclass; the typed name
        # is what surfaces (more useful to the operator).
        self.assertIn("FileNotFoundError", result["error"])

    def test_subprocess_error_is_unknown_error(self):
        def fake_run(*args, **kwargs):
            raise subprocess.SubprocessError("setup failed")
        fn = make_measurement_fn(
            self.workload,
            bench_script=Path("/fake/run_benchmark.sh"),
            run_subprocess=fake_run,
        )
        result = fn({"MAX_NUM_BATCHED_TOKENS": 8192})
        self.assertEqual(result["status"], "UNKNOWN_ERROR")

    def test_rc_zero_but_missing_throughput_is_failed(self):
        """The bench exited 0 but output had no RequestThroughput —
        partial-write artifact (the v1 sweep also catches this).
        Don't return SUCCESS with empty metrics."""
        run_mock = mock.Mock(return_value=self._make_proc(
            stdout="something happened but no metrics emitted",
        ))
        fn = make_measurement_fn(
            self.workload,
            bench_script=Path("/fake/run_benchmark.sh"),
            run_subprocess=run_mock,
        )
        result = fn({"MAX_NUM_BATCHED_TOKENS": 8192})
        self.assertEqual(result["status"], "FAILED")
        self.assertIn("RequestThroughput", result["error"])

    def test_prefers_metrics_file_over_stdout(self):
        """run_benchmark.sh persists metrics.txt to RESULT_DIR. The
        adapter should prefer that file (canonical) over the captured
        stdout (which may be truncated / interleaved with server log)."""
        captured_rdir = {}
        def fake_run(cmd, env, **kwargs):
            rdir = Path(env["RESULT_DIR"])
            (rdir / "metrics.txt").write_text(
                "RequestThroughput=9.99\nMeanTTFT=11.0\nP99TTFT=22.0\n"
            )
            captured_rdir["rdir"] = rdir
            return self._make_proc(stdout=SAMPLE_BENCH_STDOUT)
        fn = make_measurement_fn(
            self.workload,
            bench_script=Path("/fake/run_benchmark.sh"),
            run_subprocess=fake_run,
        )
        result = fn({"MAX_NUM_BATCHED_TOKENS": 8192})
        # File contents (9.99) won, not stdout (4.79).
        self.assertEqual(result["metrics"]["req_per_sec"], 9.99)

    def test_falls_back_to_stdout_when_metrics_file_absent(self):
        """If the bench didn't write metrics.txt but stdout has the
        numbers, parse those. Forward-compat for bench scripts that
        might emit to stdout only."""
        run_mock = mock.Mock(return_value=self._make_proc())   # default stdout
        fn = make_measurement_fn(
            self.workload,
            bench_script=Path("/fake/run_benchmark.sh"),
            run_subprocess=run_mock,
        )
        result = fn({"MAX_NUM_BATCHED_TOKENS": 8192})
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["metrics"]["req_per_sec"], 4.79)


class TestDefaults(unittest.TestCase):

    def test_default_timeout_is_30_minutes(self):
        """Long enough for a high-MNB bench (warmup + measure); short
        enough that a wedged server doesn't burn the sweep window."""
        self.assertEqual(DEFAULT_TIMEOUT_SECONDS, 1800)

    def test_default_bench_script_points_at_run_benchmark_sh(self):
        """When bench_script=None, the adapter resolves to
        tools/benchmark/run_benchmark.sh in the repo root."""
        captured = {}
        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            proc = mock.Mock()
            proc.returncode = 0
            proc.stdout = SAMPLE_BENCH_STDOUT
            return proc
        with tempfile.TemporaryDirectory() as t:
            wl = Path(t) / "x.workload"
            wl.write_text("MAX_NUM_SEQS=1\n")
            fn = make_measurement_fn(wl, run_subprocess=fake_run)
            fn({"MAX_NUM_BATCHED_TOKENS": 8192})
        self.assertTrue(captured["cmd"][0].endswith(
            "tools/benchmark/run_benchmark.sh"
        ))


class TestSmokeMain(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_main_missing_workload_returns_1(self):
        from tools.tuning.v2.service.measurement_bench import main
        import io, sys
        with mock.patch.object(sys, "stderr", new=io.StringIO()):
            rc = main([str(self.dir / "missing.workload")])
        self.assertEqual(rc, 1)

    def test_main_invokes_adapter(self):
        from tools.tuning.v2.service.measurement_bench import main
        w = self.dir / "x.workload"
        w.write_text("MAX_NUM_SEQS=128\n")
        with mock.patch(
            "tools.tuning.v2.service.measurement_bench.make_measurement_fn",
            return_value=lambda c: {
                "status": "SUCCESS",
                "metrics": {"req_per_sec": 4.0,
                            "ttft_mean_ms": 100, "ttft_p99_ms": 200},
            },
        ):
            import sys, io
            with mock.patch.object(sys, "stdout", new=io.StringIO()):
                rc = main([str(w), "--mnb", "8192", "--mns", "128"])
        self.assertEqual(rc, 0)

    def test_main_nonsuccess_returns_1(self):
        from tools.tuning.v2.service.measurement_bench import main
        w = self.dir / "x.workload"
        w.write_text("MAX_NUM_SEQS=128\n")
        with mock.patch(
            "tools.tuning.v2.service.measurement_bench.make_measurement_fn",
            return_value=lambda c: {"status": "FAILED", "error": "boom"},
        ):
            import sys, io
            with mock.patch.object(sys, "stdout", new=io.StringIO()):
                rc = main([str(w)])
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
