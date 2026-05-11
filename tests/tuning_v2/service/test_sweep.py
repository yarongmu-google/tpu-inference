# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for tools.tuning.v2.service.sweep."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.tuning.v2.core.git_atomic import NO_PUSH_ENV
from tools.tuning.v2.core.raw_store import read_rows
from tools.tuning.v2.service.sweep import (
    PERMANENT_STATUSES,
    enumerate_service_combos,
    main as sweep_main,
    run_service_sweep,
)


def _init_git_repo(d: Path) -> None:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Test", "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "Test", "GIT_COMMITTER_EMAIL": "t@example.com",
    }
    subprocess.run(["git", "init", "-q"], cwd=d, check=True, env=env)
    subprocess.run(["git", "config", "commit.gpgsign", "false"],
                   cwd=d, check=True, env=env)
    (d / "init.txt").write_text("hi")
    subprocess.run(["git", "add", "init.txt"], cwd=d, check=True, env=env)
    subprocess.run(["git", "commit", "-q", "-m", "init"],
                   cwd=d, check=True, env=env)


class TestPermanentStatuses(unittest.TestCase):

    def test_includes_expected_statuses(self):
        self.assertIn("SUCCESS", PERMANENT_STATUSES)
        self.assertIn("FAILED_OOM", PERMANENT_STATUSES)
        self.assertIn("FAILED", PERMANENT_STATUSES)
        self.assertIn("SKIPPED", PERMANENT_STATUSES)

    def test_excludes_unknown_error(self):
        self.assertNotIn("UNKNOWN_ERROR", PERMANENT_STATUSES)


class TestEnumerateServiceCombos(unittest.TestCase):

    def test_empty_search_space_yields_nothing(self):
        self.assertEqual(
            list(enumerate_service_combos(search_space={})), [],
        )

    def test_single_axis(self):
        combos = list(enumerate_service_combos(
            search_space={"MNS": [128, 256]},
        ))
        self.assertEqual(combos, [{"MNS": 128}, {"MNS": 256}])

    def test_cartesian_product(self):
        combos = list(enumerate_service_combos(
            search_space={"a": [1, 2], "b": [3, 4]},
        ))
        # Sorted-axis-key product, so 'a' varies first.
        self.assertEqual(combos, [
            {"a": 1, "b": 3},
            {"a": 1, "b": 4},
            {"a": 2, "b": 3},
            {"a": 2, "b": 4},
        ])

    def test_iteration_order_is_deterministic(self):
        space = {"x": [10, 20, 30], "y": [1, 2]}
        a = list(enumerate_service_combos(search_space=space))
        b = list(enumerate_service_combos(search_space=space))
        self.assertEqual(a, b)


class TestRunServiceSweep(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        _init_git_repo(self.repo)
        self.workload_dir = self.repo / "cases" / "v7x" / "llama3_8b"
        self.workload_dir.mkdir(parents=True)
        self._saved_no_push = os.environ.pop(NO_PUSH_ENV, None)
        os.environ[NO_PUSH_ENV] = "1"
        # Narrow overlay: 2 combos.
        (self.workload_dir / "test.service_axes.json").write_text(
            json.dumps({
                "MAX_NUM_BATCHED_TOKENS": [8192],
                "MAX_NUM_SEQS":           [128, 1000],
            }),
        )
        self.raw_path = (
            self.workload_dir / "test.service.raw" / "sha1-sha2.jsonl"
        )

    def tearDown(self):
        if self._saved_no_push is None:
            os.environ.pop(NO_PUSH_ENV, None)
        else:
            os.environ[NO_PUSH_ENV] = self._saved_no_push
        self.tmp.cleanup()

    def _mock_measure(self, combo):
        return {
            "status": "SUCCESS",
            "metrics": {
                "req_per_sec":   4.79 + 0.01 * combo["MAX_NUM_SEQS"],
                "ttft_mean_ms":  100_000,
                "ttft_p99_ms":   200_000,
            },
        }

    def test_runs_each_combo_returns_count(self):
        calls = []
        def measure(c):
            calls.append(c)
            return self._mock_measure(c)
        n = run_service_sweep(
            workload_env={},
            workload_dir=self.workload_dir,
            workload_name="test",
            raw_path=self.raw_path,
            measurement_fn=measure,
            service_revision="sha1-sha2",
        )
        self.assertEqual(n, 2)
        self.assertEqual(len(calls), 2)

    def test_writes_rows_with_combo_and_metrics(self):
        run_service_sweep(
            workload_env={},
            workload_dir=self.workload_dir,
            workload_name="test",
            raw_path=self.raw_path,
            measurement_fn=self._mock_measure,
            service_revision="sha1-sha2",
        )
        rows = list(read_rows(self.raw_path))
        self.assertEqual(len(rows), 2)
        for r in rows:
            self.assertIn("combo", r)
            self.assertIn("metrics", r)
            self.assertEqual(r["status"], "SUCCESS")

    def test_resume_skips_completed(self):
        run_service_sweep(
            workload_env={},
            workload_dir=self.workload_dir,
            workload_name="test",
            raw_path=self.raw_path,
            measurement_fn=self._mock_measure,
            service_revision="sha1-sha2",
        )
        # Second run: all already done.
        n = run_service_sweep(
            workload_env={},
            workload_dir=self.workload_dir,
            workload_name="test",
            raw_path=self.raw_path,
            measurement_fn=self._mock_measure,
            service_revision="sha1-sha2",
        )
        self.assertEqual(n, 0)

    def test_resume_retries_unknown_error(self):
        def measure_err(c):
            return {"status": "UNKNOWN_ERROR"}
        run_service_sweep(
            workload_env={}, workload_dir=self.workload_dir,
            workload_name="test", raw_path=self.raw_path,
            measurement_fn=measure_err, service_revision="sha1-sha2",
        )
        # Second run: succeeds.
        n = run_service_sweep(
            workload_env={}, workload_dir=self.workload_dir,
            workload_name="test", raw_path=self.raw_path,
            measurement_fn=self._mock_measure,
            service_revision="sha1-sha2",
        )
        self.assertEqual(n, 2)

    def test_measurement_exception_recorded(self):
        def raise_(c):
            raise RuntimeError("boom")
        n = run_service_sweep(
            workload_env={}, workload_dir=self.workload_dir,
            workload_name="test", raw_path=self.raw_path,
            measurement_fn=raise_, service_revision="sha1-sha2",
        )
        self.assertEqual(n, 2)
        rows = list(read_rows(self.raw_path))
        for r in rows:
            self.assertEqual(r["status"], "UNKNOWN_ERROR")
            self.assertIn("RuntimeError", r["error"])

    def test_on_progress_callback(self):
        progress = []
        run_service_sweep(
            workload_env={}, workload_dir=self.workload_dir,
            workload_name="test", raw_path=self.raw_path,
            measurement_fn=self._mock_measure,
            service_revision="sha1-sha2",
            on_progress=progress.append,
        )
        self.assertEqual(progress, [1, 2])

    def test_periodic_commit_fires_at_commit_every(self):
        # Widen overlay to 4 combos.
        (self.workload_dir / "test.service_axes.json").write_text(
            json.dumps({
                "MAX_NUM_BATCHED_TOKENS": [8192, 16384],
                "MAX_NUM_SEQS":           [128, 1000],
            }),
        )
        with mock.patch(
            "tools.tuning.v2.service.sweep.commit_and_push"
        ) as cap:
            n = run_service_sweep(
                workload_env={}, workload_dir=self.workload_dir,
                workload_name="test", raw_path=self.raw_path,
                measurement_fn=self._mock_measure,
                service_revision="sha1-sha2", commit_every=2,
            )
        self.assertEqual(n, 4)
        # 2 periodic (at row 2, 4) + 1 final = 3.
        self.assertEqual(cap.call_count, 3)

    def test_commit_every_zero_disables_periodic(self):
        with mock.patch(
            "tools.tuning.v2.service.sweep.commit_and_push"
        ) as cap:
            run_service_sweep(
                workload_env={}, workload_dir=self.workload_dir,
                workload_name="test", raw_path=self.raw_path,
                measurement_fn=self._mock_measure,
                service_revision="sha1-sha2", commit_every=0,
            )
        # Only final commit.
        self.assertEqual(cap.call_count, 1)

    def test_zero_new_rows_no_final_commit(self):
        run_service_sweep(
            workload_env={}, workload_dir=self.workload_dir,
            workload_name="test", raw_path=self.raw_path,
            measurement_fn=self._mock_measure,
            service_revision="sha1-sha2",
        )
        with mock.patch(
            "tools.tuning.v2.service.sweep.commit_and_push"
        ) as cap:
            n = run_service_sweep(
                workload_env={}, workload_dir=self.workload_dir,
                workload_name="test", raw_path=self.raw_path,
                measurement_fn=self._mock_measure,
                service_revision="sha1-sha2",
            )
            self.assertEqual(n, 0)
            cap.assert_not_called()


class TestCliMain(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_workload_not_found_returns_1(self):
        with mock.patch.object(sys, "stderr", new=open(os.devnull, "w")):
            rc = sweep_main([str(self.dir / "missing.workload")])
        self.assertEqual(rc, 1)

    def test_placeholder_cli_returns_2(self):
        w = self.dir / "x.workload"
        w.write_text("MAX_NUM_SEQS=1\n")
        with mock.patch.object(sys, "stderr", new=open(os.devnull, "w")):
            rc = sweep_main([str(w)])
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
