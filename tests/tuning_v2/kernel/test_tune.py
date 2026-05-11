# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for tools.tuning.v2.kernel.tune.

The runner is the integration point of every core/ module + the
LOGICAL enumerator. Tests cover the loop end-to-end with a mock
measurement function that returns deterministic values — TPU/JAX
never imported.

git_atomic.commit_and_push is exercised but its push step is gated
by KERNEL_TUNER_NO_PUSH which we set in setUp. The function still
makes local commits to a temp git repo; we verify (a) the runner
returns the right N, (b) raw rows are appended, (c) resume skips
already-done combos by reading the existing raw store.
"""

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
from tools.tuning.v2.kernel.tune import (
    PERMANENT_STATUSES,
    main as tune_main,
    model_shape_from_workload,
    run_kernel_tune,
)


def _init_git_repo(d: Path) -> None:
    """Create a one-commit repo so commit_and_push has somewhere to write."""
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


class TestModelShapeFromWorkload(unittest.TestCase):

    def test_extracts_all_fields_with_defaults(self):
        env = {
            "NUM_Q_HEADS": "32", "NUM_KV_HEADS": "8", "HEAD_DIM": "128",
            "MAX_MODEL_LEN": "8192",
        }
        shape = model_shape_from_workload(env)
        self.assertEqual(shape["num_q_heads"], 32)
        self.assertEqual(shape["num_kv_heads"], 8)
        self.assertEqual(shape["head_dim"], 128)
        self.assertEqual(shape["max_model_len"], 8192)
        self.assertEqual(shape["q_dtype"], "bfloat16")
        self.assertEqual(shape["kv_dtype"], "bfloat16")
        self.assertIsNone(shape["sliding_window"])

    def test_explicit_dtype_and_sliding_window(self):
        env = {
            "NUM_Q_HEADS": "32", "NUM_KV_HEADS": "8", "HEAD_DIM": "128",
            "MAX_MODEL_LEN": "8192",
            "Q_DTYPE": "float16", "KV_DTYPE": "float8",
            "SLIDING_WINDOW": "4096",
        }
        shape = model_shape_from_workload(env)
        self.assertEqual(shape["q_dtype"], "float16")
        self.assertEqual(shape["kv_dtype"], "float8")
        self.assertEqual(shape["sliding_window"], 4096)

    def test_empty_sliding_window_is_none(self):
        env = {
            "NUM_Q_HEADS": "32", "NUM_KV_HEADS": "8", "HEAD_DIM": "128",
            "MAX_MODEL_LEN": "8192", "SLIDING_WINDOW": "",
        }
        self.assertIsNone(model_shape_from_workload(env)["sliding_window"])


class TestPermanentStatuses(unittest.TestCase):

    def test_includes_expected_statuses(self):
        self.assertIn("SUCCESS", PERMANENT_STATUSES)
        self.assertIn("FAILED_OOM", PERMANENT_STATUSES)
        self.assertIn("SKIPPED", PERMANENT_STATUSES)

    def test_excludes_unknown_error(self):
        """UNKNOWN_ERROR must be retryable — otherwise post-bugfix
        re-runs wedge (the regression 35b570d7 fixed in v1)."""
        self.assertNotIn("UNKNOWN_ERROR", PERMANENT_STATUSES)


class TestRunKernelTune(unittest.TestCase):

    WORKLOAD_ENV_BASE = {
        "MAX_NUM_SEQS":          "128",
        "MAX_NUM_BATCHED_TOKENS": "8192",
        "NUM_Q_HEADS":            "32",
        "NUM_KV_HEADS":           "8",
        "HEAD_DIM":               "128",
        "MAX_MODEL_LEN":          "8192",
    }

    def setUp(self):
        # Use a git repo as the test root so commit_and_push has
        # somewhere to land.
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        _init_git_repo(self.repo)
        self.workload_dir = self.repo / "cases" / "v7x" / "llama3_8b"
        self.workload_dir.mkdir(parents=True)
        # Always disable push in tests.
        self._saved_no_push = os.environ.pop(NO_PUSH_ENV, None)
        os.environ[NO_PUSH_ENV] = "1"
        # Use a narrow overlay so the cartesian product stays small.
        (self.workload_dir / "test.kernel_axes.json").write_text(json.dumps({
            "page_size": [128],
            "kernel_K":  [256],
            "mnss":      [4224],
            "bq_sz":     [256],
            "bkv_sz":    [2048],
            "bq_csz":    [256],
            "bkv_csz":   [512],
        }))
        self.raw_path = (
            self.workload_dir / "test.kernel.raw" / "abc12345.jsonl"
        )

    def tearDown(self):
        if self._saved_no_push is None:
            os.environ.pop(NO_PUSH_ENV, None)
        else:
            os.environ[NO_PUSH_ENV] = self._saved_no_push
        self.tmp.cleanup()

    def test_runs_each_combo_once_returns_count(self):
        calls = []
        def measure(tk, tp):
            calls.append((tk, tp))
            return {"status": "SUCCESS", "latency_us": 100.0}

        n = run_kernel_tune(
            workload_env=self.WORKLOAD_ENV_BASE,
            workload_dir=self.workload_dir,
            workload_name="test",
            raw_path=self.raw_path,
            measurement_fn=measure,
            code_revision="abc12345",
        )
        # Narrow overlay above yields exactly 1 valid combo.
        self.assertEqual(n, 1)
        self.assertEqual(len(calls), 1)

    def test_writes_raw_rows(self):
        def measure(tk, tp):
            return {"status": "SUCCESS", "latency_us": 100.0}
        run_kernel_tune(
            workload_env=self.WORKLOAD_ENV_BASE,
            workload_dir=self.workload_dir,
            workload_name="test",
            raw_path=self.raw_path,
            measurement_fn=measure,
            code_revision="abc12345",
        )
        rows = list(read_rows(self.raw_path))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "SUCCESS")
        self.assertEqual(rows[0]["latency_us"], 100.0)
        self.assertEqual(rows[0]["tuning_key"]["case"], "logical")
        self.assertEqual(rows[0]["tunable_params"]["mnss"], 4224)

    def test_resume_skips_already_completed_combos(self):
        """Run twice; second invocation finds the row in the raw store
        (status=SUCCESS), skips it, returns 0."""
        def measure(tk, tp):
            return {"status": "SUCCESS", "latency_us": 100.0}
        n1 = run_kernel_tune(
            workload_env=self.WORKLOAD_ENV_BASE,
            workload_dir=self.workload_dir,
            workload_name="test",
            raw_path=self.raw_path,
            measurement_fn=measure,
            code_revision="abc12345",
        )
        n2 = run_kernel_tune(
            workload_env=self.WORKLOAD_ENV_BASE,
            workload_dir=self.workload_dir,
            workload_name="test",
            raw_path=self.raw_path,
            measurement_fn=measure,
            code_revision="abc12345",
        )
        self.assertEqual(n1, 1)
        self.assertEqual(n2, 0)

    def test_resume_retries_unknown_error_combos(self):
        """If a prior row had status=UNKNOWN_ERROR, second run re-tries.
        This is the regression 35b570d7 fixed in v1 — we replicate
        the test here for v2."""
        # First run: measurement returns UNKNOWN_ERROR.
        def measure_err(tk, tp):
            return {"status": "UNKNOWN_ERROR", "latency_us": 0.0,
                    "error": "transient"}
        run_kernel_tune(
            workload_env=self.WORKLOAD_ENV_BASE,
            workload_dir=self.workload_dir,
            workload_name="test",
            raw_path=self.raw_path,
            measurement_fn=measure_err,
            code_revision="abc12345",
        )
        # Second run: measurement now succeeds; the combo should be
        # RE-attempted because UNKNOWN_ERROR isn't permanent.
        def measure_ok(tk, tp):
            return {"status": "SUCCESS", "latency_us": 2391.0}
        n = run_kernel_tune(
            workload_env=self.WORKLOAD_ENV_BASE,
            workload_dir=self.workload_dir,
            workload_name="test",
            raw_path=self.raw_path,
            measurement_fn=measure_ok,
            code_revision="abc12345",
        )
        self.assertEqual(n, 1)
        # Raw store now has 2 rows: first UNKNOWN_ERROR, then SUCCESS.
        rows = list(read_rows(self.raw_path))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["status"], "UNKNOWN_ERROR")
        self.assertEqual(rows[1]["status"], "SUCCESS")

    def test_measurement_exception_recorded_as_unknown_error(self):
        """A raising measurement_fn doesn't kill the loop; row recorded."""
        def measure_raise(tk, tp):
            raise RuntimeError("boom")
        n = run_kernel_tune(
            workload_env=self.WORKLOAD_ENV_BASE,
            workload_dir=self.workload_dir,
            workload_name="test",
            raw_path=self.raw_path,
            measurement_fn=measure_raise,
            code_revision="abc12345",
        )
        self.assertEqual(n, 1)
        rows = list(read_rows(self.raw_path))
        self.assertEqual(rows[0]["status"], "UNKNOWN_ERROR")
        self.assertIn("RuntimeError", rows[0]["error"])
        self.assertIn("boom", rows[0]["error"])

    def test_on_progress_callback_called_per_row(self):
        # Widen the overlay so we have multiple combos.
        (self.workload_dir / "test.kernel_axes.json").write_text(
            json.dumps({
                "page_size": [64, 128],
                "kernel_K":  [256],
                "mnss":      [4224],
                "bq_sz":     [256],
                "bkv_sz":    [2048],
                "bq_csz":    [256],
                "bkv_csz":   [512],
            })
        )
        progress = []
        def measure(tk, tp):
            return {"status": "SUCCESS", "latency_us": 100.0}
        run_kernel_tune(
            workload_env=self.WORKLOAD_ENV_BASE,
            workload_dir=self.workload_dir,
            workload_name="test",
            raw_path=self.raw_path,
            measurement_fn=measure,
            code_revision="abc12345",
            on_progress=progress.append,
        )
        self.assertEqual(progress, [1, 2])

    def test_commit_every_disabled_when_zero(self):
        """commit_every=0 skips periodic commits but still does final.
        Hard to assert without spying on commit_and_push; check via
        the on_progress hook that the loop still runs."""
        (self.workload_dir / "test.kernel_axes.json").write_text(
            json.dumps({
                "page_size": [64, 128],
                "kernel_K":  [256],
                "mnss":      [4224],
                "bq_sz":     [256],
                "bkv_sz":    [2048],
                "bq_csz":    [256],
                "bkv_csz":   [512],
            })
        )
        progress = []
        n = run_kernel_tune(
            workload_env=self.WORKLOAD_ENV_BASE,
            workload_dir=self.workload_dir,
            workload_name="test",
            raw_path=self.raw_path,
            measurement_fn=lambda tk, tp: {
                "status": "SUCCESS", "latency_us": 100.0,
            },
            code_revision="abc12345",
            commit_every=0,
            on_progress=progress.append,
        )
        self.assertEqual(n, 2)
        self.assertEqual(progress, [1, 2])

    def test_zero_new_rows_no_final_commit_called(self):
        """If skip-set covers everything (already-done resume), no
        rows are appended → no commit_and_push call. Verified by
        spying."""
        # Pre-populate raw store with all combos completed.
        def measure_ok(tk, tp):
            return {"status": "SUCCESS", "latency_us": 100.0}
        run_kernel_tune(
            workload_env=self.WORKLOAD_ENV_BASE,
            workload_dir=self.workload_dir,
            workload_name="test",
            raw_path=self.raw_path,
            measurement_fn=measure_ok,
            code_revision="abc12345",
        )
        # Second invocation: spy on commit_and_push.
        with mock.patch(
            "tools.tuning.v2.kernel.tune.commit_and_push"
        ) as cap:
            n = run_kernel_tune(
                workload_env=self.WORKLOAD_ENV_BASE,
                workload_dir=self.workload_dir,
                workload_name="test",
                raw_path=self.raw_path,
                measurement_fn=measure_ok,
                code_revision="abc12345",
            )
            self.assertEqual(n, 0)
            cap.assert_not_called()

    def test_periodic_commit_fires_at_commit_every(self):
        """With commit_every=2 and 4 new rows, periodic commits fire
        at rows 2 and 4. No final commit because n_new % commit_every
        == 0 — would be an empty no-op (fix #10).
        """
        # 4-combo overlay.
        (self.workload_dir / "test.kernel_axes.json").write_text(
            json.dumps({
                "page_size": [64, 128],
                "kernel_K":  [256],
                "mnss":      [4224],
                "bq_sz":     [128, 256],
                "bkv_sz":    [2048],
                "bq_csz":    [128],
                "bkv_csz":   [512],
            })
        )
        with mock.patch(
            "tools.tuning.v2.kernel.tune.commit_and_push"
        ) as cap:
            n = run_kernel_tune(
                workload_env=self.WORKLOAD_ENV_BASE,
                workload_dir=self.workload_dir,
                workload_name="test",
                raw_path=self.raw_path,
                measurement_fn=lambda tk, tp: {
                    "status": "SUCCESS", "latency_us": 100.0,
                },
                code_revision="abc12345",
                commit_every=2,
            )
        self.assertEqual(n, 4)
        # Periodic at rows 2 and 4 — no final.
        self.assertEqual(cap.call_count, 2)

    def test_final_commit_fires_when_remainder_uncommitted(self):
        """commit_every=3 with 4 new rows: periodic at row 3 + final
        for row 4 = 2 commits."""
        (self.workload_dir / "test.kernel_axes.json").write_text(
            json.dumps({
                "page_size": [64, 128],
                "kernel_K":  [256],
                "mnss":      [4224],
                "bq_sz":     [128, 256],
                "bkv_sz":    [2048],
                "bq_csz":    [128],
                "bkv_csz":   [512],
            })
        )
        with mock.patch(
            "tools.tuning.v2.kernel.tune.commit_and_push"
        ) as cap:
            n = run_kernel_tune(
                workload_env=self.WORKLOAD_ENV_BASE,
                workload_dir=self.workload_dir,
                workload_name="test",
                raw_path=self.raw_path,
                measurement_fn=lambda tk, tp: {
                    "status": "SUCCESS", "latency_us": 100.0,
                },
                code_revision="abc12345",
                commit_every=3,
            )
        self.assertEqual(n, 4)
        self.assertEqual(cap.call_count, 2)


class TestParseWorkloadEnv(unittest.TestCase):
    """Cover the workload-env parser's branches."""

    def test_skips_lines_without_equals_sign(self):
        """Lines like '_=/path' from bash env are fine; banner lines
        without an `=` would be skipped (just in case)."""
        from tools.tuning.v2.kernel.tune import _parse_workload_env
        fake_stdout = (
            "FOO=1\n"
            "banner line without equals\n"
            "BAR=2\n"
        )
        with mock.patch("subprocess.run") as run:
            run.return_value = mock.Mock(stdout=fake_stdout, returncode=0)
            env = _parse_workload_env(Path("dummy"))
        self.assertEqual(env, {"FOO": "1", "BAR": "2"})


class TestCliMain(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_workload_not_found_returns_1(self):
        with mock.patch.object(sys, "stderr", new=open(os.devnull, "w")):
            rc = tune_main([str(self.dir / "missing.workload")])
        self.assertEqual(rc, 1)

    def test_placeholder_cli_returns_2(self):
        """The CLI is a stub today (TPU binding lands in a later
        commit). It should print a notice and return 2."""
        w = self.dir / "x.workload"
        w.write_text("MAX_NUM_SEQS=1\nMAX_NUM_BATCHED_TOKENS=8192\n"
                     "NUM_Q_HEADS=1\nNUM_KV_HEADS=1\nHEAD_DIM=1\n"
                     "MAX_MODEL_LEN=1\n")
        # Capture stderr so test output stays clean.
        with mock.patch.object(sys, "stderr", new=open(os.devnull, "w")):
            rc = tune_main([str(w)])
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
