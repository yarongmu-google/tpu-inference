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
from tools.tuning.v2.kernel.enumerate_logical import (
    enumerate_logical_combos,
)
from tools.tuning.v2.kernel.tune import (
    PERMANENT_STATUSES,
    main as tune_main,
    model_shape_from_workload,
)
from tools.tuning.v2.kernel.tune import run_kernel_tune as _run_kernel_tune


def run_kernel_tune(*args, **kwargs):
    """Test wrapper: defaults `enumerator` to the LOGICAL-only
    iterator. These existing tests pre-date the four-case
    `enumerate_all_combos` default in `kernel/tune.py` and assume
    a single-case enumeration. Tests that exercise multi-case
    behavior pass `enumerator=` explicitly."""
    kwargs.setdefault("enumerator", enumerate_logical_combos)
    return _run_kernel_tune(*args, **kwargs)


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

    def test_two_workers_partition_combos_into_disjoint_sets(self):
        """N workers, each only measures its hash-bucket of combos.
        Combined measurements cover every combo exactly once."""
        # Widen the overlay so we have ~8 combos to partition.
        (self.workload_dir / "test.kernel_axes.json").write_text(
            json.dumps({
                "page_size": [64, 128],
                "kernel_K":  [128, 256],
                "mnss":      [4224],
                "bq_sz": [64], "bkv_sz": [512],
                "bq_csz": [64], "bkv_csz": [128],
            }),
        )
        worker_calls: dict[int, list] = {0: [], 1: []}

        def measure_factory(wid):
            def fn(tk, tp):
                worker_calls[wid].append(
                    (tk["page_size"], tk["kernel_K"], tp["mnss"]),
                )
                return {"status": "SUCCESS", "latency_us": 100.0}
            return fn

        # Worker 0
        run_kernel_tune(
            workload_env=self.WORKLOAD_ENV_BASE,
            workload_dir=self.workload_dir,
            workload_name="test",
            raw_path=self.raw_path,
            measurement_fn=measure_factory(0),
            code_revision="abc12345",
            worker_id=0, worker_count=2,
        )
        # Worker 1 — shares the same .raw file.
        run_kernel_tune(
            workload_env=self.WORKLOAD_ENV_BASE,
            workload_dir=self.workload_dir,
            workload_name="test",
            raw_path=self.raw_path,
            measurement_fn=measure_factory(1),
            code_revision="abc12345",
            worker_id=1, worker_count=2,
        )
        # Each combo measured by exactly one worker.
        w0 = set(worker_calls[0])
        w1 = set(worker_calls[1])
        self.assertEqual(w0 & w1, set(),
                         f"workers double-measured: {w0 & w1}")
        # Union covers every distinct combo that the runner produced.
        final_rows = list(read_rows(self.raw_path))
        all_combos = {(r["tuning_key"]["page_size"],
                       r["tuning_key"]["kernel_K"],
                       r["tunable_params"]["mnss"])
                      for r in final_rows}
        self.assertEqual(w0 | w1, all_combos)

    def test_worker_count_zero_or_id_oob_raises(self):
        def measure(tk, tp):
            return {"status": "SUCCESS", "latency_us": 100.0}
        with self.assertRaises(ValueError):
            run_kernel_tune(
                workload_env=self.WORKLOAD_ENV_BASE,
                workload_dir=self.workload_dir,
                workload_name="test",
                raw_path=self.raw_path,
                measurement_fn=measure,
                code_revision="abc12345",
                worker_id=0, worker_count=0,
            )
        with self.assertRaises(ValueError):
            run_kernel_tune(
                workload_env=self.WORKLOAD_ENV_BASE,
                workload_dir=self.workload_dir,
                workload_name="test",
                raw_path=self.raw_path,
                measurement_fn=measure,
                code_revision="abc12345",
                worker_id=2, worker_count=2,
            )

    def test_crash_and_restart_resumes_correctly(self):
        """Real crash-and-restart simulation (broader than the
        completed-combos test): N combos enumerated, runner crashes
        midway via measurement_fn raising SystemExit, then a fresh
        process re-runs and only the unmeasured combos get touched.
        Mirrors v1's `case_set_id`-resume but on the v2 JSONL store."""
        # Multi-combo overlay so we have something to crash partway.
        (self.workload_dir / "test.kernel_axes.json").write_text(
            json.dumps({
                "page_size": [128, 64],
                "kernel_K": [256],
                "mnss":      [4224, 4225],
                "bq_sz": [256], "bkv_sz": [2048],
                "bq_csz": [256], "bkv_csz": [512],
            }),
        )
        all_combos: list = []
        # First pass: succeed 2 then crash on the 3rd.
        crash_at = 3
        def measure_crash(tk, tp):
            all_combos.append((tk["page_size"], tp["mnss"]))
            if len(all_combos) == crash_at:
                raise KeyboardInterrupt("simulated Ctrl-C")
            return {"status": "SUCCESS", "latency_us": 100.0}

        with self.assertRaises(KeyboardInterrupt):
            run_kernel_tune(
                workload_env=self.WORKLOAD_ENV_BASE,
                workload_dir=self.workload_dir,
                workload_name="test",
                raw_path=self.raw_path,
                measurement_fn=measure_crash,
                code_revision="abc12345",
            )

        # On-disk: 2 SUCCESS rows landed (the crash happened before
        # the 3rd was appended).
        rows_after_crash = list(read_rows(self.raw_path))
        self.assertEqual(len(rows_after_crash), 2)

        # Second pass: clean measurement_fn. Counts only un-measured
        # combos — should hit the remaining (3 axes minus 2 already
        # done in pass 1).
        resumed_combos: list = []
        def measure_ok(tk, tp):
            resumed_combos.append((tk["page_size"], tp["mnss"]))
            return {"status": "SUCCESS", "latency_us": 100.0}
        n2 = run_kernel_tune(
            workload_env=self.WORKLOAD_ENV_BASE,
            workload_dir=self.workload_dir,
            workload_name="test",
            raw_path=self.raw_path,
            measurement_fn=measure_ok,
            code_revision="abc12345",
        )
        # n2 = (total - already-done) — exactly the missing combos.
        # No double-measurement of the 2 pre-crash combos.
        already = {c for c in all_combos[:2]}
        re_measured = {c for c in resumed_combos}
        self.assertFalse(already & re_measured,
                         f"resume re-measured: {already & re_measured}")
        # Final state: every distinct combo measured exactly once.
        final_rows = list(read_rows(self.raw_path))
        keys = [(r["tuning_key"]["page_size"],
                 r["tunable_params"]["mnss"])
                for r in final_rows]
        self.assertEqual(len(keys), len(set(keys)),
                         "duplicate combo measurement after resume")

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

    def test_measurement_returns_none_recorded_as_unknown_error(self):
        """fix #24: a buggy measurement_fn returning None must NOT
        crash the loop. Coerced to UNKNOWN_ERROR + row stamped."""
        n = run_kernel_tune(
            workload_env=self.WORKLOAD_ENV_BASE,
            workload_dir=self.workload_dir,
            workload_name="test",
            raw_path=self.raw_path,
            measurement_fn=lambda tk, tp: None,
            code_revision="abc12345",
        )
        self.assertEqual(n, 1)
        rows = list(read_rows(self.raw_path))
        self.assertEqual(rows[0]["status"], "UNKNOWN_ERROR")
        self.assertIn("non-dict", rows[0]["error"])
        self.assertIn("NoneType", rows[0]["error"])

    def test_throughput_workload_resolves_mns_from_service_axes(self):
        """Per arch-doc §2 line 73: when MAX_NUM_SEQS is absent from
        .workload (throughput scenario), the kernel-tune sources it
        from max(service_axes.MAX_NUM_SEQS) so the kernel is tuned
        for the worst-case concurrency the service-sweep may pick."""
        # Workload env without MNS.
        env_no_mns = dict(self.WORKLOAD_ENV_BASE)
        del env_no_mns["MAX_NUM_SEQS"]
        # Remove the existing kernel_axes overlay so the default
        # mnss derivation fires; pin only page_size/kernel_K to keep
        # the cartesian product small.
        (self.workload_dir / "test.kernel_axes.json").write_text(
            json.dumps({
                "page_size": [128], "kernel_K": [256],
                "bq_sz": [256], "bkv_sz": [2048],
                "bq_csz": [256], "bkv_csz": [512],
            }),
        )
        # Service axes pinned high so the kernel-tune ceiling is
        # max(service.MAX_NUM_SEQS).
        (self.workload_dir / "test.service_axes.json").write_text(
            json.dumps({
                "MAX_NUM_BATCHED_TOKENS": [8192],
                "MAX_NUM_SEQS":           [128, 1000],
            }),
        )
        calls = []
        def measure(tk, tp):
            calls.append((tk, tp))
            return {"status": "SUCCESS", "latency_us": 100.0}
        n = run_kernel_tune(
            workload_env=env_no_mns,
            workload_dir=self.workload_dir,
            workload_name="test",
            raw_path=self.raw_path,
            measurement_fn=measure,
            code_revision="abc12345",
        )
        # The kernel-tune ran SOME combos; each tuning_key stamps
        # max_num_seqs = max(service axes) = 1000.
        self.assertGreater(n, 0)
        self.assertEqual(calls[0][0]["max_num_seqs"], 1000)

    def test_smoke_test_env_stops_at_first_success(self):
        """SMOKE_TEST=1: enumerate the full space, run combos until
        one returns SUCCESS, then break. Robust against unfortunate
        first-axis combos that would otherwise dead-end the
        pipeline at the service step (zero winners)."""
        (self.workload_dir / "test.kernel_axes.json").unlink()
        results_to_return = [
            {"status": "SKIPPED"},
            {"status": "SKIPPED"},
            {"status": "SUCCESS", "latency_us": 100.0},
            # Below should never be reached.
            {"status": "SUCCESS", "latency_us": 200.0},
        ]
        calls = []
        def measure(tk, tp):
            calls.append((tk, tp))
            return results_to_return[len(calls) - 1]
        with mock.patch.dict(os.environ, {"SMOKE_TEST": "1"}):
            n = run_kernel_tune(
                workload_env=self.WORKLOAD_ENV_BASE,
                workload_dir=self.workload_dir,
                workload_name="test",
                raw_path=self.raw_path,
                measurement_fn=measure,
                code_revision="abc12345",
            )
        # Three rows written: two SKIPPED + one SUCCESS, then break.
        self.assertEqual(n, 3)
        self.assertEqual(len(calls), 3)

    def test_smoke_test_env_stops_immediately_if_first_succeeds(self):
        """If the first combo is feasible, smoke writes one row."""
        (self.workload_dir / "test.kernel_axes.json").unlink()
        calls = []
        def measure(tk, tp):
            calls.append((tk, tp))
            return {"status": "SUCCESS", "latency_us": 100.0}
        with mock.patch.dict(os.environ, {"SMOKE_TEST": "1"}):
            n = run_kernel_tune(
                workload_env=self.WORKLOAD_ENV_BASE,
                workload_dir=self.workload_dir,
                workload_name="test",
                raw_path=self.raw_path,
                measurement_fn=measure,
                code_revision="abc12345",
            )
        self.assertEqual(n, 1)
        self.assertEqual(len(calls), 1)

    def test_measurement_returns_list_recorded_as_unknown_error(self):
        """fix #24: same defense for any non-dict return."""
        n = run_kernel_tune(
            workload_env=self.WORKLOAD_ENV_BASE,
            workload_dir=self.workload_dir,
            workload_name="test",
            raw_path=self.raw_path,
            measurement_fn=lambda tk, tp: [1, 2, 3],
            code_revision="abc12345",
        )
        self.assertEqual(n, 1)
        rows = list(read_rows(self.raw_path))
        self.assertEqual(rows[0]["status"], "UNKNOWN_ERROR")
        self.assertIn("list", rows[0]["error"])

    def test_measurement_returns_dict_without_status_coerced_to_unknown(self):
        """Review followup: a dict without `status` key would otherwise
        land on disk with no status — wedging resume because the
        skip-set never matches it against PERMANENT_STATUSES."""
        n = run_kernel_tune(
            workload_env=self.WORKLOAD_ENV_BASE,
            workload_dir=self.workload_dir,
            workload_name="test",
            raw_path=self.raw_path,
            measurement_fn=lambda tk, tp: {"latency_us": 100.0},
            code_revision="abc12345",
        )
        self.assertEqual(n, 1)
        rows = list(read_rows(self.raw_path))
        self.assertEqual(rows[0]["status"], "UNKNOWN_ERROR")
        # Other fields are preserved through the coercion.
        self.assertEqual(rows[0]["latency_us"], 100.0)

    def test_measurement_returns_dict_with_none_status_coerced_to_unknown(self):
        """Same wedge concern: status=None never matches the filter."""
        n = run_kernel_tune(
            workload_env=self.WORKLOAD_ENV_BASE,
            workload_dir=self.workload_dir,
            workload_name="test",
            raw_path=self.raw_path,
            measurement_fn=lambda tk, tp: {
                "status": None, "latency_us": 100.0,
            },
            code_revision="abc12345",
        )
        self.assertEqual(n, 1)
        rows = list(read_rows(self.raw_path))
        self.assertEqual(rows[0]["status"], "UNKNOWN_ERROR")

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


class TestDynamicPrune(unittest.TestCase):
    """prev-4: after N consecutive non-SUCCESS in the same (case,
    page_size, kernel_K, mnss) subspace, remaining combos in that
    subspace are SKIPPED_DYNAMIC_PRUNE without measurement.

    The May-11 incident: 45 consecutive UNKNOWN_ERROR at one subspace
    burned 3.5 hours. With N=5 the abort fires after ~15 minutes.
    """

    WORKLOAD_ENV_BASE = {
        "MAX_NUM_SEQS":           "128",
        "MAX_NUM_BATCHED_TOKENS": "8192",
        "NUM_Q_HEADS":            "32",
        "NUM_KV_HEADS":           "8",
        "HEAD_DIM":               "128",
        "MAX_MODEL_LEN":          "8192",
    }

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        _init_git_repo(self.repo)
        self.workload_dir = self.repo / "cases" / "v7x" / "llama3_8b"
        self.workload_dir.mkdir(parents=True)
        self._saved_no_push = os.environ.pop(NO_PUSH_ENV, None)
        os.environ[NO_PUSH_ENV] = "1"
        # Wide-ish overlay at ONE (page, K, mnss) outer combo with
        # many inner block-size combos. All combos share the same
        # subspace tuple — the failure mode dynamic-prune is designed
        # for.
        (self.workload_dir / "test.kernel_axes.json").write_text(json.dumps({
            "page_size": [128],
            "kernel_K":  [256],
            "mnss":      [4224],
            "bq_sz":     [128, 256],
            "bkv_sz":    [1024, 2048],
            "bq_csz":    [64, 128, 256],
            "bkv_csz":   [256, 512],
        }))
        self.raw_path = (
            self.workload_dir / "test.kernel.raw" / "abc12345.jsonl"
        )

    def tearDown(self):
        if self._saved_no_push is None:
            os.environ.pop(NO_PUSH_ENV, None)
        else:
            os.environ[NO_PUSH_ENV] = self._saved_no_push
        os.environ.pop("KERNEL_TUNE_DYN_PRUNE_N", None)
        self.tmp.cleanup()

    def test_aborts_after_threshold_consecutive_fails(self):
        """All combos OOM. Threshold N=3. Only first 3 should call
        measurement_fn; the remaining combos in the same subspace
        write SKIPPED_DYNAMIC_PRUNE rows without measuring."""
        os.environ["KERNEL_TUNE_DYN_PRUNE_N"] = "3"
        calls = []
        def measure(tk, tp):
            calls.append((tk, tp))
            return {"status": "FAILED_OOM"}
        run_kernel_tune(
            workload_env=self.WORKLOAD_ENV_BASE,
            workload_dir=self.workload_dir,
            workload_name="test",
            raw_path=self.raw_path,
            measurement_fn=measure,
            code_revision="abc12345",
        )
        rows = list(read_rows(self.raw_path))
        # measurement_fn called exactly N times before abort.
        self.assertEqual(len(calls), 3)
        # All remaining rows are SKIPPED_DYNAMIC_PRUNE.
        self.assertEqual(
            sum(1 for r in rows if r["status"] == "FAILED_OOM"), 3,
        )
        n_dyn_pruned = sum(
            1 for r in rows
            if r["status"] == "SKIPPED_DYNAMIC_PRUNE"
        )
        self.assertGreater(n_dyn_pruned, 0)
        self.assertEqual(len(rows), len(calls) + n_dyn_pruned)

    def test_success_resets_counter(self):
        """Intermittent non-SUCCESS interleaved with SUCCESS shouldn't
        trip the abort — the counter resets on each SUCCESS."""
        os.environ["KERNEL_TUNE_DYN_PRUNE_N"] = "3"
        statuses = iter([
            "FAILED_OOM", "FAILED_OOM", "SUCCESS",  # reset
            "FAILED_OOM", "FAILED_OOM", "SUCCESS",  # reset
            "FAILED_OOM", "SUCCESS",                # reset
        ] + ["SUCCESS"] * 100)
        def measure(tk, tp):
            s = next(statuses)
            r = {"status": s}
            if s == "SUCCESS":
                r["latency_us"] = 100.0
            return r
        run_kernel_tune(
            workload_env=self.WORKLOAD_ENV_BASE,
            workload_dir=self.workload_dir,
            workload_name="test",
            raw_path=self.raw_path,
            measurement_fn=measure,
            code_revision="abc12345",
        )
        rows = list(read_rows(self.raw_path))
        # No dynamic-prune rows — counter never crossed the threshold.
        self.assertEqual(
            sum(1 for r in rows if r["status"] == "SKIPPED_DYNAMIC_PRUNE"),
            0,
        )

    def test_disabled_when_threshold_is_zero(self):
        """Env override to 0 disables dynamic prune entirely. Useful
        when an operator wants the legacy behavior (measure every
        combo even if all fail) for diagnostic purposes."""
        os.environ["KERNEL_TUNE_DYN_PRUNE_N"] = "0"
        calls = []
        def measure(tk, tp):
            calls.append(1)
            return {"status": "FAILED_OOM"}
        run_kernel_tune(
            workload_env=self.WORKLOAD_ENV_BASE,
            workload_dir=self.workload_dir,
            workload_name="test",
            raw_path=self.raw_path,
            measurement_fn=measure,
            code_revision="abc12345",
        )
        rows = list(read_rows(self.raw_path))
        # Every row is FAILED_OOM; none dynamic-pruned.
        self.assertGreater(len(calls), 3)
        self.assertEqual(
            sum(1 for r in rows if r["status"] == "FAILED_OOM"),
            len(calls),
        )

    def test_pruned_rows_in_permanent_statuses(self):
        """SKIPPED_DYNAMIC_PRUNE must be in PERMANENT_STATUSES so
        resume doesn't re-attempt the doomed combos."""
        from tools.tuning.v2.kernel.tune import PERMANENT_STATUSES
        self.assertIn("SKIPPED_DYNAMIC_PRUNE", PERMANENT_STATUSES)


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

    def test_cli_invokes_adapter_and_writes_rows(self):
        """CLI is now functional (kernel/measurement_tpu.py wired).
        Mock the adapter to keep the test off-TPU; assert the CLI
        sets up env vars, builds the measurement_fn, and writes
        rows to the SHA-named raw path."""
        from tools.tuning.v2.core.git_atomic import NO_PUSH_ENV
        repo = self.dir / "repo"
        cases = repo / "cases" / "v7x" / "llama3_8b"
        cases.mkdir(parents=True)
        _init_git_repo(repo)
        # Narrow overlay so each of the four cases (D/M/P/L)
        # produces exactly one combo. DECODE pins bq_sz=bq_csz=1
        # internally so it ignores the overlay's bq_sz / bq_csz.
        (cases / "x.kernel_axes.json").write_text(json.dumps({
            "page_size": [128], "kernel_K": [256], "mnss": [4224],
            "bq_sz": [256], "bkv_sz": [2048],
            "bq_csz": [256], "bkv_csz": [512],
        }))
        w = cases / "x.workload"
        w.write_text(
            'MAX_NUM_SEQS=128\n'
            'MAX_NUM_BATCHED_TOKENS=8192\n'
            'NUM_Q_HEADS=32\nNUM_KV_HEADS=8\nHEAD_DIM=128\n'
            'MAX_MODEL_LEN=8192\n'
        )
        saved_no_push = os.environ.pop(NO_PUSH_ENV, None)
        os.environ[NO_PUSH_ENV] = "1"
        try:
            measure_calls = []
            def fake_measure(tk, tp):
                measure_calls.append((tk, tp))
                return {"status": "SUCCESS", "latency_us": 100.0}
            with mock.patch(
                "tools.tuning.v2.kernel.measurement_tpu.make_measurement_fn",
                return_value=fake_measure,
            ):
                with mock.patch.object(
                    sys, "stderr", new=open(os.devnull, "w"),
                ):
                    with mock.patch.object(
                        sys, "stdout", new=open(os.devnull, "w"),
                    ):
                        rc = tune_main([str(w)])
        finally:
            if saved_no_push is None:
                os.environ.pop(NO_PUSH_ENV, None)
            else:
                os.environ[NO_PUSH_ENV] = saved_no_push
        self.assertEqual(rc, 0)
        # Default CLI enumerator is enumerate_all_combos → one
        # combo per case (D/M/P/L) under this narrow overlay.
        self.assertEqual(len(measure_calls), 4)
        cases_seen = sorted(tk["case"] for tk, _ in measure_calls)
        self.assertEqual(
            cases_seen, ["decode", "logical", "mixed", "prefill"],
        )
        # Workload env was pushed into os.environ before the
        # adapter built (v1 tuner reads them at construction).
        self.assertEqual(os.environ.get("MAX_NUM_SEQS"), "128")

    def test_cli_throughput_workload_pushes_max_mns_to_environ(self):
        """When .workload omits MAX_NUM_SEQS (throughput scenario),
        the CLI pushes max(service_axes.MAX_NUM_SEQS) into os.environ
        so the v1 RpaV3KernelTuner reads the right ceiling at
        construction time."""
        from tools.tuning.v2.core.git_atomic import NO_PUSH_ENV
        repo = self.dir / "repo"
        cases = repo / "cases" / "v7x" / "llama3_8b" / "throughput"
        cases.mkdir(parents=True)
        _init_git_repo(repo)
        (cases / "x.kernel_axes.json").write_text(json.dumps({
            "page_size": [128], "kernel_K": [256], "mnss": [4224],
            "bq_sz": [256], "bkv_sz": [2048],
            "bq_csz": [256], "bkv_csz": [512],
        }))
        # Pin service MNS axis so we know what the CLI should pick.
        (cases / "x.service_axes.json").write_text(json.dumps({
            "MAX_NUM_BATCHED_TOKENS": [8192],
            "MAX_NUM_SEQS":           [128, 1000],
        }))
        w = cases / "x.workload"
        # Throughput scenario: NO MAX_NUM_SEQS pin.
        w.write_text(
            'NUM_Q_HEADS=32\nNUM_KV_HEADS=8\nHEAD_DIM=128\n'
            'MAX_MODEL_LEN=8192\n'
        )
        saved_no_push = os.environ.pop(NO_PUSH_ENV, None)
        saved_mns = os.environ.pop("MAX_NUM_SEQS", None)
        os.environ[NO_PUSH_ENV] = "1"
        try:
            with mock.patch(
                "tools.tuning.v2.kernel.measurement_tpu.make_measurement_fn",
                return_value=lambda tk, tp: {
                    "status": "SUCCESS", "latency_us": 1.0,
                },
            ):
                with mock.patch.object(
                    sys, "stderr", new=open(os.devnull, "w"),
                ):
                    with mock.patch.object(
                        sys, "stdout", new=open(os.devnull, "w"),
                    ):
                        tune_main([str(w)])
            # CLI pushed the resolved MNS ceiling into env.
            self.assertEqual(os.environ.get("MAX_NUM_SEQS"), "1000")
        finally:
            if saved_no_push is None:
                os.environ.pop(NO_PUSH_ENV, None)
            else:
                os.environ[NO_PUSH_ENV] = saved_no_push
            if saved_mns is None:
                os.environ.pop("MAX_NUM_SEQS", None)
            else:
                os.environ["MAX_NUM_SEQS"] = saved_mns

    def test_cli_iters_and_warmup_threaded_through(self):
        """--iters / --warmup flags reach make_measurement_fn."""
        from tools.tuning.v2.core.git_atomic import NO_PUSH_ENV
        repo = self.dir / "repo"
        cases = repo / "cases" / "v7x" / "llama3_8b"
        cases.mkdir(parents=True)
        _init_git_repo(repo)
        (cases / "x.kernel_axes.json").write_text(json.dumps({
            "page_size": [128], "kernel_K": [256], "mnss": [4224],
            "bq_sz": [256], "bkv_sz": [2048],
            "bq_csz": [256], "bkv_csz": [512],
        }))
        w = cases / "x.workload"
        w.write_text(
            'MAX_NUM_SEQS=128\nMAX_NUM_BATCHED_TOKENS=8192\n'
            'NUM_Q_HEADS=32\nNUM_KV_HEADS=8\nHEAD_DIM=128\n'
            'MAX_MODEL_LEN=8192\n'
        )
        saved_no_push = os.environ.pop(NO_PUSH_ENV, None)
        os.environ[NO_PUSH_ENV] = "1"
        try:
            with mock.patch(
                "tools.tuning.v2.kernel.measurement_tpu.make_measurement_fn",
                return_value=lambda tk, tp: {
                    "status": "SUCCESS", "latency_us": 1.0,
                },
            ) as make_mock:
                with mock.patch.object(
                    sys, "stderr", new=open(os.devnull, "w"),
                ):
                    with mock.patch.object(
                        sys, "stdout", new=open(os.devnull, "w"),
                    ):
                        tune_main([
                            str(w), "--iters", "5", "--warmup", "1",
                        ])
        finally:
            if saved_no_push is None:
                os.environ.pop(NO_PUSH_ENV, None)
            else:
                os.environ[NO_PUSH_ENV] = saved_no_push
        make_mock.assert_called_once_with(iters=5, warmup_iters=1)


if __name__ == "__main__":
    unittest.main()
