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
    KernelRegistryMissingError,
    PERMANENT_STATUSES,
    _is_feasible,
    enumerate_service_combos,
    main as sweep_main,
    resolve_kernel_pin_keys,
    run_service_sweep,
)


# Minimal kernel registry doc the sweep can resolve pin_keys from.
SAMPLE_KERNEL_DOC = {
    "schema_version": 1,
    "workload": "test",
    "code_revision": "abc12345",
    "raw_source": "abc12345.jsonl",
    "n_winners": 1,
    "winners": [{
        "tuning_key": {
            "case": "logical",
            "page_size": 128,
            "kernel_K": 256,
            "code_revision": "abc12345",
        },
        "tunable_params": {
            "bq_sz": 256, "bkv_sz": 2048,
            "bq_csz": 256, "bkv_csz": 512,
            "mnss": 4224,
        },
        "status": "SUCCESS",
        "latency_us": 2391.0,
    }],
}

SAMPLE_PIN_KEYS = {
    "case":          "logical",
    "page_size":     128,
    "kernel_K":      256,
    "code_revision": "abc12345",
    "mnss":          4224,
}


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
        # Kernel registry — required for sweep to resolve pin_keys.
        (self.workload_dir / "test.kernel").write_text(
            json.dumps(SAMPLE_KERNEL_DOC),
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
            self.assertIn("kernel_pin_keys", r)
            self.assertEqual(r["kernel_pin_keys"], SAMPLE_PIN_KEYS)
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

    def test_smoke_test_env_runs_exactly_one_combo(self):
        """fix #11: SMOKE_TEST=1 truncates the sweep to one combo
        end-to-end through run_service_sweep. Removes the narrow
        overlay so the full default space is what smoke compresses."""
        (self.workload_dir / "test.service_axes.json").unlink()
        calls = []
        def measure(c):
            calls.append(c)
            return self._mock_measure(c)
        with mock.patch.dict(os.environ, {"SMOKE_TEST": "1"}):
            n = run_service_sweep(
                workload_env={},
                workload_dir=self.workload_dir,
                workload_name="test",
                raw_path=self.raw_path,
                measurement_fn=measure,
                service_revision="sha1-sha2",
            )
        self.assertEqual(n, 1)
        self.assertEqual(len(calls), 1)

    def test_measurement_returns_non_dict_recorded_as_unknown_error(self):
        """fix #24: a buggy measurement_fn returning None / list / etc.
        must NOT crash the sweep. Coerced to UNKNOWN_ERROR + row stamped."""
        run_service_sweep(
            workload_env={},
            workload_dir=self.workload_dir,
            workload_name="test",
            raw_path=self.raw_path,
            measurement_fn=lambda c: None,
            service_revision="sha1-sha2",
        )
        rows = list(read_rows(self.raw_path))
        for r in rows:
            self.assertEqual(r["status"], "UNKNOWN_ERROR")
            self.assertIn("NoneType", r["error"])

    def test_measurement_returns_list_recorded_as_unknown_error(self):
        """Review followup: parity with kernel-side coverage — list
        return is also coerced to UNKNOWN_ERROR."""
        run_service_sweep(
            workload_env={},
            workload_dir=self.workload_dir,
            workload_name="test",
            raw_path=self.raw_path,
            measurement_fn=lambda c: [1, 2, 3],
            service_revision="sha1-sha2",
        )
        rows = list(read_rows(self.raw_path))
        for r in rows:
            self.assertEqual(r["status"], "UNKNOWN_ERROR")
            self.assertIn("list", r["error"])

    def test_measurement_returns_dict_without_status_coerced_to_unknown(self):
        """Review followup: a dict missing `status` would wedge the
        resume skip-set — coerce to UNKNOWN_ERROR."""
        run_service_sweep(
            workload_env={},
            workload_dir=self.workload_dir,
            workload_name="test",
            raw_path=self.raw_path,
            measurement_fn=lambda c: {"metrics": {"req_per_sec": 1.0}},
            service_revision="sha1-sha2",
        )
        rows = list(read_rows(self.raw_path))
        for r in rows:
            self.assertEqual(r["status"], "UNKNOWN_ERROR")
            # Original fields preserved through coercion.
            self.assertEqual(r["metrics"]["req_per_sec"], 1.0)

    def test_measurement_returns_dict_with_none_status_coerced_to_unknown(self):
        """status=None never matches PERMANENT_STATUSES — same wedge."""
        run_service_sweep(
            workload_env={},
            workload_dir=self.workload_dir,
            workload_name="test",
            raw_path=self.raw_path,
            measurement_fn=lambda c: {"status": None,
                                      "metrics": {"req_per_sec": 1.0}},
            service_revision="sha1-sha2",
        )
        rows = list(read_rows(self.raw_path))
        for r in rows:
            self.assertEqual(r["status"], "UNKNOWN_ERROR")

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
        # 2 periodic (at row 2, 4) — no final commit since
        # n_new % commit_every == 0 (fix #10).
        self.assertEqual(cap.call_count, 2)

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


class TestResolveKernelPinKeys(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_kernel_file_raises(self):
        with self.assertRaises(KernelRegistryMissingError):
            resolve_kernel_pin_keys(self.dir, "missing")

    def test_logical_winner_extracts_pin_keys(self):
        (self.dir / "x.kernel").write_text(json.dumps(SAMPLE_KERNEL_DOC))
        pk = resolve_kernel_pin_keys(self.dir, "x")
        self.assertEqual(pk, SAMPLE_PIN_KEYS)

    def test_prefers_logical_over_prefill(self):
        doc = {
            "winners": [
                {"tuning_key": {"case": "prefill",  "page_size": 64,
                                "kernel_K": 128, "code_revision": "x"}},
                {"tuning_key": {"case": "logical",  "page_size": 128,
                                "kernel_K": 256, "code_revision": "y"}},
            ],
        }
        (self.dir / "x.kernel").write_text(json.dumps(doc))
        pk = resolve_kernel_pin_keys(self.dir, "x")
        # LOGICAL preferred over PREFILL.
        self.assertEqual(pk["case"], "logical")
        self.assertEqual(pk["kernel_K"], 256)
        self.assertEqual(pk["code_revision"], "y")

    def test_falls_back_to_prefill_when_no_logical(self):
        doc = {
            "winners": [
                {"tuning_key": {"case": "decode",
                                "page_size": 128, "code_revision": "x"}},
                {"tuning_key": {"case": "prefill", "page_size": 128,
                                "kernel_K": 256, "code_revision": "x"}},
            ],
        }
        (self.dir / "x.kernel").write_text(json.dumps(doc))
        pk = resolve_kernel_pin_keys(self.dir, "x")
        self.assertEqual(pk["case"], "prefill")

    def test_only_decode_mixed_raises(self):
        """No LOGICAL or PREFILL winner means service can't resolve a
        decoupled-K / chunked-prefill kernel. Loud failure."""
        doc = {
            "winners": [
                {"tuning_key": {"case": "decode",
                                "page_size": 128, "code_revision": "x"}},
                {"tuning_key": {"case": "mixed",
                                "page_size": 128, "code_revision": "x"}},
            ],
        }
        (self.dir / "x.kernel").write_text(json.dumps(doc))
        with self.assertRaises(KernelRegistryMissingError):
            resolve_kernel_pin_keys(self.dir, "x")

    def test_empty_winners_raises(self):
        (self.dir / "x.kernel").write_text(json.dumps({"winners": []}))
        with self.assertRaises(KernelRegistryMissingError):
            resolve_kernel_pin_keys(self.dir, "x")

    def test_logical_winner_includes_mnss_from_tunable_params(self):
        """Review followup: pin_keys now carries mnss so the
        service-sweep pre-filter can drop infeasible MNS combos.
        For LOGICAL (decoupled-K), mnss is in tunable_params."""
        (self.dir / "x.kernel").write_text(json.dumps(SAMPLE_KERNEL_DOC))
        pk = resolve_kernel_pin_keys(self.dir, "x")
        self.assertEqual(pk["mnss"], 4224)

    def test_prefill_winner_includes_mnss_equal_to_max_num_seqs(self):
        """For PREFILL (coupled-K, no decoupling), the iter capacity
        equals max_num_seqs — mnss is sourced from tuning_key."""
        doc = {
            "winners": [{
                "tuning_key": {
                    "case": "prefill",
                    "page_size": 128, "kernel_K": 256,
                    "code_revision": "x",
                    "max_num_seqs": 256,
                },
                "tunable_params": {
                    "bq_sz": 256, "bkv_sz": 2048,
                    "bq_csz": 256, "bkv_csz": 512,
                },
            }],
        }
        (self.dir / "x.kernel").write_text(json.dumps(doc))
        pk = resolve_kernel_pin_keys(self.dir, "x")
        self.assertEqual(pk["case"], "prefill")
        self.assertEqual(pk["mnss"], 256)


class TestRunServiceSweepKernelPinKeys(unittest.TestCase):
    """Specific to the kernel_pin_keys integration."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.workload_dir = Path(self.tmp.name)
        self._saved_no_push = os.environ.pop(NO_PUSH_ENV, None)
        os.environ[NO_PUSH_ENV] = "1"
        (self.workload_dir / "test.service_axes.json").write_text(
            json.dumps({
                "MAX_NUM_BATCHED_TOKENS": [8192],
                "MAX_NUM_SEQS":           [128],
            }),
        )
        self.raw_path = (
            self.workload_dir / "test.service.raw" / "sha.jsonl"
        )

    def tearDown(self):
        if self._saved_no_push is None:
            os.environ.pop(NO_PUSH_ENV, None)
        else:
            os.environ[NO_PUSH_ENV] = self._saved_no_push
        self.tmp.cleanup()

    def test_missing_kernel_file_raises_preflight(self):
        """No .kernel file at startup => sweep refuses to run, doesn't
        produce zero-winner sweeps full of FAILED rows. (fix #9)"""
        with self.assertRaises(KernelRegistryMissingError):
            run_service_sweep(
                workload_env={},
                workload_dir=self.workload_dir,
                workload_name="test",
                raw_path=self.raw_path,
                measurement_fn=lambda c: {
                    "status": "SUCCESS",
                    "metrics": {"req_per_sec": 4.0, "ttft_mean_ms": 100,
                                "ttft_p99_ms": 200},
                },
                service_revision="sha",
            )
        # No rows written.
        self.assertFalse(self.raw_path.exists())

    def test_explicit_pin_keys_bypasses_kernel_read(self):
        """Caller can inject pin_keys to skip the .kernel discovery
        (useful for tests, also for ad-hoc runs)."""
        custom = {
            "case": "logical", "page_size": 64, "kernel_K": 512,
            "code_revision": "deadbeef",
        }
        n = run_service_sweep(
            workload_env={},
            workload_dir=self.workload_dir,
            workload_name="test",
            raw_path=self.raw_path,
            measurement_fn=lambda c: {
                "status": "SUCCESS",
                "metrics": {"req_per_sec": 4.0, "ttft_mean_ms": 100,
                            "ttft_p99_ms": 200},
            },
            service_revision="sha",
            kernel_pin_keys=custom,
        )
        self.assertEqual(n, 1)
        rows = list(read_rows(self.raw_path))
        self.assertEqual(rows[0]["kernel_pin_keys"], custom)


class TestRunServiceSweepCommitGuard(unittest.TestCase):
    """Cover the double-commit guard (fix #10)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.workload_dir = Path(self.tmp.name)
        self._saved_no_push = os.environ.pop(NO_PUSH_ENV, None)
        os.environ[NO_PUSH_ENV] = "1"
        (self.workload_dir / "test.kernel").write_text(
            json.dumps(SAMPLE_KERNEL_DOC),
        )
        self.raw_path = (
            self.workload_dir / "test.service.raw" / "sha.jsonl"
        )

    def tearDown(self):
        if self._saved_no_push is None:
            os.environ.pop(NO_PUSH_ENV, None)
        else:
            os.environ[NO_PUSH_ENV] = self._saved_no_push
        self.tmp.cleanup()

    def test_no_final_commit_when_periodic_exactly_covered(self):
        """When n_new is an exact multiple of commit_every, the
        periodic commit already covered everything — the final
        commit should NOT fire (would be a no-op empty diff but
        clutters git log)."""
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
                workload_env={},
                workload_dir=self.workload_dir,
                workload_name="test",
                raw_path=self.raw_path,
                measurement_fn=lambda c: {
                    "status": "SUCCESS",
                    "metrics": {"req_per_sec": 4.0, "ttft_mean_ms": 100,
                                "ttft_p99_ms": 200},
                },
                service_revision="sha",
                commit_every=2,
            )
        self.assertEqual(n, 4)
        # 2 periodic commits (at row 2 and 4); NO final commit.
        # (Was 3 before the fix.)
        self.assertEqual(cap.call_count, 2)

    def test_final_commit_fires_when_remainder_uncommitted(self):
        """When the periodic stride doesn't divide n_new evenly, the
        tail uncommitted rows need the final commit."""
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
                workload_env={},
                workload_dir=self.workload_dir,
                workload_name="test",
                raw_path=self.raw_path,
                measurement_fn=lambda c: {
                    "status": "SUCCESS",
                    "metrics": {"req_per_sec": 4.0, "ttft_mean_ms": 100,
                                "ttft_p99_ms": 200},
                },
                service_revision="sha",
                commit_every=3,
            )
        self.assertEqual(n, 4)
        # 1 periodic (at row 3) + 1 final (for row 4) = 2.
        self.assertEqual(cap.call_count, 2)


class TestIsFeasible(unittest.TestCase):
    """fix #7: pre-filter for infeasible service combos."""

    def test_combo_above_input_len_is_feasible(self):
        feasible, reason = _is_feasible(
            {"MAX_NUM_BATCHED_TOKENS": 8192},
            {"INPUT_LEN": "1024"},
        )
        self.assertTrue(feasible)
        self.assertIsNone(reason)

    def test_combo_equal_to_input_len_is_feasible(self):
        """Edge: MNB == INPUT_LEN. With chunked prefill off this just
        barely fits."""
        feasible, _ = _is_feasible(
            {"MAX_NUM_BATCHED_TOKENS": 1024},
            {"INPUT_LEN": "1024"},
        )
        self.assertTrue(feasible)

    def test_combo_below_input_len_is_infeasible(self):
        feasible, reason = _is_feasible(
            {"MAX_NUM_BATCHED_TOKENS": 512},
            {"INPUT_LEN": "1024"},
        )
        self.assertFalse(feasible)
        self.assertIn("MAX_NUM_BATCHED_TOKENS=512", reason)
        self.assertIn("INPUT_LEN=1024", reason)

    def test_no_mnb_in_combo_is_feasible(self):
        """If the combo doesn't sweep MNB, we have nothing to filter on."""
        feasible, _ = _is_feasible(
            {"MAX_NUM_SEQS": 128},
            {"INPUT_LEN": "1024"},
        )
        self.assertTrue(feasible)

    def test_no_input_len_in_workload_env_is_feasible(self):
        """Defensive: workload missing INPUT_LEN -> don't pre-filter
        (validate.py is the responsible layer for required fields)."""
        feasible, _ = _is_feasible(
            {"MAX_NUM_BATCHED_TOKENS": 512},
            {},
        )
        self.assertTrue(feasible)

    def test_non_int_value_does_not_crash(self):
        feasible, _ = _is_feasible(
            {"MAX_NUM_BATCHED_TOKENS": "not a number"},
            {"INPUT_LEN": "1024"},
        )
        self.assertTrue(feasible)

    # ----- followup: mnss-based feasibility -----

    def test_mns_above_mnss_is_infeasible(self):
        """Review followup: combo MNS exceeding the pinned kernel's
        iter capacity (mnss) is unrunnable — kernel was tuned with
        fewer slots than the swept concurrency."""
        feasible, reason = _is_feasible(
            {"MAX_NUM_SEQS": 1000},
            {},
            kernel_pin_keys={"mnss": 256},
        )
        self.assertFalse(feasible)
        self.assertIn("MAX_NUM_SEQS=1000", reason)
        self.assertIn("mnss=256", reason)

    def test_mns_equal_to_mnss_is_feasible(self):
        feasible, _ = _is_feasible(
            {"MAX_NUM_SEQS": 256},
            {},
            kernel_pin_keys={"mnss": 256},
        )
        self.assertTrue(feasible)

    def test_mns_below_mnss_is_feasible(self):
        feasible, _ = _is_feasible(
            {"MAX_NUM_SEQS": 128},
            {},
            kernel_pin_keys={"mnss": 4224},
        )
        self.assertTrue(feasible)

    def test_no_pin_keys_skips_mnss_check(self):
        """pin_keys is optional; when absent, the mnss check is skipped
        (the MNB-vs-INPUT_LEN check still runs if applicable)."""
        feasible, _ = _is_feasible(
            {"MAX_NUM_SEQS": 99999},
            {},
            kernel_pin_keys=None,
        )
        self.assertTrue(feasible)

    def test_pin_keys_without_mnss_skips_mnss_check(self):
        """A pin_keys dict that doesn't carry mnss (e.g. legacy raw
        rows from before this followup) doesn't block the sweep."""
        feasible, _ = _is_feasible(
            {"MAX_NUM_SEQS": 99999},
            {},
            kernel_pin_keys={"page_size": 128, "kernel_K": 256},
        )
        self.assertTrue(feasible)

    def test_non_int_mns_or_mnss_does_not_crash(self):
        """Defensive: bad-typed mns/mnss shouldn't crash the filter;
        treat as unfilterable and let the runner record whatever
        comes back."""
        feasible, _ = _is_feasible(
            {"MAX_NUM_SEQS": "not a number"},
            {},
            kernel_pin_keys={"mnss": 256},
        )
        self.assertTrue(feasible)


class TestRunServiceSweepFeasibilityPrefilter(unittest.TestCase):
    """fix #7: infeasible combos get recorded as SKIPPED rows so
    resume picks them up next time instead of re-attempting."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.workload_dir = Path(self.tmp.name)
        self._saved_no_push = os.environ.pop(NO_PUSH_ENV, None)
        os.environ[NO_PUSH_ENV] = "1"
        (self.workload_dir / "test.kernel").write_text(
            json.dumps(SAMPLE_KERNEL_DOC),
        )
        # MNB=[256, 8192] x MNS=[128]. With INPUT_LEN=1024, the
        # MNB=256 row is infeasible.
        (self.workload_dir / "test.service_axes.json").write_text(
            json.dumps({
                "MAX_NUM_BATCHED_TOKENS": [256, 8192],
                "MAX_NUM_SEQS":           [128],
            }),
        )
        self.raw_path = (
            self.workload_dir / "test.service.raw" / "sha.jsonl"
        )

    def tearDown(self):
        if self._saved_no_push is None:
            os.environ.pop(NO_PUSH_ENV, None)
        else:
            os.environ[NO_PUSH_ENV] = self._saved_no_push
        self.tmp.cleanup()

    def _measure(self, c):
        return {"status": "SUCCESS",
                "metrics": {"req_per_sec": 1.0, "ttft_mean_ms": 100,
                            "ttft_p99_ms": 200}}

    def test_infeasible_combo_recorded_as_skipped_not_measured(self):
        calls = []
        def measure(c):
            calls.append(c)
            return self._measure(c)
        run_service_sweep(
            workload_env={"INPUT_LEN": "1024"},
            workload_dir=self.workload_dir,
            workload_name="test",
            raw_path=self.raw_path,
            measurement_fn=measure,
            service_revision="sha",
        )
        rows = list(read_rows(self.raw_path))
        self.assertEqual(len(rows), 2)
        # measurement_fn was called only for the feasible MNB=8192 combo.
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["MAX_NUM_BATCHED_TOKENS"], 8192)
        # The infeasible row landed as SKIPPED with a reason.
        skipped = [r for r in rows if r["status"] == "SKIPPED"]
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0]["combo"]["MAX_NUM_BATCHED_TOKENS"], 256)
        self.assertIn("reason", skipped[0])

    def test_skipped_row_picked_up_by_resume(self):
        """A SKIPPED status is in PERMANENT_STATUSES; the second run
        must NOT re-attempt the infeasible combo."""
        # First run.
        run_service_sweep(
            workload_env={"INPUT_LEN": "1024"},
            workload_dir=self.workload_dir,
            workload_name="test",
            raw_path=self.raw_path,
            measurement_fn=self._measure,
            service_revision="sha",
        )
        # Second run on the same store — should append zero new rows.
        n_second = run_service_sweep(
            workload_env={"INPUT_LEN": "1024"},
            workload_dir=self.workload_dir,
            workload_name="test",
            raw_path=self.raw_path,
            measurement_fn=self._measure,
            service_revision="sha",
        )
        self.assertEqual(n_second, 0)


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
