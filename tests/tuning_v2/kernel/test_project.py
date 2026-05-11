# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for tools.tuning.v2.kernel.project."""

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from tools.tuning.v2.core.git_atomic import NO_PUSH_ENV
from tools.tuning.v2.core.raw_store import append_row
from tools.tuning.v2.kernel.project import (
    main as project_main,
    project_kernel,
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


def _sample_row(latency_us: float, mnss: int, status: str = "SUCCESS") -> dict:
    return {
        "status": status,
        "latency_us": latency_us,
        "tuning_key": {
            "case": "logical",
            "page_size": 128,
            "kernel_K": 256,
            "max_num_seqs": 128,
            "code_revision": "abc12345",
            "num_q_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "max_model_len": 8192,
            "q_dtype": "bfloat16",
            "kv_dtype": "bfloat16",
            "sliding_window": None,
        },
        "tunable_params": {
            "bq_sz": 256, "bkv_sz": 2048,
            "bq_csz": 256, "bkv_csz": 512,
            "mnss": mnss,
        },
    }


class TestProjectKernel(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.raw_dir = self.dir / "prefill_heavy.kernel.raw"
        self.raw_dir.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_returns_none_when_no_raw_dir(self):
        empty = self.dir / "no_workload"
        empty.mkdir()
        out = project_kernel(empty, "missing", code_revision="abc")
        self.assertIsNone(out)

    def test_returns_none_when_raw_dir_empty(self):
        out = project_kernel(self.dir, "prefill_heavy",
                             code_revision="abc12345")
        self.assertIsNone(out)

    def test_picks_matching_sha_file_when_present(self):
        a = self.raw_dir / "abc12345.jsonl"
        b = self.raw_dir / "def98765.jsonl"
        append_row(a, _sample_row(2391.0, mnss=4224))
        append_row(b, _sample_row(9999.0, mnss=33))
        # code_revision "abc12345" should pick file a, ignoring b.
        out = project_kernel(self.dir, "prefill_heavy",
                             code_revision="abc12345")
        self.assertIsNotNone(out)
        doc = json.loads(out.read_text())
        self.assertEqual(doc["raw_source"], "abc12345.jsonl")
        self.assertEqual(doc["winners"][0]["tunable_params"]["mnss"], 4224)

    def test_fallback_to_most_recent_when_sha_missing(self):
        old = self.raw_dir / "old.jsonl"
        new = self.raw_dir / "new.jsonl"
        append_row(old, _sample_row(9999.0, mnss=33))
        # Ensure ordering by mtime.
        time.sleep(0.01)
        append_row(new, _sample_row(2391.0, mnss=4224))
        # Request a SHA that doesn't exist; falls back to most-recent.
        out = project_kernel(self.dir, "prefill_heavy",
                             code_revision="nonexistent")
        doc = json.loads(out.read_text())
        self.assertEqual(doc["raw_source"], "new.jsonl")

    def test_explicit_raw_path_bypasses_discovery(self):
        a = self.raw_dir / "abc.jsonl"
        b = self.raw_dir / "def.jsonl"
        append_row(a, _sample_row(2391.0, mnss=4224))
        append_row(b, _sample_row(99.0, mnss=33))
        # Pass raw_path explicitly; SHA arg ignored.
        out = project_kernel(self.dir, "prefill_heavy",
                             code_revision="abc",
                             raw_path=b)
        doc = json.loads(out.read_text())
        self.assertEqual(doc["raw_source"], "def.jsonl")
        # Winner from file b (only row): mnss=33.
        self.assertEqual(doc["winners"][0]["tunable_params"]["mnss"], 33)

    def test_picks_lowest_latency_per_tuning_key(self):
        path = self.raw_dir / "abc.jsonl"
        # Same tuning_key, different tunable_params.
        append_row(path, _sample_row(2391.0, mnss=4224))
        append_row(path, _sample_row(2203.0, mnss=4224))    # winner
        append_row(path, _sample_row(2500.0, mnss=4224))
        out = project_kernel(self.dir, "prefill_heavy",
                             code_revision="abc")
        doc = json.loads(out.read_text())
        self.assertEqual(len(doc["winners"]), 1)
        self.assertEqual(doc["winners"][0]["latency_us"], 2203.0)

    def test_filters_non_success_status(self):
        path = self.raw_dir / "abc.jsonl"
        append_row(path, _sample_row(100.0, mnss=4224, status="FAILED_OOM"))
        append_row(path, _sample_row(100.0, mnss=4224, status="UNKNOWN_ERROR"))
        append_row(path, _sample_row(2391.0, mnss=4224, status="SUCCESS"))
        out = project_kernel(self.dir, "prefill_heavy",
                             code_revision="abc")
        doc = json.loads(out.read_text())
        self.assertEqual(doc["winners"][0]["latency_us"], 2391.0)

    def test_filters_rows_missing_latency_field(self):
        path = self.raw_dir / "abc.jsonl"
        # Row with status=SUCCESS but no latency_us — pathological.
        bad = _sample_row(0, mnss=4224)
        del bad["latency_us"]
        append_row(path, bad)
        # And a normal row.
        append_row(path, _sample_row(2391.0, mnss=4224))
        out = project_kernel(self.dir, "prefill_heavy",
                             code_revision="abc")
        doc = json.loads(out.read_text())
        self.assertEqual(doc["winners"][0]["latency_us"], 2391.0)

    def test_writes_envelope_shape(self):
        path = self.raw_dir / "abc.jsonl"
        append_row(path, _sample_row(2391.0, mnss=4224))
        out = project_kernel(self.dir, "prefill_heavy",
                             code_revision="abc")
        doc = json.loads(out.read_text())
        self.assertEqual(doc["schema_version"], 1)
        self.assertEqual(doc["workload"], "prefill_heavy")
        self.assertEqual(doc["code_revision"], "abc")
        self.assertEqual(doc["n_winners"], 1)
        self.assertIn("winners", doc)

    def test_idempotent_rewrite(self):
        path = self.raw_dir / "abc.jsonl"
        append_row(path, _sample_row(2391.0, mnss=4224))
        a = project_kernel(self.dir, "prefill_heavy",
                           code_revision="abc")
        first = a.read_text()
        b = project_kernel(self.dir, "prefill_heavy",
                           code_revision="abc")
        second = b.read_text()
        self.assertEqual(first, second)

    def test_default_code_revision_from_sha_module(self):
        """When code_revision is None, falls back to kernel_sha()."""
        path = self.raw_dir / "abc12345.jsonl"
        append_row(path, _sample_row(2391.0, mnss=4224))
        with mock.patch(
            "tools.tuning.v2.kernel.project.kernel_sha",
            return_value="abc12345",
        ):
            out = project_kernel(self.dir, "prefill_heavy")
        doc = json.loads(out.read_text())
        self.assertEqual(doc["code_revision"], "abc12345")

    def test_multiple_tuning_keys_each_winner(self):
        """If raw has rows for multiple tuning_keys (different K's),
        each gets its own winner."""
        path = self.raw_dir / "abc.jsonl"
        # K=256 group, three rows.
        r1 = _sample_row(2391.0, mnss=4224); r1["tuning_key"]["kernel_K"] = 256
        r2 = _sample_row(2203.0, mnss=4224); r2["tuning_key"]["kernel_K"] = 256
        # K=512 group, two rows.
        r3 = _sample_row(3100.0, mnss=4224); r3["tuning_key"]["kernel_K"] = 512
        r4 = _sample_row(3050.0, mnss=4224); r4["tuning_key"]["kernel_K"] = 512
        append_row(path, r1)
        append_row(path, r2)
        append_row(path, r3)
        append_row(path, r4)
        out = project_kernel(self.dir, "prefill_heavy",
                             code_revision="abc")
        doc = json.loads(out.read_text())
        self.assertEqual(doc["n_winners"], 2)
        ks_and_lats = {(w["tuning_key"]["kernel_K"], w["latency_us"])
                       for w in doc["winners"]}
        self.assertEqual(ks_and_lats, {(256, 2203.0), (512, 3050.0)})


class TestHashableHelper(unittest.TestCase):
    """Cover _hashable for dict/list values inside a tuning_key."""

    def test_dict_value_in_tuning_key_hashed(self):
        """Some experimental tuning keys carry nested dicts. The
        projection's group-key computation needs to handle them."""
        from tools.tuning.v2.kernel.project import _kernel_group_key
        row = {
            "tuning_key": {
                "case": "logical",
                "model_meta": {"variant": "x", "config": "y"},
            },
            "tunable_params": {},
        }
        key = _kernel_group_key(row)
        # Returns a hashable tuple; no exception.
        self.assertIsInstance(key, tuple)
        # Hashable.
        self.assertIsNotNone(hash(key))

    def test_list_value_in_tuning_key_hashed(self):
        from tools.tuning.v2.kernel.project import _kernel_group_key
        row = {
            "tuning_key": {
                "case": "logical",
                "page_sizes_attempted": [64, 128, 256],
            },
            "tunable_params": {},
        }
        key = _kernel_group_key(row)
        self.assertIsInstance(key, tuple)
        self.assertIsNotNone(hash(key))


class TestResolveRawPath(unittest.TestCase):
    """Cover the internal helper independently to hit the
    code_revision-is-None branch."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.raw_dir = self.dir / "x.kernel.raw"
        self.raw_dir.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_none_code_revision_falls_back_to_mtime(self):
        from tools.tuning.v2.kernel.project import _resolve_raw_path
        a = self.raw_dir / "a.jsonl"
        b = self.raw_dir / "b.jsonl"
        a.write_text('{"x":1}\n')
        time.sleep(0.01)
        b.write_text('{"x":2}\n')
        result = _resolve_raw_path(self.dir, "x", code_revision=None)
        self.assertEqual(result, b)


class TestCliMain(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        _init_git_repo(self.repo)
        self._saved_no_push = os.environ.pop(NO_PUSH_ENV, None)
        os.environ[NO_PUSH_ENV] = "1"

    def tearDown(self):
        if self._saved_no_push is None:
            os.environ.pop(NO_PUSH_ENV, None)
        else:
            os.environ[NO_PUSH_ENV] = self._saved_no_push
        self.tmp.cleanup()

    def test_main_returns_1_when_workload_missing(self):
        with mock.patch.object(sys, "stderr", new=open(os.devnull, "w")):
            rc = project_main([str(self.repo / "absent.workload")])
        self.assertEqual(rc, 1)

    def test_main_returns_1_when_no_raw(self):
        # workload exists but no raw store.
        w = self.repo / "x.workload"
        w.write_text("MAX_NUM_SEQS=1\n")
        with mock.patch.object(sys, "stderr", new=open(os.devnull, "w")):
            rc = project_main([str(w)])
        self.assertEqual(rc, 1)

    def test_main_writes_output_returns_0(self):
        w = self.repo / "x.workload"
        w.write_text("MAX_NUM_SEQS=1\n")
        raw_dir = self.repo / "x.kernel.raw"
        raw_dir.mkdir()
        append_row(raw_dir / "abc.jsonl",
                   _sample_row(2391.0, mnss=4224))
        with mock.patch(
            "tools.tuning.v2.kernel.project.kernel_sha",
            return_value="abc",
        ):
            with mock.patch.object(sys, "stdout", new=open(os.devnull, "w")):
                rc = project_main([str(w), "--no-commit"])
        self.assertEqual(rc, 0)
        out = self.repo / "x.kernel"
        self.assertTrue(out.exists())

    def test_main_commits_by_default(self):
        w = self.repo / "x.workload"
        w.write_text("MAX_NUM_SEQS=1\n")
        raw_dir = self.repo / "x.kernel.raw"
        raw_dir.mkdir()
        append_row(raw_dir / "abc.jsonl",
                   _sample_row(2391.0, mnss=4224))
        with mock.patch(
            "tools.tuning.v2.kernel.project.kernel_sha",
            return_value="abc",
        ):
            with mock.patch(
                "tools.tuning.v2.kernel.project.commit_and_push"
            ) as cap:
                with mock.patch.object(
                    sys, "stdout", new=open(os.devnull, "w"),
                ):
                    rc = project_main([str(w)])   # NO --no-commit
        self.assertEqual(rc, 0)
        cap.assert_called_once()


if __name__ == "__main__":
    unittest.main()
