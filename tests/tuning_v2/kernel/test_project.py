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


def _sample_row(latency_us: float, mnss: int,
                status: str = "SUCCESS",
                code_revision: str = "abc12345") -> dict:
    return {
        "status": status,
        "latency_us": latency_us,
        "tuning_key": {
            "case": "logical",
            "page_size": 128,
            "kernel_K": 256,
            "max_num_seqs": 128,
            "code_revision": code_revision,
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
        out = project_kernel(empty, "missing", code_revision="abc12345")
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

    def test_explicit_sha_missing_raises(self):
        """fix #2: passing a code_revision that doesn't exist in the
        raw dir must fail loud, not silently fall back to mtime. This
        was the v1-style behavior that hid which SHA the projection
        actually came from."""
        old = self.raw_dir / "old.jsonl"
        # Match the file's "stem" SHA with each row's code_revision.
        append_row(old, _sample_row(9999.0, mnss=33, code_revision="old"))
        with self.assertRaises(FileNotFoundError):
            project_kernel(self.dir, "prefill_heavy",
                           code_revision="nonexistent")

    def test_fallback_to_most_recent_when_revision_none(self):
        """When the caller passes code_revision=None, discovery picks
        the most-recently-modified .jsonl and the output's
        code_revision is stamped from THAT file's name (not None,
        not the caller's request)."""
        old = self.raw_dir / "old.jsonl"
        new = self.raw_dir / "new.jsonl"
        append_row(old, _sample_row(9999.0, mnss=33, code_revision="old"))
        time.sleep(0.01)
        append_row(new, _sample_row(2391.0, mnss=4224, code_revision="new"))
        out = project_kernel(self.dir, "prefill_heavy",
                             code_revision=None)
        doc = json.loads(out.read_text())
        self.assertEqual(doc["raw_source"], "new.jsonl")
        # Stamped from the FILE, not from caller (fix #2).
        self.assertEqual(doc["code_revision"], "new")

    def test_explicit_raw_path_bypasses_discovery(self):
        a = self.raw_dir / "abc12345.jsonl"
        b = self.raw_dir / "def98765.jsonl"
        append_row(a, _sample_row(2391.0, mnss=4224,
                                  code_revision="abc12345"))
        append_row(b, _sample_row(99.0, mnss=33,
                                  code_revision="def98765"))
        # Pass raw_path explicitly; SHA arg ignored.
        out = project_kernel(self.dir, "prefill_heavy",
                             code_revision="abc12345",
                             raw_path=b)
        doc = json.loads(out.read_text())
        self.assertEqual(doc["raw_source"], "def98765.jsonl")
        # code_revision stamped from raw_path.stem (fix #2).
        self.assertEqual(doc["code_revision"], "def98765")
        self.assertEqual(doc["winners"][0]["tunable_params"]["mnss"], 33)

    def test_picks_lowest_latency_per_tuning_key(self):
        path = self.raw_dir / "abc12345.jsonl"
        # Same tuning_key, different tunable_params.
        append_row(path, _sample_row(2391.0, mnss=4224))
        append_row(path, _sample_row(2203.0, mnss=4224))    # winner
        append_row(path, _sample_row(2500.0, mnss=4224))
        out = project_kernel(self.dir, "prefill_heavy",
                             code_revision="abc12345")
        doc = json.loads(out.read_text())
        self.assertEqual(len(doc["winners"]), 1)
        self.assertEqual(doc["winners"][0]["latency_us"], 2203.0)

    def test_filters_non_success_status(self):
        path = self.raw_dir / "abc12345.jsonl"
        append_row(path, _sample_row(100.0, mnss=4224, status="FAILED_OOM"))
        append_row(path, _sample_row(100.0, mnss=4224, status="UNKNOWN_ERROR"))
        append_row(path, _sample_row(2391.0, mnss=4224, status="SUCCESS"))
        out = project_kernel(self.dir, "prefill_heavy",
                             code_revision="abc12345")
        doc = json.loads(out.read_text())
        self.assertEqual(doc["winners"][0]["latency_us"], 2391.0)

    def test_filters_rows_missing_latency_field(self):
        path = self.raw_dir / "abc12345.jsonl"
        # Row with status=SUCCESS but no latency_us — pathological.
        bad = _sample_row(0, mnss=4224)
        del bad["latency_us"]
        append_row(path, bad)
        # And a normal row.
        append_row(path, _sample_row(2391.0, mnss=4224))
        out = project_kernel(self.dir, "prefill_heavy",
                             code_revision="abc12345")
        doc = json.loads(out.read_text())
        self.assertEqual(doc["winners"][0]["latency_us"], 2391.0)

    def test_writes_envelope_shape(self):
        path = self.raw_dir / "abc12345.jsonl"
        append_row(path, _sample_row(2391.0, mnss=4224))
        out = project_kernel(self.dir, "prefill_heavy",
                             code_revision="abc12345")
        doc = json.loads(out.read_text())
        self.assertEqual(doc["schema_version"], 1)
        self.assertEqual(doc["workload"], "prefill_heavy")
        self.assertEqual(doc["code_revision"], "abc12345")
        self.assertEqual(doc["n_winners"], 1)
        self.assertIn("winners", doc)

    def test_idempotent_rewrite(self):
        path = self.raw_dir / "abc12345.jsonl"
        append_row(path, _sample_row(2391.0, mnss=4224))
        a = project_kernel(self.dir, "prefill_heavy",
                           code_revision="abc12345")
        first = a.read_text()
        b = project_kernel(self.dir, "prefill_heavy",
                           code_revision="abc12345")
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
        path = self.raw_dir / "abc12345.jsonl"
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
                             code_revision="abc12345")
        doc = json.loads(out.read_text())
        self.assertEqual(doc["n_winners"], 2)
        ks_and_lats = {(w["tuning_key"]["kernel_K"], w["latency_us"])
                       for w in doc["winners"]}
        self.assertEqual(ks_and_lats, {(256, 2203.0), (512, 3050.0)})


class TestCodeRevisionCrossValidation(unittest.TestCase):
    """fix #3: rows whose tuning_key.code_revision doesn't match the
    .jsonl filename's SHA should be caught at projection time."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.raw_dir = self.dir / "x.kernel.raw"
        self.raw_dir.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_mismatched_code_revision_raises(self):
        """File is named abc12345.jsonl but contains a row with
        code_revision='wrongsha'. Projection must reject."""
        from tools.tuning.v2.kernel.project import (
            CodeRevisionMismatchError,
            project_kernel,
        )
        path = self.raw_dir / "abc12345.jsonl"
        # Row has code_revision='wrongsha', mismatching filename.
        append_row(path, _sample_row(2391.0, mnss=4224,
                                     code_revision="wrongsha"))
        with self.assertRaises(CodeRevisionMismatchError):
            project_kernel(self.dir, "x", code_revision="abc12345")

    def test_row_with_no_code_revision_field_is_tolerated(self):
        """Rows missing the field entirely (legacy / migrated rows)
        don't fail validation — only rows that ASSERT a wrong SHA
        do."""
        from tools.tuning.v2.kernel.project import project_kernel
        path = self.raw_dir / "abc12345.jsonl"
        row = _sample_row(2391.0, mnss=4224)
        del row["tuning_key"]["code_revision"]
        append_row(path, row)
        # Should NOT raise.
        out = project_kernel(self.dir, "x", code_revision="abc12345")
        self.assertIsNotNone(out)


class TestDiscriminatorCrossValidation(unittest.TestCase):
    """fix: arch doc §13.4 line 452 — kernel_variant / hardware /
    schema_version mismatches across rows in one .raw partition
    must be caught at projection time (was only documented; never
    enforced)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.raw_dir = self.dir / "x.kernel.raw"
        self.raw_dir.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def _stamp(self, row: dict, **discriminators) -> dict:
        row["tuning_key"].update(discriminators)
        return row

    def test_mixed_kernel_variant_raises(self):
        from tools.tuning.v2.kernel.project import (
            DiscriminatorMismatchError, project_kernel,
        )
        path = self.raw_dir / "abc12345.jsonl"
        append_row(path, self._stamp(
            _sample_row(100.0, mnss=4224), kernel_variant="rpa_v3",
        ))
        append_row(path, self._stamp(
            _sample_row(110.0, mnss=4224),
            kernel_variant="rpa_v3_hd64",
        ))
        with self.assertRaisesRegex(DiscriminatorMismatchError,
                                    "kernel_variant"):
            project_kernel(self.dir, "x", code_revision="abc12345")

    def test_mixed_hardware_raises(self):
        from tools.tuning.v2.kernel.project import (
            DiscriminatorMismatchError, project_kernel,
        )
        path = self.raw_dir / "abc12345.jsonl"
        append_row(path, self._stamp(
            _sample_row(100.0, mnss=4224), hardware="tpu_v7x",
        ))
        append_row(path, self._stamp(
            _sample_row(110.0, mnss=4224), hardware="tpu_v6e",
        ))
        with self.assertRaisesRegex(DiscriminatorMismatchError,
                                    "hardware"):
            project_kernel(self.dir, "x", code_revision="abc12345")

    def test_mixed_schema_version_raises(self):
        from tools.tuning.v2.kernel.project import (
            DiscriminatorMismatchError, project_kernel,
        )
        path = self.raw_dir / "abc12345.jsonl"
        append_row(path, self._stamp(
            _sample_row(100.0, mnss=4224), schema_version=1,
        ))
        append_row(path, self._stamp(
            _sample_row(110.0, mnss=4224), schema_version=2,
        ))
        with self.assertRaisesRegex(DiscriminatorMismatchError,
                                    "schema_version"):
            project_kernel(self.dir, "x", code_revision="abc12345")

    def test_consistent_discriminators_pass(self):
        from tools.tuning.v2.kernel.project import project_kernel
        path = self.raw_dir / "abc12345.jsonl"
        for _ in range(3):
            append_row(path, self._stamp(
                _sample_row(100.0, mnss=4224),
                kernel_variant="rpa_v3",
                hardware="tpu_v7x",
                schema_version=1,
            ))
        out = project_kernel(self.dir, "x", code_revision="abc12345")
        self.assertIsNotNone(out)

    def test_missing_discriminator_field_tolerated(self):
        """Forward-compat: rows produced before the stamp landed
        won't carry the field. They don't contribute to the check
        — only rows that DO carry conflicting values fail."""
        from tools.tuning.v2.kernel.project import project_kernel
        path = self.raw_dir / "abc12345.jsonl"
        # First row: stamped.
        append_row(path, self._stamp(
            _sample_row(100.0, mnss=4224), kernel_variant="rpa_v3",
        ))
        # Second row: legacy, no kernel_variant.
        append_row(path, _sample_row(110.0, mnss=4224))
        out = project_kernel(self.dir, "x", code_revision="abc12345")
        self.assertIsNotNone(out)


class TestRawPrunePostProjection(unittest.TestCase):
    """TTL=2 prune runs after a successful kernel projection (arch
    doc §8 line 241)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.raw_dir = self.dir / "x.kernel.raw"
        self.raw_dir.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_third_oldest_partition_pruned(self):
        from tools.tuning.v2.kernel.project import project_kernel
        import time
        # Three partitions, ascending mtime.
        for sha in ("oldsha00", "midsha00", "newsha00"):
            p = self.raw_dir / f"{sha}.jsonl"
            append_row(p, _sample_row(100.0, mnss=4224,
                                      code_revision=sha))
            time.sleep(0.01)
        # Project against the newest.
        project_kernel(self.dir, "x", code_revision="newsha00")
        # Oldest gone, two newest remain.
        remaining = sorted(p.name for p in self.raw_dir.glob("*.jsonl"))
        self.assertEqual(remaining, ["midsha00.jsonl", "newsha00.jsonl"])


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
        append_row(raw_dir / "abc12345.jsonl",
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
        append_row(raw_dir / "abc12345.jsonl",
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
