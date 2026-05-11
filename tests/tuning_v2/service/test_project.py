# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for tools.tuning.v2.service.project."""

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
from tools.tuning.v2.service.project import (
    DEFAULT_OBJECTIVES,
    main as project_main,
    project_service,
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


def _row(*, mnb: int, mns: int, req_per_sec: float,
         ttft_mean: float, ttft_p99: float,
         status: str = "SUCCESS") -> dict:
    return {
        "status": status,
        "combo": {
            "MAX_NUM_BATCHED_TOKENS": mnb,
            "MAX_NUM_SEQS":           mns,
        },
        "metrics": {
            "req_per_sec":  req_per_sec,
            "ttft_mean_ms": ttft_mean,
            "ttft_p99_ms":  ttft_p99,
        },
    }


class TestProjectService(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.raw_dir = self.dir / "prefill_heavy.service.raw"
        self.raw_dir.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_returns_none_when_no_raw_dir(self):
        empty = self.dir / "no_workload"
        empty.mkdir()
        out = project_service(empty, "missing", service_revision="sha1-sha2")
        self.assertIsNone(out)

    def test_returns_none_when_raw_dir_empty(self):
        out = project_service(self.dir, "prefill_heavy",
                              service_revision="sha1-sha2")
        self.assertIsNone(out)

    def test_picks_matching_sha_file(self):
        a = self.raw_dir / "sha-a.jsonl"
        b = self.raw_dir / "sha-b.jsonl"
        append_row(a, _row(mnb=8192,  mns=128,  req_per_sec=4.79,
                           ttft_mean=100, ttft_p99=200))
        append_row(b, _row(mnb=131072, mns=1000, req_per_sec=4.90,
                           ttft_mean=99,  ttft_p99=199))
        out = project_service(self.dir, "prefill_heavy",
                              service_revision="sha-a")
        doc = json.loads(out.read_text())
        self.assertEqual(doc["raw_source"], "sha-a.jsonl")
        # Throughput winner from a-only: req_per_sec=4.79.
        self.assertEqual(
            doc["winners"]["throughput_max"]["metrics"]["req_per_sec"],
            4.79,
        )

    def test_explicit_sha_missing_raises(self):
        """fix #2: explicit service_revision with no matching file
        fails loud."""
        old = self.raw_dir / "old.jsonl"
        append_row(old, _row(mnb=8192, mns=128, req_per_sec=4.0,
                             ttft_mean=300, ttft_p99=400))
        with self.assertRaises(FileNotFoundError):
            project_service(self.dir, "prefill_heavy",
                            service_revision="nonexistent-sha")

    def test_fallback_to_most_recent_when_revision_none(self):
        old = self.raw_dir / "old.jsonl"
        new = self.raw_dir / "new.jsonl"
        append_row(old, _row(mnb=8192, mns=128, req_per_sec=4.0,
                             ttft_mean=300, ttft_p99=400))
        time.sleep(0.01)
        append_row(new, _row(mnb=131072, mns=1000, req_per_sec=4.9,
                             ttft_mean=100, ttft_p99=200))
        out = project_service(self.dir, "prefill_heavy",
                              service_revision=None)
        doc = json.loads(out.read_text())
        self.assertEqual(doc["raw_source"], "new.jsonl")
        # Stamped from the FILE, not None (fix #2).
        self.assertEqual(doc["service_revision"], "new")

    def test_explicit_raw_path_bypasses_discovery(self):
        a = self.raw_dir / "a.jsonl"
        b = self.raw_dir / "b.jsonl"
        append_row(a, _row(mnb=8192, mns=128, req_per_sec=4.79,
                           ttft_mean=100, ttft_p99=200))
        append_row(b, _row(mnb=16384, mns=256, req_per_sec=4.66,
                           ttft_mean=110, ttft_p99=220))
        out = project_service(self.dir, "prefill_heavy",
                              service_revision="anything",
                              raw_path=b)
        doc = json.loads(out.read_text())
        self.assertEqual(doc["raw_source"], "b.jsonl")
        self.assertEqual(
            doc["winners"]["throughput_max"]["combo"]["MAX_NUM_BATCHED_TOKENS"],
            16384,
        )

    def test_throughput_max_picks_highest_req_per_sec(self):
        path = self.raw_dir / "sha.jsonl"
        append_row(path, _row(mnb=8192,  mns=128,  req_per_sec=4.79,
                              ttft_mean=100, ttft_p99=200))
        append_row(path, _row(mnb=131072, mns=1000, req_per_sec=4.90,
                              ttft_mean=104, ttft_p99=203))
        append_row(path, _row(mnb=16384, mns=128,  req_per_sec=4.66,
                              ttft_mean=110, ttft_p99=220))
        out = project_service(self.dir, "prefill_heavy",
                              service_revision="sha")
        doc = json.loads(out.read_text())
        w = doc["winners"]["throughput_max"]
        self.assertEqual(w["metrics"]["req_per_sec"], 4.90)
        self.assertEqual(w["combo"]["MAX_NUM_BATCHED_TOKENS"], 131072)

    def test_ttft_min_picks_lowest_ttft_mean(self):
        path = self.raw_dir / "sha.jsonl"
        append_row(path, _row(mnb=8192,  mns=1, req_per_sec=2.50,
                              ttft_mean=400, ttft_p99=400))
        append_row(path, _row(mnb=16384, mns=1, req_per_sec=2.55,
                              ttft_mean=388, ttft_p99=388))
        append_row(path, _row(mnb=32768, mns=1, req_per_sec=2.50,
                              ttft_mean=395, ttft_p99=395))
        out = project_service(self.dir, "prefill_heavy",
                              service_revision="sha")
        doc = json.loads(out.read_text())
        w = doc["winners"]["ttft_min"]
        self.assertEqual(w["metrics"]["ttft_mean_ms"], 388)

    def test_p99_min_picks_lowest_p99(self):
        path = self.raw_dir / "sha.jsonl"
        append_row(path, _row(mnb=8192,  mns=128, req_per_sec=4.79,
                              ttft_mean=100, ttft_p99=210))
        append_row(path, _row(mnb=131072, mns=1000, req_per_sec=4.90,
                              ttft_mean=104, ttft_p99=203))
        out = project_service(self.dir, "prefill_heavy",
                              service_revision="sha")
        doc = json.loads(out.read_text())
        w = doc["winners"]["p99_min"]
        self.assertEqual(w["metrics"]["ttft_p99_ms"], 203)

    def test_filters_non_success_rows(self):
        path = self.raw_dir / "sha.jsonl"
        append_row(path, _row(mnb=8192,  mns=128, req_per_sec=99.0,
                              ttft_mean=1, ttft_p99=1,
                              status="FAILED_OOM"))
        append_row(path, _row(mnb=131072, mns=1000, req_per_sec=4.90,
                              ttft_mean=104, ttft_p99=203))
        out = project_service(self.dir, "prefill_heavy",
                              service_revision="sha")
        doc = json.loads(out.read_text())
        # The 99.0 row is filtered out by status; winner is the real one.
        self.assertEqual(
            doc["winners"]["throughput_max"]["metrics"]["req_per_sec"],
            4.90,
        )

    def test_filters_rows_missing_metric_field(self):
        path = self.raw_dir / "sha.jsonl"
        # Row with metrics dict missing req_per_sec.
        partial = _row(mnb=8192, mns=128, req_per_sec=99,
                       ttft_mean=1, ttft_p99=1)
        del partial["metrics"]["req_per_sec"]
        append_row(path, partial)
        # Normal row.
        append_row(path, _row(mnb=131072, mns=1000, req_per_sec=4.90,
                              ttft_mean=104, ttft_p99=203))
        out = project_service(self.dir, "prefill_heavy",
                              service_revision="sha")
        doc = json.loads(out.read_text())
        self.assertEqual(
            doc["winners"]["throughput_max"]["metrics"]["req_per_sec"],
            4.90,
        )
        # But ttft_min and p99_min, where partial row IS valid, may still
        # be the partial row if its ttft is better.

    def test_filters_rows_missing_metrics_subdict(self):
        """A row with no metrics field at all → filtered for every
        objective."""
        path = self.raw_dir / "sha.jsonl"
        no_metrics = _row(mnb=8192, mns=128, req_per_sec=99,
                          ttft_mean=1, ttft_p99=1)
        del no_metrics["metrics"]
        append_row(path, no_metrics)
        append_row(path, _row(mnb=131072, mns=1000, req_per_sec=4.90,
                              ttft_mean=104, ttft_p99=203))
        out = project_service(self.dir, "prefill_heavy",
                              service_revision="sha")
        doc = json.loads(out.read_text())
        self.assertEqual(
            doc["winners"]["throughput_max"]["metrics"]["req_per_sec"],
            4.90,
        )

    def test_filters_rows_with_none_metric(self):
        """Metric present but None → filtered."""
        path = self.raw_dir / "sha.jsonl"
        none_metric = _row(mnb=8192, mns=128, req_per_sec=99,
                           ttft_mean=1, ttft_p99=1)
        none_metric["metrics"]["req_per_sec"] = None
        append_row(path, none_metric)
        append_row(path, _row(mnb=131072, mns=1000, req_per_sec=4.90,
                              ttft_mean=104, ttft_p99=203))
        out = project_service(self.dir, "prefill_heavy",
                              service_revision="sha")
        doc = json.loads(out.read_text())
        self.assertEqual(
            doc["winners"]["throughput_max"]["metrics"]["req_per_sec"],
            4.90,
        )

    def test_filters_rows_with_non_numeric_metric(self):
        path = self.raw_dir / "sha.jsonl"
        bad = _row(mnb=8192, mns=128, req_per_sec=99,
                   ttft_mean=1, ttft_p99=1)
        bad["metrics"]["req_per_sec"] = "not a number"
        append_row(path, bad)
        append_row(path, _row(mnb=131072, mns=1000, req_per_sec=4.90,
                              ttft_mean=104, ttft_p99=203))
        out = project_service(self.dir, "prefill_heavy",
                              service_revision="sha")
        doc = json.loads(out.read_text())
        self.assertEqual(
            doc["winners"]["throughput_max"]["metrics"]["req_per_sec"],
            4.90,
        )

    def test_no_valid_rows_winner_is_none(self):
        path = self.raw_dir / "sha.jsonl"
        append_row(path, _row(mnb=8192, mns=128, req_per_sec=99,
                              ttft_mean=1, ttft_p99=1,
                              status="FAILED"))
        out = project_service(self.dir, "prefill_heavy",
                              service_revision="sha")
        doc = json.loads(out.read_text())
        self.assertIsNone(doc["winners"]["throughput_max"])
        self.assertIsNone(doc["winners"]["ttft_min"])
        self.assertIsNone(doc["winners"]["p99_min"])

    def test_writes_envelope_shape(self):
        path = self.raw_dir / "sha.jsonl"
        append_row(path, _row(mnb=8192, mns=128, req_per_sec=4.79,
                              ttft_mean=100, ttft_p99=200))
        out = project_service(self.dir, "prefill_heavy",
                              service_revision="sha")
        doc = json.loads(out.read_text())
        self.assertEqual(doc["schema_version"], 1)
        self.assertEqual(doc["workload"], "prefill_heavy")
        self.assertEqual(doc["service_revision"], "sha")
        self.assertIn("winners", doc)
        # All default objectives present.
        self.assertEqual(
            set(doc["winners"].keys()),
            set(DEFAULT_OBJECTIVES.keys()),
        )

    def test_custom_objectives(self):
        path = self.raw_dir / "sha.jsonl"
        append_row(path, _row(mnb=8192,   mns=128,  req_per_sec=4.79,
                              ttft_mean=100, ttft_p99=200))
        append_row(path, _row(mnb=131072, mns=1000, req_per_sec=4.90,
                              ttft_mean=104, ttft_p99=203))
        custom = {
            "fastest_throughput": (("metrics", "req_per_sec"), True),
        }
        out = project_service(self.dir, "prefill_heavy",
                              service_revision="sha",
                              objectives=custom)
        doc = json.loads(out.read_text())
        self.assertEqual(set(doc["winners"].keys()), {"fastest_throughput"})

    def test_default_service_revision_from_sha_module(self):
        path = self.raw_dir / "sha1-sha2.jsonl"
        append_row(path, _row(mnb=8192, mns=128, req_per_sec=4.79,
                              ttft_mean=100, ttft_p99=200))
        with mock.patch(
            "tools.tuning.v2.service.project.service_sha",
            return_value="sha1-sha2",
        ):
            out = project_service(self.dir, "prefill_heavy")
        doc = json.loads(out.read_text())
        self.assertEqual(doc["service_revision"], "sha1-sha2")

    def test_idempotent(self):
        path = self.raw_dir / "sha.jsonl"
        append_row(path, _row(mnb=8192, mns=128, req_per_sec=4.79,
                              ttft_mean=100, ttft_p99=200))
        a = project_service(self.dir, "prefill_heavy",
                            service_revision="sha")
        b = project_service(self.dir, "prefill_heavy",
                            service_revision="sha")
        self.assertEqual(a.read_text(), b.read_text())


class TestServiceRevisionCrossValidation(unittest.TestCase):
    """Review followup: symmetric with kernel-side fix-2 cross-
    validation. Per-row service_revision must match the .raw/<sha>.
    jsonl filename; mismatch raises."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.raw_dir = self.dir / "prefill_heavy.service.raw"
        self.raw_dir.mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_rows_with_matching_revision_project_cleanly(self):
        path = self.raw_dir / "sha1-sha2.jsonl"
        append_row(path, {
            "status":           "SUCCESS",
            "combo":            {"MAX_NUM_BATCHED_TOKENS": 8192,
                                 "MAX_NUM_SEQS": 128},
            "metrics":          {"req_per_sec": 4.0, "ttft_mean_ms": 100,
                                 "ttft_p99_ms": 200},
            "service_revision": "sha1-sha2",
        })
        out = project_service(self.dir, "prefill_heavy",
                              service_revision="sha1-sha2")
        self.assertIsNotNone(out)

    def test_row_with_mismatched_revision_raises(self):
        from tools.tuning.v2.service.project import (
            ServiceRevisionMismatchError,
        )
        path = self.raw_dir / "sha1-sha2.jsonl"
        append_row(path, {
            "status":           "SUCCESS",
            "combo":            {"MAX_NUM_BATCHED_TOKENS": 8192,
                                 "MAX_NUM_SEQS": 128},
            "metrics":          {"req_per_sec": 4.0, "ttft_mean_ms": 100,
                                 "ttft_p99_ms": 200},
            "service_revision": "WRONG-SHA",
        })
        with self.assertRaises(ServiceRevisionMismatchError) as cm:
            project_service(self.dir, "prefill_heavy",
                            service_revision="sha1-sha2")
        self.assertIn("WRONG-SHA", str(cm.exception))
        self.assertIn("sha1-sha2", str(cm.exception))

    def test_row_without_revision_field_is_tolerated(self):
        """Forward-compat: older raw files predate the per-row stamp.
        Don't break projection on them — only mismatches raise."""
        path = self.raw_dir / "sha1-sha2.jsonl"
        append_row(path, {
            "status":  "SUCCESS",
            "combo":   {"MAX_NUM_BATCHED_TOKENS": 8192,
                        "MAX_NUM_SEQS": 128},
            "metrics": {"req_per_sec": 4.0, "ttft_mean_ms": 100,
                        "ttft_p99_ms": 200},
            # no service_revision field
        })
        out = project_service(self.dir, "prefill_heavy",
                              service_revision="sha1-sha2")
        self.assertIsNotNone(out)


class TestServicePinKeyDiscriminatorValidation(unittest.TestCase):
    """fix: arch doc §13.4 line 452 — service-side discriminators
    come from kernel_pin_keys. Mixing rpa_v3 + rpa_v3_hd64 rows in
    one .service.raw partition must be caught at projection time."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.raw_dir = self.dir / "x.service.raw"
        self.raw_dir.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def _row(self, *, kernel_variant="rpa_v3", hardware="tpu_v7x",
             mnb=8192, req_per_sec=4.0) -> dict:
        return {
            "status": "SUCCESS",
            "combo": {"MAX_NUM_BATCHED_TOKENS": mnb,
                      "MAX_NUM_SEQS": 128},
            "metrics": {"req_per_sec": req_per_sec,
                        "ttft_mean_ms": 100.0,
                        "ttft_p99_ms": 200.0},
            "service_revision": "sha",
            "kernel_pin_keys": {
                "case": "logical", "page_size": 128,
                "kernel_K": 256, "mnss": 4224,
                "code_revision": "kabc",
                "kernel_variant": kernel_variant,
                "hardware": hardware,
            },
        }

    def test_mixed_kernel_variant_raises(self):
        from tools.tuning.v2.service.project import (
            ServiceDiscriminatorMismatchError, project_service,
        )
        path = self.raw_dir / "sha.jsonl"
        append_row(path, self._row(kernel_variant="rpa_v3"))
        append_row(path, self._row(kernel_variant="rpa_v3_hd64"))
        with self.assertRaisesRegex(
            ServiceDiscriminatorMismatchError, "kernel_variant",
        ):
            project_service(self.dir, "x",
                            service_revision="sha")

    def test_mixed_hardware_raises(self):
        from tools.tuning.v2.service.project import (
            ServiceDiscriminatorMismatchError, project_service,
        )
        path = self.raw_dir / "sha.jsonl"
        append_row(path, self._row(hardware="tpu_v7x"))
        append_row(path, self._row(hardware="tpu_v6e"))
        with self.assertRaisesRegex(
            ServiceDiscriminatorMismatchError, "hardware",
        ):
            project_service(self.dir, "x",
                            service_revision="sha")

    def test_consistent_discriminators_pass(self):
        from tools.tuning.v2.service.project import project_service
        path = self.raw_dir / "sha.jsonl"
        for r in range(3):
            append_row(path, self._row(req_per_sec=4.0 + 0.1 * r))
        out = project_service(self.dir, "x", service_revision="sha")
        self.assertIsNotNone(out)

    def test_row_without_pin_keys_tolerated(self):
        """Forward-compat: pre-stamping service rows had no
        kernel_pin_keys. Don't fail on absence — only on conflict."""
        from tools.tuning.v2.service.project import project_service
        path = self.raw_dir / "sha.jsonl"
        # First row: stamped.
        append_row(path, self._row(kernel_variant="rpa_v3"))
        # Second row: no kernel_pin_keys at all.
        legacy = {
            "status": "SUCCESS",
            "combo": {"MAX_NUM_BATCHED_TOKENS": 8192,
                      "MAX_NUM_SEQS": 128},
            "metrics": {"req_per_sec": 4.0,
                        "ttft_mean_ms": 100.0,
                        "ttft_p99_ms": 200.0},
            "service_revision": "sha",
        }
        append_row(path, legacy)
        out = project_service(self.dir, "x", service_revision="sha")
        self.assertIsNotNone(out)


class TestServiceRawPrunePostProjection(unittest.TestCase):
    """TTL=2 prune runs after a successful service projection."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.raw_dir = self.dir / "x.service.raw"
        self.raw_dir.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_third_oldest_partition_pruned(self):
        from tools.tuning.v2.service.project import project_service
        import time
        for sha in ("oldsha", "midsha", "newsha"):
            p = self.raw_dir / f"{sha}.jsonl"
            append_row(p, {
                "status": "SUCCESS",
                "combo": {"MAX_NUM_BATCHED_TOKENS": 8192,
                          "MAX_NUM_SEQS": 128},
                "metrics": {"req_per_sec": 4.0,
                            "ttft_mean_ms": 100.0,
                            "ttft_p99_ms": 200.0},
                "service_revision": sha,
            })
            time.sleep(0.01)
        project_service(self.dir, "x", service_revision="newsha")
        remaining = sorted(p.name for p in self.raw_dir.glob("*.jsonl"))
        self.assertEqual(remaining, ["midsha.jsonl", "newsha.jsonl"])


class TestResolveRawPath(unittest.TestCase):

    def test_none_service_revision_falls_back_to_mtime(self):
        from tools.tuning.v2.service.project import _resolve_raw_path
        tmp = tempfile.TemporaryDirectory()
        try:
            d = Path(tmp.name)
            raw = d / "x.service.raw"
            raw.mkdir()
            a = raw / "a.jsonl"
            b = raw / "b.jsonl"
            a.write_text('{"x":1}\n')
            time.sleep(0.01)
            b.write_text('{"x":2}\n')
            result = _resolve_raw_path(d, "x", service_revision=None)
            self.assertEqual(result, b)
        finally:
            tmp.cleanup()


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
        w = self.repo / "x.workload"
        w.write_text("MAX_NUM_SEQS=1\n")
        with mock.patch.object(sys, "stderr", new=open(os.devnull, "w")):
            rc = project_main([str(w)])
        self.assertEqual(rc, 1)

    def test_main_writes_output_returns_0(self):
        w = self.repo / "x.workload"
        w.write_text("MAX_NUM_SEQS=1\n")
        raw_dir = self.repo / "x.service.raw"
        raw_dir.mkdir()
        append_row(raw_dir / "sha.jsonl",
                   _row(mnb=8192, mns=128, req_per_sec=4.79,
                        ttft_mean=100, ttft_p99=200))
        with mock.patch(
            "tools.tuning.v2.service.project.service_sha",
            return_value="sha",
        ):
            with mock.patch.object(sys, "stdout", new=open(os.devnull, "w")):
                rc = project_main([str(w), "--no-commit"])
        self.assertEqual(rc, 0)
        self.assertTrue((self.repo / "x.service").exists())

    def test_main_commits_by_default(self):
        w = self.repo / "x.workload"
        w.write_text("MAX_NUM_SEQS=1\n")
        raw_dir = self.repo / "x.service.raw"
        raw_dir.mkdir()
        append_row(raw_dir / "sha.jsonl",
                   _row(mnb=8192, mns=128, req_per_sec=4.79,
                        ttft_mean=100, ttft_p99=200))
        with mock.patch(
            "tools.tuning.v2.service.project.service_sha",
            return_value="sha",
        ):
            with mock.patch(
                "tools.tuning.v2.service.project.commit_and_push"
            ) as cap:
                with mock.patch.object(
                    sys, "stdout", new=open(os.devnull, "w"),
                ):
                    rc = project_main([str(w)])
        self.assertEqual(rc, 0)
        cap.assert_called_once()


if __name__ == "__main__":
    unittest.main()
