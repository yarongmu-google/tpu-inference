# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for tools.tuning.v2.cli.impact — bottom-up impact-analysis
queries per architecture-doc §7."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.tuning.v2.cli.impact import (
    _coerce,
    by_kernel_key,
    by_service_combo,
    find_kernel_files,
    find_service_files,
    kernel_winners_matching,
    main as impact_main,
    service_winners_matching,
    stale_tunes,
)


def _kernel_doc(*winners) -> dict:
    return {"schema_version": 1, "workload": "test",
            "code_revision": "abc12345",
            "winners": list(winners)}


def _kernel_winner(*, case="logical", page_size=128, kernel_K=256,
                   mnss=4224, code_revision="abc12345") -> dict:
    return {
        "tuning_key": {
            "case": case, "page_size": page_size,
            "kernel_K": kernel_K, "code_revision": code_revision,
        },
        "tunable_params": {
            "bq_sz": 256, "bkv_sz": 2048,
            "bq_csz": 256, "bkv_csz": 512,
            "mnss": mnss,
        },
        "status": "SUCCESS", "latency_us": 100.0,
    }


def _service_doc(**winners) -> dict:
    return {"schema_version": 1, "workload": "test",
            "service_revision": "sha",
            "winners": winners or {"throughput_max": None}}


def _service_winner(*, mnb=8192, mns=128, req_per_sec=4.0):
    return {
        "status": "SUCCESS",
        "combo": {"MAX_NUM_BATCHED_TOKENS": mnb, "MAX_NUM_SEQS": mns},
        "metrics": {"req_per_sec": req_per_sec,
                    "ttft_mean_ms": 100.0, "ttft_p99_ms": 200.0},
    }


class TestCoerce(unittest.TestCase):

    def test_int_string_becomes_int(self):
        self.assertEqual(_coerce("256"), 256)

    def test_float_string_becomes_float(self):
        self.assertEqual(_coerce("4.79"), 4.79)

    def test_non_numeric_stays_string(self):
        self.assertEqual(_coerce("logical"), "logical")


class TestFindFiles(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_walks_recursively(self):
        (self.dir / "a" / "b").mkdir(parents=True)
        (self.dir / "a" / "x.kernel").write_text("{}")
        (self.dir / "a" / "b" / "y.kernel").write_text("{}")
        result = find_kernel_files(self.dir)
        self.assertEqual(len(result), 2)

    def test_excludes_production_kernel(self):
        (self.dir / "x.kernel").write_text("{}")
        (self.dir / "production.kernel").write_text("{}")
        result = find_kernel_files(self.dir)
        self.assertEqual([p.name for p in result], ["x.kernel"])

    def test_excludes_production_service(self):
        (self.dir / "x.service").write_text("{}")
        (self.dir / "production.service").write_text("{}")
        result = find_service_files(self.dir)
        self.assertEqual([p.name for p in result], ["x.service"])

    def test_missing_root_returns_empty(self):
        self.assertEqual(find_kernel_files(self.dir / "no"), [])
        self.assertEqual(find_service_files(self.dir / "no"), [])

    def test_skips_directories_matching_kernel_glob(self):
        """A directory named `foo.kernel/` matches the `*.kernel`
        rglob but isn't a file. Filter it out — only return files."""
        d = self.dir / "foo.kernel"   # directory ending in .kernel
        d.mkdir()
        f = self.dir / "real.kernel"
        f.write_text("{}")
        result = find_kernel_files(self.dir)
        self.assertEqual([p.name for p in result], ["real.kernel"])

    def test_skips_directories_matching_service_glob(self):
        d = self.dir / "foo.service"
        d.mkdir()
        f = self.dir / "real.service"
        f.write_text("{}")
        result = find_service_files(self.dir)
        self.assertEqual([p.name for p in result], ["real.service"])


class TestKernelWinnersMatching(unittest.TestCase):

    def test_matches_on_tuning_key_field(self):
        doc = _kernel_doc(_kernel_winner(kernel_K=256),
                          _kernel_winner(kernel_K=128))
        out = kernel_winners_matching(doc, "kernel_K", 256)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["tuning_key"]["kernel_K"], 256)

    def test_matches_on_tunable_params_field(self):
        """Operator queries `mnss 4224` and the field is in
        tunable_params, not tuning_key. The query should still find it."""
        doc = _kernel_doc(_kernel_winner(mnss=4224))
        out = kernel_winners_matching(doc, "mnss", 4224)
        self.assertEqual(len(out), 1)

    def test_no_match_returns_empty(self):
        doc = _kernel_doc(_kernel_winner(kernel_K=256))
        self.assertEqual(kernel_winners_matching(doc, "kernel_K", 512), [])

    def test_empty_winners_returns_empty(self):
        self.assertEqual(kernel_winners_matching({"winners": []},
                                                 "kernel_K", 256), [])

    def test_missing_winners_field_tolerated(self):
        self.assertEqual(kernel_winners_matching({}, "kernel_K", 256), [])


class TestServiceWinnersMatching(unittest.TestCase):

    def test_matches_on_combo_field(self):
        doc = _service_doc(throughput_max=_service_winner(mnb=131072),
                           ttft_min=_service_winner(mnb=8192))
        out = service_winners_matching(doc, "MAX_NUM_BATCHED_TOKENS",
                                       131072)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][0], "throughput_max")

    def test_skips_none_winners(self):
        """`p99_min: null` is a valid v2 .service shape (objective
        not yet swept). Must not crash."""
        doc = _service_doc(throughput_max=_service_winner(mnb=8192),
                           p99_min=None)
        out = service_winners_matching(doc, "MAX_NUM_BATCHED_TOKENS",
                                       8192)
        self.assertEqual(len(out), 1)

    def test_missing_winners_field_tolerated(self):
        self.assertEqual(service_winners_matching({}, "x", 1), [])


class TestByKernelKey(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_finds_matching_workload(self):
        a = self.root / "wl_a.kernel"
        b = self.root / "wl_b.kernel"
        a.write_text(json.dumps(_kernel_doc(_kernel_winner(kernel_K=256))))
        b.write_text(json.dumps(_kernel_doc(_kernel_winner(kernel_K=128))))
        result = by_kernel_key(self.root, "kernel_K", 256)
        paths = [p for p, _ in result]
        self.assertEqual(paths, [a])

    def test_malformed_json_skipped_not_crashed(self):
        bad = self.root / "broken.kernel"
        bad.write_text("{not valid")
        good = self.root / "wl.kernel"
        good.write_text(json.dumps(_kernel_doc(_kernel_winner())))
        # Query that the good file matches; the bad one is skipped.
        result = by_kernel_key(self.root, "case", "logical")
        self.assertEqual([p for p, _ in result], [good])

    def test_malformed_service_json_skipped(self):
        bad = self.root / "broken.service"
        bad.write_text("{not valid")
        good = self.root / "wl.service"
        good.write_text(json.dumps(_service_doc(
            throughput_max=_service_winner(mnb=8192),
        )))
        result = by_service_combo(self.root, "MAX_NUM_BATCHED_TOKENS",
                                  8192)
        self.assertEqual([p for p, _ in result], [good])

    def test_malformed_kernel_json_skipped_in_stale_tunes(self):
        bad = self.root / "broken.kernel"
        bad.write_text("{not valid")
        good = self.root / "wl.kernel"
        good.write_text(json.dumps(_kernel_doc(
            _kernel_winner(code_revision="OLD"),
        )))
        result = stale_tunes(self.root, "NEW")
        self.assertEqual([p for p, _ in result], [good])


class TestByServiceCombo(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_finds_matching_objectives_across_files(self):
        a = self.root / "wl_a.service"
        b = self.root / "wl_b.service"
        a.write_text(json.dumps(_service_doc(
            throughput_max=_service_winner(mnb=131072),
            ttft_min=_service_winner(mnb=8192),
        )))
        b.write_text(json.dumps(_service_doc(
            throughput_max=_service_winner(mnb=8192),
        )))
        result = by_service_combo(
            self.root, "MAX_NUM_BATCHED_TOKENS", 8192,
        )
        # Both files contain an 8192 winner.
        paths = sorted(p.name for p, _ in result)
        self.assertEqual(paths, ["wl_a.service", "wl_b.service"])


class TestStaleTunes(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_identifies_workloads_with_old_revisions(self):
        a = self.root / "wl_a.kernel"
        b = self.root / "wl_b.kernel"
        c = self.root / "wl_c.kernel"
        a.write_text(json.dumps(_kernel_doc(
            _kernel_winner(code_revision="OLD_SHA"),
        )))
        b.write_text(json.dumps(_kernel_doc(
            _kernel_winner(code_revision="NEW_SHA"),
        )))
        # c has multiple winners; one is stale.
        c.write_text(json.dumps(_kernel_doc(
            _kernel_winner(case="logical", code_revision="NEW_SHA"),
            _kernel_winner(case="prefill", code_revision="OLD_SHA"),
        )))
        result = stale_tunes(self.root, "NEW_SHA")
        paths = sorted(p.name for p, _ in result)
        self.assertEqual(paths, ["wl_a.kernel", "wl_c.kernel"])

    def test_workload_with_only_current_sha_omitted(self):
        a = self.root / "wl.kernel"
        a.write_text(json.dumps(_kernel_doc(
            _kernel_winner(code_revision="NEW"),
        )))
        self.assertEqual(stale_tunes(self.root, "NEW"), [])

    def test_collects_distinct_stale_revisions(self):
        a = self.root / "wl.kernel"
        a.write_text(json.dumps(_kernel_doc(
            _kernel_winner(case="logical", code_revision="OLD1"),
            _kernel_winner(case="prefill", code_revision="OLD2"),
        )))
        result = stale_tunes(self.root, "NEW")
        self.assertEqual(len(result), 1)
        _, revs = result[0]
        self.assertEqual(revs, ["OLD1", "OLD2"])


class TestCliMain(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_by_kernel_key_matches_returns_0(self):
        (self.dir / "wl.kernel").write_text(
            json.dumps(_kernel_doc(_kernel_winner(kernel_K=256))),
        )
        with mock.patch.object(sys, "stdout", new=open(os.devnull, "w")):
            rc = impact_main([
                "by-kernel-key", "kernel_K", "256",
                "--root", str(self.dir),
            ])
        self.assertEqual(rc, 0)

    def test_by_kernel_key_no_match_returns_1(self):
        (self.dir / "wl.kernel").write_text(
            json.dumps(_kernel_doc(_kernel_winner(kernel_K=128))),
        )
        with mock.patch.object(sys, "stdout", new=open(os.devnull, "w")):
            rc = impact_main([
                "by-kernel-key", "kernel_K", "999",
                "--root", str(self.dir),
            ])
        self.assertEqual(rc, 1)

    def test_by_service_combo_matches(self):
        (self.dir / "wl.service").write_text(json.dumps(_service_doc(
            throughput_max=_service_winner(mnb=131072),
        )))
        with mock.patch.object(sys, "stdout", new=open(os.devnull, "w")):
            rc = impact_main([
                "by-service-combo", "MAX_NUM_BATCHED_TOKENS", "131072",
                "--root", str(self.dir),
            ])
        self.assertEqual(rc, 0)

    def test_by_service_combo_no_match_returns_1(self):
        (self.dir / "wl.service").write_text(json.dumps(_service_doc(
            throughput_max=_service_winner(mnb=8192),
        )))
        with mock.patch.object(sys, "stdout", new=open(os.devnull, "w")):
            rc = impact_main([
                "by-service-combo", "MAX_NUM_BATCHED_TOKENS", "999",
                "--root", str(self.dir),
            ])
        self.assertEqual(rc, 1)

    def test_stale_tunes_matches(self):
        (self.dir / "wl.kernel").write_text(
            json.dumps(_kernel_doc(_kernel_winner(code_revision="OLD"))),
        )
        with mock.patch.object(sys, "stdout", new=open(os.devnull, "w")):
            rc = impact_main(["stale-tunes", "NEW",
                              "--root", str(self.dir)])
        self.assertEqual(rc, 0)

    def test_stale_tunes_no_match_returns_1(self):
        (self.dir / "wl.kernel").write_text(
            json.dumps(_kernel_doc(_kernel_winner(code_revision="NEW"))),
        )
        with mock.patch.object(sys, "stdout", new=open(os.devnull, "w")):
            rc = impact_main(["stale-tunes", "NEW",
                              "--root", str(self.dir)])
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
