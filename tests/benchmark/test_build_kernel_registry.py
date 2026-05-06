# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for build_kernel_registry manifest/runlog parsing.

Covers the three paths that produce the (case, case_set_id, db_path)
phase list:
  1. JSONL sidecar manifest (preferred).
  2. Legacy single-array JSON manifest (fallback for older runs).
  3. Runlog regex (fallback when no manifest is present).

local_query_min_latency is mocked so the tests dont need a real DB —
we only need to verify the right (case_set_id, db_path) pairs are fed
to it.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools.kernel.tuner.v1 import build_kernel_registry as bkr


def _make_runlog(dirpath, label, body=""):
    runlog = os.path.join(dirpath, f"tune_all_{label}.txt")
    with open(runlog, "w") as f:
        f.write(body)
    return runlog


def _make_jsonl_manifest(runlog, entries):
    path = runlog.replace(".txt", ".manifest.jsonl")
    with open(path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    return path


def _make_json_manifest(runlog, entries):
    path = runlog.replace(".txt", ".manifest.json")
    with open(path, "w") as f:
        json.dump(entries, f)
    return path


def _fake_winner(case, tk_extra=None, latency=1.0):
    """Synth a single winner result row matching the local_query_min_latency
    return contract."""
    tk = {"case": case}
    if tk_extra:
        tk.update(tk_extra)
    return {
        "tuning_key": tk,
        "tunable_params": {"bq_sz": 16, "bkv_sz": 64, "bq_csz": 16, "bkv_csz": 64},
        "Latency": latency,
    }


class TestBuildKernelRegistryManifest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cwd = os.getcwd()
        os.chdir(self.tmp)
        os.makedirs("tmp/log", exist_ok=True)

    def tearDown(self):
        os.chdir(self.cwd)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, runlog, out=None):
        argv = ["build_kernel_registry", runlog]
        if out:
            argv += ["--out", out]
        with patch.object(sys, "argv", argv):
            bkr.main()

    def test_jsonl_manifest_drives_phases(self):
        runlog = _make_runlog(".", "label_a")
        entries = [
            {"case": "decode", "case_set_id": "csid_d", "db_path": "/tmp/db_d"},
            {"case": "prefill", "case_set_id": "csid_p", "db_path": "/tmp/db_p"},
        ]
        _make_jsonl_manifest(runlog, entries)
        out_path = "out.kernel"

        captured_calls = []

        def fake_query(db_path, case_set_id, run_id):
            captured_calls.append((db_path, case_set_id, run_id))
            # Distinguish per case in the result so we can verify
            # the per-case bucketing is correct.
            case = next(e["case"] for e in entries if e["case_set_id"] == case_set_id)
            return [_fake_winner(case)]

        with patch.object(bkr, "local_query_min_latency", side_effect=fake_query):
            self._run(runlog, out=out_path)

        self.assertEqual(
            captured_calls,
            [("/tmp/db_d", "csid_d", "0"), ("/tmp/db_p", "csid_p", "0")],
        )
        with open(out_path) as f:
            data = json.load(f)
        self.assertEqual(len(data["results"]["decode"]), 1)
        self.assertEqual(len(data["results"]["prefill"]), 1)
        self.assertEqual(data["results"]["mixed"], [])

    def test_legacy_json_manifest_used_when_jsonl_absent(self):
        runlog = _make_runlog(".", "label_b")
        entries = [
            {"case": "mixed", "case_set_id": "csid_m", "db_path": "/tmp/db_m"},
        ]
        _make_json_manifest(runlog, entries)

        captured_calls = []
        with patch.object(
            bkr, "local_query_min_latency",
            side_effect=lambda db, csid, rid: captured_calls.append((db, csid, rid))
            or [_fake_winner("mixed")],
        ):
            self._run(runlog, out="out.kernel")

        self.assertEqual(captured_calls, [("/tmp/db_m", "csid_m", "0")])

    def test_jsonl_takes_precedence_over_json(self):
        runlog = _make_runlog(".", "label_c")
        # Both formats present; JSONL must win.
        _make_jsonl_manifest(runlog, [
            {"case": "decode", "case_set_id": "csid_jsonl", "db_path": "/tmp/jsonl"},
        ])
        _make_json_manifest(runlog, [
            {"case": "decode", "case_set_id": "csid_json", "db_path": "/tmp/json"},
        ])

        captured = []
        with patch.object(
            bkr, "local_query_min_latency",
            side_effect=lambda db, csid, rid: captured.append((db, csid))
            or [_fake_winner("decode")],
        ):
            self._run(runlog, out="out.kernel")

        self.assertEqual(captured, [("/tmp/jsonl", "csid_jsonl")])

    def test_jsonl_corrupt_falls_back_to_json(self):
        runlog = _make_runlog(".", "label_d")
        path = runlog.replace(".txt", ".manifest.jsonl")
        with open(path, "w") as f:
            f.write("{not json\n")  # garbage
        _make_json_manifest(runlog, [
            {"case": "prefill", "case_set_id": "csid_json", "db_path": "/tmp/json"},
        ])

        captured = []
        with patch.object(
            bkr, "local_query_min_latency",
            side_effect=lambda db, csid, rid: captured.append((db, csid))
            or [_fake_winner("prefill")],
        ):
            self._run(runlog, out="out.kernel")

        self.assertEqual(captured, [("/tmp/json", "csid_json")])

    def test_regex_fallback_when_no_manifest(self):
        body = (
            "===== 2026-05-04 12:00:00 Tuning decode (case_set_id=label_e_decode_X) =====\n"
            "Database initialized at /tmp/kernel_tuner_run_decode_X\n"
            "===== 2026-05-04 12:30:00 Tuning prefill (case_set_id=label_e_prefill_X) =====\n"
            "Database initialized at /tmp/kernel_tuner_run_prefill_X\n"
        )
        runlog = _make_runlog(".", "label_e", body=body)

        captured = []
        with patch.object(
            bkr, "local_query_min_latency",
            side_effect=lambda db, csid, rid: captured.append((db, csid))
            or [_fake_winner("decode")],
        ):
            self._run(runlog, out="out.kernel")

        self.assertEqual(captured, [
            ("/tmp/kernel_tuner_run_decode_X", "label_e_decode_X"),
            ("/tmp/kernel_tuner_run_prefill_X", "label_e_prefill_X"),
        ])

    def test_jsonl_skips_blank_lines(self):
        runlog = _make_runlog(".", "label_f")
        path = runlog.replace(".txt", ".manifest.jsonl")
        # Append-only writes can leave trailing blanks; reader must tolerate.
        with open(path, "w") as f:
            f.write('{"case":"decode","case_set_id":"a","db_path":"/tmp/a"}\n')
            f.write("\n")
            f.write('{"case":"prefill","case_set_id":"b","db_path":"/tmp/b"}\n')
            f.write("\n")

        captured = []
        with patch.object(
            bkr, "local_query_min_latency",
            side_effect=lambda db, csid, rid: captured.append((db, csid))
            or [_fake_winner("decode")],
        ):
            self._run(runlog, out="out.kernel")

        self.assertEqual(captured, [("/tmp/a", "a"), ("/tmp/b", "b")])


if __name__ == "__main__":
    unittest.main()
