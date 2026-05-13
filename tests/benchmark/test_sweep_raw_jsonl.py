# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for the durable JSONL log + resume skip-set helpers in
`tools.benchmark.sweep`.

Mirrors `test_kernel_tuner_raw_jsonl` for the service-sweep side.

Covers:
  - `_append_service_raw_jsonl`: row schema for COMPLETED_FRESH /
    SKIPPED_RESUMED / FAILED; metrics.txt parsing; FAILED-with-no-
    metrics tolerance; parent-dir auto-create.
  - `_load_service_raw_skip_set`: empty inputs, permanence policy
    (completed_fresh + skipped_resumed in, failed out),
    truncated-line tolerance.
  - `run_sweep` integration: skip-set bypasses run_one for already-
    completed combos.
"""

import json
import tempfile
import unittest
from pathlib import Path

from tools.benchmark.sweep import (
    RunResult, RunStatus,
    _append_service_raw_jsonl, _load_service_raw_skip_set, run_sweep,
)


class TestAppendServiceRawJsonl(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.sweep_dir = Path(self.tmp.name)
        self.raw_path = self.sweep_dir / "service.raw.jsonl"
        self.combo_rdir = self.sweep_dir / "abc123"
        self.combo_rdir.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_completed_row_with_parsed_metrics(self):
        """COMPLETED_FRESH: row carries combo, status, return_code,
        duration_seconds, parsed metrics, timestamp_sec."""
        (self.combo_rdir / "metrics.txt").write_text(
            "MeanTTFT=100.5\nP99TTFT=250.0\nRequestThroughput=4.9\n"
        )
        r = RunResult(
            status=RunStatus.COMPLETED_FRESH, combo_id="abc123",
            result_dir=self.combo_rdir, return_code=0,
            duration_seconds=42.5,
        )
        _append_service_raw_jsonl(
            self.raw_path, {"MAX_NUM_SEQS": "128"}, r,
        )
        row = json.loads(self.raw_path.read_text().strip())
        self.assertEqual(row["combo"], {"MAX_NUM_SEQS": "128"})
        self.assertEqual(row["combo_id"], "abc123")
        self.assertEqual(row["status"], "completed_fresh")
        self.assertEqual(row["return_code"], 0)
        self.assertEqual(row["duration_seconds"], 42.5)
        self.assertEqual(row["metrics"]["RequestThroughput"], "4.9")
        self.assertEqual(row["metrics"]["MeanTTFT"], "100.5")
        self.assertIn("timestamp_sec", row)
        self.assertIsNone(row["error"])

    def test_failed_row_with_no_metrics_txt_tolerated(self):
        """A FAILED combo may not have produced a metrics.txt at all
        (server crashed early, timeout, etc.). The row still lands;
        metrics={}."""
        r = RunResult(
            status=RunStatus.FAILED, combo_id="failed",
            result_dir=self.sweep_dir / "failed_dir",  # doesn't exist
            return_code=1, duration_seconds=2.0,
            error="server crashed",
        )
        _append_service_raw_jsonl(self.raw_path, {"X": "1"}, r)
        row = json.loads(self.raw_path.read_text().strip())
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["error"], "server crashed")
        self.assertEqual(row["metrics"], {})

    def test_skipped_resumed_row(self):
        """SKIPPED_RESUMED: pre-existing metrics.txt was reused by
        run_one (or the new fast-path in run_sweep). Row should still
        carry the metrics."""
        (self.combo_rdir / "metrics.txt").write_text(
            "RequestThroughput=4.78\n"
        )
        r = RunResult(
            status=RunStatus.SKIPPED_RESUMED, combo_id="abc123",
            result_dir=self.combo_rdir,
        )
        _append_service_raw_jsonl(self.raw_path, {"X": "1"}, r)
        row = json.loads(self.raw_path.read_text().strip())
        self.assertEqual(row["status"], "skipped_resumed")
        self.assertEqual(row["metrics"]["RequestThroughput"], "4.78")

    def test_appends_not_overwrites(self):
        r1 = RunResult(status=RunStatus.FAILED, combo_id="c1",
                       result_dir=self.combo_rdir)
        r2 = RunResult(status=RunStatus.FAILED, combo_id="c2",
                       result_dir=self.combo_rdir)
        _append_service_raw_jsonl(self.raw_path, {"k": "1"}, r1)
        _append_service_raw_jsonl(self.raw_path, {"k": "2"}, r2)
        rows = self.raw_path.read_text().strip().splitlines()
        self.assertEqual(len(rows), 2)
        self.assertEqual(json.loads(rows[0])["combo_id"], "c1")
        self.assertEqual(json.loads(rows[1])["combo_id"], "c2")

    def test_creates_parent_dir_if_missing(self):
        nested = self.sweep_dir / "deeper" / "service.raw.jsonl"
        r = RunResult(status=RunStatus.FAILED, combo_id="c",
                      result_dir=self.combo_rdir)
        _append_service_raw_jsonl(nested, {}, r)
        self.assertTrue(nested.exists())


class TestLoadServiceRawSkipSet(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.raw_path = Path(self.tmp.name) / "service.raw.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def test_empty_set_when_file_absent(self):
        self.assertEqual(_load_service_raw_skip_set(self.raw_path), set())

    def test_empty_set_when_file_empty(self):
        self.raw_path.touch()
        self.assertEqual(_load_service_raw_skip_set(self.raw_path), set())

    def test_completed_and_resumed_included_failed_excluded(self):
        """Permanence policy: completed_fresh + skipped_resumed → skip.
        failed → retry (NOT in skip-set). Matches kernel-tune's
        UNKNOWN_ERROR-retries-on-resume convention."""
        rows = [
            {"combo_id": "ok", "status": "completed_fresh"},
            {"combo_id": "resumed", "status": "skipped_resumed"},
            {"combo_id": "failed", "status": "failed"},
        ]
        with open(self.raw_path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        s = _load_service_raw_skip_set(self.raw_path)
        self.assertEqual(s, {"ok", "resumed"})

    def test_truncated_trailing_line_tolerated(self):
        with open(self.raw_path, "w") as f:
            f.write(json.dumps(
                {"combo_id": "ok", "status": "completed_fresh"},
            ) + "\n")
            f.write('{"combo_id":')  # truncated
        s = _load_service_raw_skip_set(self.raw_path)
        self.assertEqual(s, {"ok"})

    def test_missing_combo_id_or_status_ignored(self):
        """Malformed rows (no combo_id, no status, etc.) shouldn't
        crash the loader — silently skip and continue."""
        with open(self.raw_path, "w") as f:
            f.write(json.dumps({"status": "completed_fresh"}) + "\n")
            f.write(json.dumps({"combo_id": "x"}) + "\n")
            f.write(json.dumps(
                {"combo_id": "valid", "status": "completed_fresh"},
            ) + "\n")
        s = _load_service_raw_skip_set(self.raw_path)
        self.assertEqual(s, {"valid"})


class TestRunSweepResumeIntegration(unittest.TestCase):
    """`run_sweep` skips combos whose combo_id is in the JSONL skip-set
    — verifies the path that wires the loader into the main loop."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _write_spec(self, sweep_name: str) -> Path:
        """Minimal valid sweep spec. case_file must exist on disk —
        load_spec validates it. Use an absolute path to a real
        workload so this test runs from any cwd."""
        import os
        repo_root = Path(__file__).resolve().parents[2]
        case_file = (
            repo_root
            / "tools/benchmark/cases/v7x/llama3_8b/prefill_heavy.workload"
        )
        spec_path = self.base_dir / "test.service"
        spec_path.write_text(json.dumps({
            "case_file":  str(case_file),
            "sweep_name": sweep_name,
            "sweep_axes": {"MAX_NUM_SEQS": [128, 256]},
            "fixed": {"BLOCK_SIZE": "128"},
            "timeout_seconds": 60,
        }))
        return spec_path

    def test_skip_set_short_circuits_run_one(self):
        """Pre-populate service.raw.jsonl with completed_fresh rows
        for both combos. run_sweep with these should call run_one
        ZERO times — every combo is served from the skip-set."""
        from tools.benchmark.sweep import (
            enumerate_combos, load_spec, combo_id, case_name_from_path,
            sweep_dir,
        )
        spec_path = self._write_spec("smoketest")
        # Pre-populate JSONL with the combo_ids the enumerator will
        # produce, marked completed_fresh.
        spec = load_spec(spec_path)
        combos = enumerate_combos(spec)
        case_name = case_name_from_path(spec["case_file"])
        s_dir = sweep_dir(self.base_dir, case_name, "smoketest")
        s_dir.mkdir(parents=True, exist_ok=True)
        raw_path = s_dir / "service.raw.jsonl"
        with open(raw_path, "w") as f:
            for c in combos:
                f.write(json.dumps({
                    "combo_id": combo_id(c),
                    "status": "completed_fresh",
                }) + "\n")
        # Stub run_one — it must NOT be called.
        run_one_calls = []
        def fake_run_one(spec, combo, **kw):
            run_one_calls.append(combo)
            return RunResult(
                status=RunStatus.COMPLETED_FRESH,
                combo_id=combo_id(combo),
                result_dir=Path("never"),
            )
        results = run_sweep(
            spec_path, base_dir=self.base_dir,
            run_one_fn=fake_run_one,
        )
        # All combos resumed from JSONL → no run_one invocations.
        self.assertEqual(len(run_one_calls), 0)
        # results still has one entry per combo, all SKIPPED_RESUMED.
        self.assertEqual(len(results), len(combos))
        self.assertTrue(
            all(r.status == RunStatus.SKIPPED_RESUMED for r in results),
        )

    def test_unknown_combo_calls_run_one(self):
        """Confirm the inverse: a combo NOT in the skip-set goes
        through run_one as normal. Without this, we couldn't tell
        the skip path from a no-op."""
        from tools.benchmark.sweep import combo_id
        spec_path = self._write_spec("smoketest2")
        run_one_calls = []
        def fake_run_one(spec, combo, **kw):
            run_one_calls.append(combo)
            return RunResult(
                status=RunStatus.COMPLETED_FRESH,
                combo_id=combo_id(combo),
                result_dir=Path("never"),
            )
        results = run_sweep(
            spec_path, base_dir=self.base_dir,
            run_one_fn=fake_run_one,
        )
        # No prior JSONL → every combo calls run_one.
        self.assertEqual(len(run_one_calls), 2)
        self.assertEqual(len(results), 2)


if __name__ == "__main__":
    unittest.main()
