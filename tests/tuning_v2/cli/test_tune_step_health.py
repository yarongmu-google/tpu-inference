# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for tools.tuning.v2.cli.tune_step_health (prev-5).

The May-11 incident's exact shape:
  - 45 UNKNOWN_ERROR rows
  - 5 SKIPPED rows
  - 0 SUCCESS rows
  - process died silently at the next combo's iters=3 mark

This module catches both shapes (silent death = no rows; zero
SUCCESS = the actual May-11 raw store) and fails the pipeline
BEFORE the sweep step blindly proceeds.
"""

import json
import tempfile
import time
import unittest
from pathlib import Path

from tools.tuning.v2.cli.tune_step_health import check, main


def _write_raw(workload: Path, rows: list[dict], sha: str = "abc12345") -> Path:
    """Helper: build a kernel.raw/<sha>.jsonl beside `workload`."""
    raw_dir = workload.parent / f"{workload.stem}.kernel.raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{sha}.jsonl"
    with open(raw_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return raw_path


class TestCheck(unittest.TestCase):
    """Exit-code contract: 0 = healthy, 1 = block pipeline."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.workload = self.dir / "prefill_heavy.workload"
        self.workload.write_text("# stub workload file\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_healthy_when_at_least_one_success(self):
        _write_raw(self.workload, [
            {"status": "SUCCESS", "latency_us": 100.0},
            {"status": "FAILED_OOM"},
            {"status": "SKIPPED"},
        ])
        code, msg = check(self.workload)
        self.assertEqual(code, 0)
        self.assertIn("OK", msg)
        self.assertIn("SUCCESS=1", msg)

    def test_no_raw_dir_silent_death(self):
        """The worst-case May-11 shape: no .kernel.raw/ at all.
        Either tune step never ran or process died before its first
        commit / first row write."""
        code, msg = check(self.workload)
        self.assertEqual(code, 1)
        self.assertIn("no kernel.raw", msg.lower())

    def test_empty_raw_file_treated_as_silent_death(self):
        """Empty .jsonl: file exists but no rows — process opened
        the file but died (or enumerator yielded nothing)."""
        _write_raw(self.workload, [])
        code, msg = check(self.workload)
        self.assertEqual(code, 1)
        self.assertIn("empty", msg.lower())

    def test_zero_success_rows_blocks_pipeline(self):
        """The exact May-11 shape: rows exist, none are SUCCESS.
        Without this check the sweep step would run and fail with a
        cryptic 'no kernel winners' error after a fresh JIT cycle."""
        _write_raw(self.workload, [
            {"status": "UNKNOWN_ERROR"} for _ in range(45)
        ] + [
            {"status": "SKIPPED"} for _ in range(5)
        ])
        code, msg = check(self.workload)
        self.assertEqual(code, 1)
        self.assertIn("0 SUCCESS", msg)
        # Breakdown should include both observed statuses.
        self.assertIn("UNKNOWN_ERROR=45", msg)
        self.assertIn("SKIPPED=5", msg)

    def test_zero_success_with_dynamic_prune_blocks_pipeline(self):
        """Post-prev-4 May-11 shape: 5 FAILED_OOM + N
        SKIPPED_DYNAMIC_PRUNE = 0 SUCCESS. Still blocks the pipeline
        — the operator needs to widen the search space."""
        _write_raw(self.workload, [
            {"status": "FAILED_OOM"} for _ in range(5)
        ] + [
            {"status": "SKIPPED_DYNAMIC_PRUNE"} for _ in range(40)
        ])
        code, msg = check(self.workload)
        self.assertEqual(code, 1)
        self.assertIn("0 SUCCESS", msg)

    def test_resolves_most_recent_jsonl_when_multiple(self):
        """Multi-SHA stores: TTL=2 keeps the previous SHA's file. The
        health check uses the most-recently-modified one (matches the
        projection step's choice)."""
        _write_raw(self.workload, [
            {"status": "SUCCESS", "latency_us": 100.0},
        ], sha="old_sha_")
        # Touch the new one slightly after.
        time.sleep(0.01)
        _write_raw(self.workload, [
            {"status": "UNKNOWN_ERROR"} for _ in range(10)
        ], sha="new_sha_")
        code, _msg = check(self.workload)
        # Most recent has 0 SUCCESS → fail.
        self.assertEqual(code, 1)

    def test_malformed_row_counted_separately(self):
        """A truncated JSONL line (process killed mid-write) shouldn't
        crash the health check — count it as MALFORMED so the
        operator sees it."""
        raw_dir = self.workload.parent / f"{self.workload.stem}.kernel.raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / "abc12345.jsonl"
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"status": "SUCCESS",
                                "latency_us": 100.0}) + "\n")
            f.write('{"status": "UNKNOWN_ERROR", "tuning_key": ')
            # Truncated — no newline, no closing braces.
        code, msg = check(self.workload)
        # SUCCESS row still counts; check passes despite the
        # malformed tail.
        self.assertEqual(code, 0)
        self.assertIn("MALFORMED=1", msg)


class TestCliMain(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.workload = self.dir / "prefill_heavy.workload"
        self.workload.write_text("# stub\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_exit_zero_on_healthy(self):
        _write_raw(self.workload, [
            {"status": "SUCCESS", "latency_us": 100.0},
        ])
        self.assertEqual(main([str(self.workload)]), 0)

    def test_exit_one_on_silent_death(self):
        self.assertEqual(main([str(self.workload)]), 1)


if __name__ == "__main__":
    unittest.main()
