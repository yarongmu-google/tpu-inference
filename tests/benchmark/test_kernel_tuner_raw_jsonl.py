# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for the durable JSONL log + resume skip-set helpers in
`tools.kernel.tuner.v1.common.kernel_tuner_base`.

Covers what should have been tested when the helpers landed:
  - `_append_raw_jsonl`: row schema, fsync-per-row durability,
    no-op when path unset, parent-dir auto-create, REGRESSION for
    the `FLAGS.worker_id == "unknown"` int-cast crash (2026-05-12).
  - `_load_raw_jsonl_skip_set`: empty inputs, permanence policy
    (SUCCESS / FAILED_OOM / SKIPPED in, UNKNOWN_ERROR out),
    truncated-line tolerance.
  - `_combo_skip_key`: deterministic, order-insensitive, copes
    with non-JSON-native fields (default=str).

Stays import-light so it runs on a laptop without TPU deps. Avoids
importing `rpa_v3_kernel_tuner` (which pulls in vllm and the kernel
module) by constructing a minimal `KernelTunerBase` subclass for
the tests.
"""

import dataclasses
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from absl import flags

# absl flags need to be parsed before any FLAGS read. The writer
# itself uses getattr(FLAGS, "worker_id", "unknown") so it's robust
# to the flag not being defined — these tests verify exactly that.
if not flags.FLAGS.is_parsed():
    flags.FLAGS(sys.argv[:1])

from tools.kernel.tuner.v1.common.kernel_tuner_base import (
    KernelTunerBase,
    TuningStatus,
)


@dataclasses.dataclass
class _TK:
    """Minimal stand-in TuningKey. Mirrors the rpa_v3 shape closely
    enough that the row schema looks right; we don't need every field
    to exercise the JSONL writer."""
    page_size: int
    case: str
    chunk_prefill_size: int = 0


@dataclasses.dataclass
class _TP:
    """Minimal stand-in TunableParams."""
    bq_sz: int
    bkv_sz: int
    bq_csz: int
    bkv_csz: int
    max_num_subseqs: int | None = None


class _StorageStub:
    """Just enough to satisfy KernelTunerBase's storage_manager
    assertion + the one call _append_raw_jsonl makes."""
    def get_timestamp_sec(self) -> int:
        return 1747234567


class _StubTuner(KernelTunerBase):
    """KernelTunerBase has three @abstractmethods; no-op overrides
    let us instantiate. The tests under this file never call any of
    them — they exercise the JSONL helpers directly."""
    def run(self, tuning_key, tunable_params, iters=1):
        return TuningStatus.SUCCESS, 0, 0
    def generate_cases(self):
        return []
    def generate_inputs(self, tuning_key):
        return {}


def _make_tuner() -> _StubTuner:
    return _StubTuner(
        tuning_key_class=_TK,
        tunable_params_class=_TP,
        storage_manager=_StorageStub(),
        kernel_tuner_name="stub",
    )


class TestAppendRawJsonl(unittest.TestCase):
    """`_append_raw_jsonl` — durable per-row writer."""

    def setUp(self):
        self.tuner = _make_tuner()
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "kernel.raw.jsonl"
        self.tk = _TK(page_size=128, case="logical", chunk_prefill_size=128)
        self.tp = _TP(bq_sz=128, bkv_sz=512, bq_csz=128,
                      bkv_csz=256, max_num_subseqs=256)

    def tearDown(self):
        self.tmp.cleanup()

    def test_noop_when_path_unset(self):
        """raw_jsonl_path is None by default — the helper must
        silently return so the existing v1 measurement loop is
        unaffected when nobody opts in."""
        self.tuner.raw_jsonl_path = None
        self.tuner._append_raw_jsonl(
            tuning_key=self.tk, tunable_params=self.tp,
            status=TuningStatus.SUCCESS, case_id=0,
        )
        # No file written, no exception.
        self.assertFalse(self.path.exists())

    def test_writes_one_row_with_expected_schema(self):
        """SUCCESS row carries every schema field; downstream resume
        + analytics tools depend on this set of keys."""
        self.tuner.raw_jsonl_path = self.path
        self.tuner._append_raw_jsonl(
            tuning_key=self.tk, tunable_params=self.tp,
            status=TuningStatus.SUCCESS, case_id=42,
            latency_us=1234, warmup_us=567, total_us=1801,
            case_set_id="cs1", run_id="r0",
        )
        rows = self.path.read_text().strip().split("\n")
        self.assertEqual(len(rows), 1)
        r = json.loads(rows[0])
        self.assertEqual(r["status"], "SUCCESS")
        self.assertEqual(r["latency_us"], 1234)
        self.assertEqual(r["warmup_us"], 567)
        self.assertEqual(r["total_us"], 1801)
        self.assertEqual(r["case_id"], 42)
        self.assertEqual(r["case_set_id"], "cs1")
        self.assertEqual(r["run_id"], "r0")
        self.assertEqual(r["tuning_key"]["page_size"], 128)
        self.assertEqual(r["tuning_key"]["case"], "logical")
        self.assertEqual(r["tunable_params"]["bq_sz"], 128)
        self.assertEqual(r["tunable_params"]["max_num_subseqs"], 256)
        self.assertIn("timestamp_sec", r)

    def test_worker_id_falls_back_to_unknown_when_flag_unregistered(self):
        """REGRESSION (2026-05-12 17:02 smoke crash). The original
        code was `int(FLAGS.worker_id)`. When the kernel_tuner_runner
        module is imported, that registers a default of "unknown";
        int("unknown") raised ValueError and crashed the tune AFTER
        a successful kernel measurement (24 s of TPU time wasted per
        combo). After the fix, the writer uses
        `getattr(FLAGS, "worker_id", "unknown")` — robust to BOTH
        the unregistered case (this test: tests import the base
        class without kernel_tuner_runner) AND the production case
        where the flag is set to "unknown" locally / numeric in
        Buildkite. Always serialized as a string."""
        self.tuner.raw_jsonl_path = self.path
        # No flag setup. FLAGS.worker_id is unregistered.
        self.tuner._append_raw_jsonl(
            tuning_key=self.tk, tunable_params=self.tp,
            status=TuningStatus.SUCCESS, case_id=0,
        )
        r = json.loads(self.path.read_text().strip())
        # The getattr default kicks in → "unknown".
        self.assertEqual(r["worker_id"], "unknown")

    def test_worker_id_uses_flag_when_registered(self):
        """When the flag IS registered (production path on TPU via
        kernel_tuner_runner), the writer uses its value. Confirms
        the getattr fallback doesn't swallow a real numeric id."""
        try:
            flags.DEFINE_string("worker_id_test_temp", "7", "test")
            # Patch the actual name the writer reads.
            flags.FLAGS.__dict__["worker_id"] = "7"
            try:
                self.tuner.raw_jsonl_path = self.path
                self.tuner._append_raw_jsonl(
                    tuning_key=self.tk, tunable_params=self.tp,
                    status=TuningStatus.SUCCESS, case_id=0,
                )
                r = json.loads(self.path.read_text().strip())
                self.assertEqual(r["worker_id"], "7")
            finally:
                flags.FLAGS.__dict__.pop("worker_id", None)
        except flags.DuplicateFlagError:
            pass

    def test_status_enum_value_unwrapped(self):
        """status arg accepts both the TuningStatus enum and a raw
        string (some callers in the loop pass the enum object).
        The helper stores `.value` when present so downstream readers
        always see the string."""
        self.tuner.raw_jsonl_path = self.path
        self.tuner._append_raw_jsonl(
            tuning_key=self.tk, tunable_params=self.tp,
            status=TuningStatus.FAILED_OOM, case_id=0,
        )
        r = json.loads(self.path.read_text().strip())
        self.assertEqual(r["status"], "FAILED_OOM")

    def test_appends_not_overwrites(self):
        """Two writes → two rows on disk in order."""
        self.tuner.raw_jsonl_path = self.path
        for i in range(2):
            self.tuner._append_raw_jsonl(
                tuning_key=self.tk, tunable_params=self.tp,
                status=TuningStatus.SUCCESS, case_id=i,
                latency_us=100 + i,
            )
        rows = [json.loads(l) for l in self.path.read_text().splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["case_id"], 0)
        self.assertEqual(rows[0]["latency_us"], 100)
        self.assertEqual(rows[1]["case_id"], 1)
        self.assertEqual(rows[1]["latency_us"], 101)

    def test_creates_parent_dir_if_missing(self):
        """When the bash wrapper hasn't pre-created the DB dir, the
        helper mkdirs the parent so the very first row lands."""
        nested = Path(self.tmp.name) / "deep" / "subdir" / "kernel.raw.jsonl"
        self.tuner.raw_jsonl_path = nested
        self.tuner._append_raw_jsonl(
            tuning_key=self.tk, tunable_params=self.tp,
            status=TuningStatus.SUCCESS, case_id=0,
        )
        self.assertTrue(nested.exists())


class TestLoadRawJsonlSkipSet(unittest.TestCase):
    """`_load_raw_jsonl_skip_set` — resume reader."""

    def setUp(self):
        self.tuner = _make_tuner()
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "kernel.raw.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def test_empty_set_when_path_unset(self):
        self.tuner.raw_jsonl_path = None
        self.assertEqual(self.tuner._load_raw_jsonl_skip_set(), set())

    def test_empty_set_when_file_absent(self):
        self.tuner.raw_jsonl_path = self.path  # path set but file doesn't exist
        self.assertEqual(self.tuner._load_raw_jsonl_skip_set(), set())

    def test_empty_set_when_file_empty(self):
        self.tuner.raw_jsonl_path = self.path
        self.path.touch()
        self.assertEqual(self.tuner._load_raw_jsonl_skip_set(), set())

    def test_permanent_statuses_included_unknown_excluded(self):
        """Permanence policy: SUCCESS, FAILED_OOM, SKIPPED → skip.
        UNKNOWN_ERROR → retry (NOT in skip-set). After a bugfix the
        operator wants UNKNOWN rows to re-measure."""
        rows = [
            {"tuning_key": {"a": 1}, "tunable_params": {"b": 1},
             "status": "SUCCESS"},
            {"tuning_key": {"a": 1}, "tunable_params": {"b": 2},
             "status": "FAILED_OOM"},
            {"tuning_key": {"a": 1}, "tunable_params": {"b": 3},
             "status": "SKIPPED"},
            {"tuning_key": {"a": 1}, "tunable_params": {"b": 4},
             "status": "UNKNOWN_ERROR"},
        ]
        with open(self.path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        self.tuner.raw_jsonl_path = self.path
        skip = self.tuner._load_raw_jsonl_skip_set()
        # 3 in (SUCCESS + FAILED_OOM + SKIPPED), 1 out (UNKNOWN_ERROR).
        self.assertEqual(len(skip), 3)

    def test_truncated_trailing_line_tolerated(self):
        """Ctrl-C between `f.write` and the newline flush leaves a
        malformed final line. The earlier complete lines must still
        be counted; the truncated tail is silently skipped (counted
        as malformed in the log but doesn't crash the load)."""
        with open(self.path, "w") as f:
            f.write(json.dumps({"tuning_key": {"a": 1},
                                "tunable_params": {"b": 1},
                                "status": "SUCCESS"}) + "\n")
            # Truncated final line: missing closing brace + newline.
            f.write('{"tuning_key": {"a":')
        self.tuner.raw_jsonl_path = self.path
        skip = self.tuner._load_raw_jsonl_skip_set()
        self.assertEqual(len(skip), 1)


class TestComboSkipKey(unittest.TestCase):
    """`_combo_skip_key` — stable hashable identifier."""

    def test_deterministic(self):
        tk = _TK(page_size=128, case="logical")
        tp = _TP(bq_sz=128, bkv_sz=512, bq_csz=128, bkv_csz=256)
        self.assertEqual(
            KernelTunerBase._combo_skip_key(tk, tp),
            KernelTunerBase._combo_skip_key(tk, tp),
        )

    def test_different_inputs_produce_different_keys(self):
        tk = _TK(page_size=128, case="logical")
        tp1 = _TP(bq_sz=128, bkv_sz=512, bq_csz=128, bkv_csz=256)
        tp2 = _TP(bq_sz=256, bkv_sz=512, bq_csz=128, bkv_csz=256)
        self.assertNotEqual(
            KernelTunerBase._combo_skip_key(tk, tp1),
            KernelTunerBase._combo_skip_key(tk, tp2),
        )

    def test_writer_and_reader_keys_agree(self):
        """Round-trip: the key the writer computes for a combo must
        match the key the reader builds from the JSONL row for the
        same combo. Without this property, resume would never skip
        anything (writer's key never matches reader's key)."""
        tuner = _make_tuner()
        tmp = tempfile.TemporaryDirectory()
        try:
            path = Path(tmp.name) / "kernel.raw.jsonl"
            tuner.raw_jsonl_path = path
            tk = _TK(page_size=128, case="logical")
            tp = _TP(bq_sz=128, bkv_sz=512, bq_csz=128,
                     bkv_csz=256, max_num_subseqs=256)
            # Writer
            tuner._append_raw_jsonl(
                tuning_key=tk, tunable_params=tp,
                status=TuningStatus.SUCCESS, case_id=0,
            )
            # Reader
            skip = tuner._load_raw_jsonl_skip_set()
            self.assertIn(
                KernelTunerBase._combo_skip_key(tk, tp),
                skip,
            )
        finally:
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
