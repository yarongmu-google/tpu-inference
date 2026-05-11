# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for tools.tuning.v2.core.raw_store.

Covers: append + read round-trip; parent-dir creation; missing/empty
files; blank-line tolerance; malformed-row tolerance (the crashed-write
case); skip-set construction with and without status filter; compact
JSON serialization; unicode pass-through; resume from existing file.
"""

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

from tools.tuning.v2.core.raw_store import (
    append_row,
    build_skip_set,
    read_rows,
)


class TestRawStore(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    # ----- append + read round-trip -----

    def test_append_then_read_single_row(self):
        path = self.dir / "single.jsonl"
        append_row(path, {"k": "v"})
        self.assertEqual(list(read_rows(path)), [{"k": "v"}])

    def test_append_then_read_many_rows_in_order(self):
        path = self.dir / "many.jsonl"
        for i in range(20):
            append_row(path, {"i": i, "label": f"row-{i}"})
        rows = list(read_rows(path))
        self.assertEqual(len(rows), 20)
        for i, row in enumerate(rows):
            self.assertEqual(row, {"i": i, "label": f"row-{i}"})

    def test_append_creates_parent_dirs(self):
        path = self.dir / "a" / "b" / "c" / "nested.jsonl"
        append_row(path, {"x": 1})
        self.assertTrue(path.exists())
        self.assertEqual(list(read_rows(path)), [{"x": 1}])

    def test_append_uses_compact_json(self):
        """No spaces between keys/values — saves disk on large stores."""
        path = self.dir / "compact.jsonl"
        append_row(path, {"alpha": 1, "beta": "two"})
        content = path.read_text(encoding="utf-8")
        self.assertEqual(content, '{"alpha":1,"beta":"two"}\n')

    def test_append_preserves_unicode(self):
        path = self.dir / "unicode.jsonl"
        append_row(path, {"name": "café", "emoji": "🚀"})
        self.assertEqual(
            list(read_rows(path)),
            [{"name": "café", "emoji": "🚀"}],
        )

    def test_append_to_existing_file_resumes(self):
        """Append to a file with prior rows; old rows preserved, new appended."""
        path = self.dir / "resume.jsonl"
        append_row(path, {"i": 0})
        append_row(path, {"i": 1})
        # Simulate restart: new "process" appends two more.
        append_row(path, {"i": 2})
        append_row(path, {"i": 3})
        self.assertEqual(
            list(read_rows(path)),
            [{"i": 0}, {"i": 1}, {"i": 2}, {"i": 3}],
        )

    # ----- read edge cases -----

    def test_read_missing_file_is_empty(self):
        self.assertEqual(list(read_rows(self.dir / "absent.jsonl")), [])

    def test_read_empty_file_is_empty(self):
        path = self.dir / "empty.jsonl"
        path.touch()
        self.assertEqual(list(read_rows(path)), [])

    def test_read_skips_blank_and_whitespace_lines(self):
        path = self.dir / "blank.jsonl"
        path.write_text(
            '{"a":1}\n'
            '\n'
            '   \n'
            '{"b":2}\n'
            '\n',
            encoding="utf-8",
        )
        self.assertEqual(list(read_rows(path)), [{"a": 1}, {"b": 2}])

    def test_read_tolerates_malformed_row_at_end(self):
        """Crashed-write leaves a partial last line; reader skips it."""
        path = self.dir / "crashed.jsonl"
        path.write_text(
            '{"a":1}\n'
            '{"b":2}\n'
            '{"partial":',  # crashed mid-write
            encoding="utf-8",
        )
        # Capture stderr to verify warning emitted.
        buf = io.StringIO()
        old_stderr = sys.stderr
        try:
            sys.stderr = buf
            rows = list(read_rows(path))
        finally:
            sys.stderr = old_stderr
        self.assertEqual(rows, [{"a": 1}, {"b": 2}])
        self.assertIn("skipping malformed row", buf.getvalue())
        self.assertIn(":3:", buf.getvalue())   # line 3 was the bad one

    def test_read_tolerates_malformed_row_in_middle(self):
        """Even a malformed line in the middle doesn't drop later good rows."""
        path = self.dir / "middle.jsonl"
        path.write_text(
            '{"a":1}\n'
            'not json at all\n'
            '{"b":2}\n',
            encoding="utf-8",
        )
        buf = io.StringIO()
        old_stderr = sys.stderr
        try:
            sys.stderr = buf
            rows = list(read_rows(path))
        finally:
            sys.stderr = old_stderr
        self.assertEqual(rows, [{"a": 1}, {"b": 2}])

    # ----- build_skip_set -----

    def test_build_skip_set_no_filter_includes_all_rows(self):
        path = self.dir / "all.jsonl"
        for i in range(5):
            append_row(path, {"id": i})
        skip = build_skip_set(path, key_fn=lambda r: r["id"])
        self.assertEqual(skip, {0, 1, 2, 3, 4})

    def test_build_skip_set_status_filter_skips_permanent_only(self):
        """Mirrors the kernel-tune resume policy: SUCCESS / FAILED_OOM /
        SKIPPED are permanent; UNKNOWN_ERROR retried."""
        path = self.dir / "mixed.jsonl"
        append_row(path, {"id": 1, "status": "SUCCESS"})
        append_row(path, {"id": 2, "status": "UNKNOWN_ERROR"})
        append_row(path, {"id": 3, "status": "FAILED_OOM"})
        append_row(path, {"id": 4, "status": "SUCCESS"})
        append_row(path, {"id": 5, "status": "SKIPPED"})
        append_row(path, {"id": 6, "status": "UNKNOWN_ERROR"})

        skip = build_skip_set(
            path,
            key_fn=lambda r: r["id"],
            status_filter={"SUCCESS", "FAILED_OOM", "SKIPPED"},
        )
        # Permanent statuses skipped; UNKNOWN_ERROR rows (id 2, 6) absent.
        self.assertEqual(skip, {1, 3, 4, 5})

    def test_build_skip_set_with_tuple_keys(self):
        """key_fn can return any hashable — tuple is the common case for
        compound (tuning_key, tunable_params)."""
        path = self.dir / "tuples.jsonl"
        append_row(path, {"page": 128, "K": 256, "v": 1})
        append_row(path, {"page": 128, "K": 512, "v": 2})
        append_row(path, {"page": 64, "K": 256, "v": 3})
        skip = build_skip_set(
            path,
            key_fn=lambda r: (r["page"], r["K"]),
        )
        self.assertEqual(skip, {(128, 256), (128, 512), (64, 256)})

    def test_build_skip_set_missing_file_returns_empty(self):
        self.assertEqual(
            build_skip_set(self.dir / "missing.jsonl",
                           key_fn=lambda r: r["id"]),
            set(),
        )

    def test_build_skip_set_empty_status_filter_set_skips_everything(self):
        """An EMPTY (but non-None) status_filter means "no row matches" —
        every row is treated as retryable. Useful for "re-tune everything"."""
        path = self.dir / "all_retried.jsonl"
        append_row(path, {"id": 1, "status": "SUCCESS"})
        append_row(path, {"id": 2, "status": "SUCCESS"})
        skip = build_skip_set(
            path,
            key_fn=lambda r: r["id"],
            status_filter=set(),
        )
        self.assertEqual(skip, set())

    def test_build_skip_set_handles_missing_status_field(self):
        """A row without a 'status' field is dropped under any non-empty
        status_filter (it doesn't claim permanence)."""
        path = self.dir / "no_status.jsonl"
        append_row(path, {"id": 1})        # no status field
        append_row(path, {"id": 2, "status": "SUCCESS"})
        skip = build_skip_set(
            path,
            key_fn=lambda r: r["id"],
            status_filter={"SUCCESS"},
        )
        # Row 1 has no status, falls through filter, NOT added.
        self.assertEqual(skip, {2})


if __name__ == "__main__":
    unittest.main()
