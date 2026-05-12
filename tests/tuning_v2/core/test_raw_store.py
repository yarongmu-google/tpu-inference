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
import threading
import unittest
from pathlib import Path
from unittest import mock

from tools.tuning.v2.core.raw_store import (
    append_row,
    build_skip_set,
    prune_raw_ttl,
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
        """Crashed-write leaves a partial last line; reader skips it
        and emits a warning via logging."""
        path = self.dir / "crashed.jsonl"
        path.write_text(
            '{"a":1}\n'
            '{"b":2}\n'
            '{"partial":',  # crashed mid-write
            encoding="utf-8",
        )
        with self.assertLogs(
            "tools.tuning.v2.core.raw_store", level="WARNING",
        ) as cap:
            rows = list(read_rows(path))
        self.assertEqual(rows, [{"a": 1}, {"b": 2}])
        joined = "\n".join(cap.output)
        self.assertIn("skipping malformed row", joined)
        self.assertIn(":3:", joined)   # line 3 was the bad one

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


class TestPruneRawTtl(unittest.TestCase):
    """fix: arch doc §8 line 241 — TTL=2 prune. Was promised but
    unimplemented; old .raw/<sha>.jsonl partitions accumulated."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _make_files(self, names: list[str]) -> list[Path]:
        """Create files in mtime order: first listed = oldest."""
        import time
        paths = []
        for n in names:
            p = self.dir / n
            p.write_text("{}\n")
            paths.append(p)
            time.sleep(0.01)   # ensure distinct mtimes
        return paths

    def test_keeps_two_most_recent(self):
        old, mid, new = self._make_files(["a.jsonl", "b.jsonl", "c.jsonl"])
        deleted = prune_raw_ttl(self.dir, keep=2)
        self.assertEqual(deleted, [old])
        self.assertFalse(old.exists())
        self.assertTrue(mid.exists())
        self.assertTrue(new.exists())

    def test_default_keep_is_two(self):
        """Per arch doc §8 — TTL=2."""
        paths = self._make_files(["a.jsonl", "b.jsonl", "c.jsonl", "d.jsonl"])
        deleted = prune_raw_ttl(self.dir)   # no explicit keep
        # Two oldest deleted.
        self.assertEqual(sorted(p.name for p in deleted),
                         ["a.jsonl", "b.jsonl"])

    def test_no_files_to_prune_returns_empty(self):
        self._make_files(["a.jsonl", "b.jsonl"])
        # Exactly 2 files; keep=2 → nothing to delete.
        self.assertEqual(prune_raw_ttl(self.dir, keep=2), [])

    def test_missing_dir_returns_empty(self):
        self.assertEqual(prune_raw_ttl(self.dir / "no", keep=2), [])

    def test_keep_zero_deletes_all(self):
        paths = self._make_files(["a.jsonl", "b.jsonl"])
        deleted = prune_raw_ttl(self.dir, keep=0)
        self.assertEqual(sorted(p.name for p in deleted),
                         ["a.jsonl", "b.jsonl"])

    def test_ignores_non_jsonl_files(self):
        self._make_files(["a.jsonl", "b.jsonl"])
        (self.dir / "readme.txt").write_text("nope")
        prune_raw_ttl(self.dir, keep=1)
        # Non-jsonl untouched.
        self.assertTrue((self.dir / "readme.txt").exists())

    def test_oserror_does_not_crash(self):
        """If unlink fails for one file (e.g. permission denied on
        a network FS), keep going for the rest. Tune shouldn't die
        because the prune step hiccupped."""
        old, _ = self._make_files(["a.jsonl", "b.jsonl"])
        # Patch unlink to raise once.
        original_unlink = Path.unlink
        def flaky_unlink(self_p, *a, **kw):
            if self_p == old:
                raise OSError("simulated EIO")
            return original_unlink(self_p, *a, **kw)
        with mock.patch("pathlib.Path.unlink", flaky_unlink), \
             self.assertLogs(
                 "tools.tuning.v2.core.raw_store", level="WARNING",
             ) as cap:
            deleted = prune_raw_ttl(self.dir, keep=0)
        # The unflaky file got deleted; the flaky one didn't, but
        # no crash and the failure logged.
        self.assertEqual(len(deleted), 1)
        self.assertIn("prune failed", "\n".join(cap.output))


class TestAtomicityGuarantees(unittest.TestCase):
    """fix #10 / #20: explicit tests for the atomicity properties the
    raw store relies on (O_APPEND + fsync per write)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_every_append_ends_in_newline(self):
        """Each row terminator must be a literal `\\n` byte so that
        even a kill mid-write of the NEXT row leaves the file
        parseable up to the last completed row."""
        path = self.dir / "newlines.jsonl"
        for i in range(50):
            append_row(path, {"i": i})
        data = path.read_bytes()
        self.assertTrue(data.endswith(b"\n"),
                        "file does not end in a newline")
        # Every line that's not the trailing empty piece must be a
        # full JSON object.
        for ln in data.splitlines():
            self.assertEqual(ln[:1] + ln[-1:], b"{}",
                             f"row not a complete object: {ln!r}")

    def test_fsync_called_once_per_append(self):
        """The crash-durability guarantee relies on os.fsync after
        each write. Verify it's actually invoked."""
        path = self.dir / "fsync.jsonl"
        with mock.patch("tools.tuning.v2.core.raw_store.os.fsync") as f:
            append_row(path, {"a": 1})
            append_row(path, {"a": 2})
            append_row(path, {"a": 3})
        self.assertEqual(f.call_count, 3)

    def test_open_uses_o_append_and_o_creat(self):
        """O_APPEND is load-bearing for the atomic-seek-then-write
        guarantee; O_CREAT lets the first append create the file.
        Verify both are in the flags."""
        import os as _os
        path = self.dir / "flags.jsonl"
        with mock.patch("tools.tuning.v2.core.raw_store.os.open",
                        wraps=_os.open) as o:
            append_row(path, {"a": 1})
        self.assertTrue(o.called)
        flags = o.call_args[0][1]
        self.assertTrue(flags & _os.O_APPEND)
        self.assertTrue(flags & _os.O_CREAT)
        self.assertTrue(flags & _os.O_WRONLY)


class TestConcurrentAppenders(unittest.TestCase):
    """fix #10 / #21: O_APPEND atomically seeks-to-EOF + writes, so
    multiple appenders to the same file must not interleave bytes
    within a single row. We don't currently use this in production
    (single tuner per .raw file), but the test enforces the safety
    property the comment in raw_store.py claims."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_threaded_appends_produce_no_interleaved_lines(self):
        path = self.dir / "concurrent.jsonl"
        n_threads = 8
        rows_per_thread = 50

        def worker(tid: int):
            for i in range(rows_per_thread):
                # Pad the payload so a non-atomic write would have an
                # obvious chance to interleave; rows have varied sizes.
                append_row(path, {
                    "tid": tid, "i": i,
                    "pad": "x" * (10 + (i % 17) * 5),
                })

        threads = [threading.Thread(target=worker, args=(t,))
                   for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        rows = list(read_rows(path))
        # All N*K rows present, every line parses cleanly.
        self.assertEqual(len(rows), n_threads * rows_per_thread)
        # Per-thread row counts match exactly (no lost / duplicated rows).
        seen: dict[int, set[int]] = {}
        for r in rows:
            seen.setdefault(r["tid"], set()).add(r["i"])
        for t in range(n_threads):
            self.assertEqual(seen[t], set(range(rows_per_thread)))


class TestSchemaForwardCompatibility(unittest.TestCase):
    """fix #10 / #23: extra unknown fields in raw rows must round-trip
    through read_rows and the skip_set without crashing. The
    projection step iterates rows verbatim, but the read layer is
    the boundary that must be permissive."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_unknown_fields_preserved_on_read(self):
        path = self.dir / "extra.jsonl"
        append_row(path, {
            "tuning_key": {"case": "logical"},
            "status": "SUCCESS",
            "latency_us": 1.0,
            "future_field_v2": {"nested": [1, 2, 3]},
        })
        rows = list(read_rows(path))
        self.assertIn("future_field_v2", rows[0])
        self.assertEqual(rows[0]["future_field_v2"]["nested"], [1, 2, 3])

    def test_skip_set_tolerates_unknown_fields(self):
        path = self.dir / "extra_skip.jsonl"
        append_row(path, {
            "tuning_key": {"case": "logical"},
            "status": "SUCCESS",
            "_future": "stuff",
        })
        skip = build_skip_set(
            path,
            key_fn=lambda r: r["tuning_key"]["case"],
            status_filter={"SUCCESS"},
        )
        self.assertEqual(skip, {"logical"})


if __name__ == "__main__":
    unittest.main()
