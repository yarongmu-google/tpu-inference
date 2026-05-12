# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for tools.benchmark.plot_ttft_cdf.

Coverage focuses on the data-handling layer (loader, percentile
computation, label/path resolution). The matplotlib plot is exercised
through `plot_cdfs` to confirm a file is produced; we don't assert on
pixel content.
"""

import json
import logging
import pathlib
import tempfile
import unittest
from unittest.mock import patch

from tools.benchmark import plot_ttft_cdf as ptc

try:
    import numpy  # noqa: F401
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

try:
    import matplotlib  # noqa: F401
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


class TestResolveDetailedJson(unittest.TestCase):
    """The accepted-inputs contract: file OR dir-containing-file."""

    def test_file_path_returned_as_is(self):
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False
        ) as tf:
            tf.write(b"{}")
            tmppath = pathlib.Path(tf.name)
        try:
            self.assertEqual(
                ptc._resolve_detailed_json(tmppath), tmppath,
            )
        finally:
            tmppath.unlink()

    def test_dir_with_detailed_json(self):
        with tempfile.TemporaryDirectory() as td:
            tdpath = pathlib.Path(td)
            target = tdpath / "detailed.json"
            target.write_text("{}")
            self.assertEqual(
                ptc._resolve_detailed_json(tdpath), target,
            )

    def test_dir_with_alternate_detailed_filename(self):
        """Tolerate users who set --result-filename foo_detailed.json."""
        with tempfile.TemporaryDirectory() as td:
            tdpath = pathlib.Path(td)
            target = tdpath / "run_a_detailed.json"
            target.write_text("{}")
            self.assertEqual(
                ptc._resolve_detailed_json(tdpath), target,
            )

    def test_dir_with_no_detailed_raises(self):
        with tempfile.TemporaryDirectory() as td:
            tdpath = pathlib.Path(td)
            (tdpath / "metrics.txt").write_text("MeanTTFT=1\n")
            with self.assertRaises(FileNotFoundError) as ctx:
                ptc._resolve_detailed_json(tdpath)
            self.assertIn("--save-detailed", str(ctx.exception))

    def test_nonexistent_path_raises(self):
        with self.assertRaises(FileNotFoundError):
            ptc._resolve_detailed_json(
                pathlib.Path("/nonexistent/path/xyz")
            )


class TestLoadTtftsSeconds(unittest.TestCase):
    """The JSON-shape contract: ttfts is a list[float] in seconds."""

    def _write_detailed(self, td, ttfts):
        path = pathlib.Path(td) / "detailed.json"
        path.write_text(json.dumps({"ttfts": ttfts}))
        return path

    def test_returns_list_of_floats(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._write_detailed(td, [0.1, 0.5, 1.0, 27.3])
            out = ptc.load_ttfts_seconds(path)
            self.assertEqual(out, [0.1, 0.5, 1.0, 27.3])

    def test_accepts_dir_input(self):
        with tempfile.TemporaryDirectory() as td:
            self._write_detailed(td, [0.1, 0.5])
            out = ptc.load_ttfts_seconds(pathlib.Path(td))
            self.assertEqual(out, [0.1, 0.5])

    def test_missing_ttfts_key_raises(self):
        """If --save-detailed wasn't passed, the JSON has only
        summary aggregates and no per-request list. The error message
        must point the user at the fix."""
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "detailed.json"
            path.write_text(json.dumps({"mean_ttft_ms": 1000.0}))
            with self.assertRaises(ValueError) as ctx:
                ptc.load_ttfts_seconds(path)
            self.assertIn("--save-detailed", str(ctx.exception))

    def test_empty_ttfts_list_raises(self):
        """An empty list is almost certainly a bench failure (no
        streaming, no completions); we should error rather than make
        a meaningless plot."""
        with tempfile.TemporaryDirectory() as td:
            path = self._write_detailed(td, [])
            with self.assertRaises(ValueError):
                ptc.load_ttfts_seconds(path)


@unittest.skipUnless(_HAS_NUMPY, "numpy not installed in this env")
class TestPercentileTable(unittest.TestCase):

    def test_returns_six_canonical_percentiles_in_order(self):
        ttfts = [float(i) for i in range(1, 101)]
        table = ptc.percentile_table(ttfts)
        self.assertEqual(
            [p for p, _ in table], [10, 25, 50, 75, 90, 99],
        )

    def test_p50_is_median_for_uniform_input(self):
        ttfts = [float(i) for i in range(1, 101)]
        table = dict(ptc.percentile_table(ttfts))
        # numpy's default linear interpolation gives 50.5 for 1..100.
        self.assertAlmostEqual(table[50], 50.5, places=5)

    def test_handles_single_value(self):
        table = ptc.percentile_table([42.0])
        for _, v in table:
            self.assertEqual(v, 42.0)


class TestDefaultLabel(unittest.TestCase):

    def test_file_uses_parent_dir_name(self):
        """detailed.json under bench_*/<combo_sha>/ should label
        as the combo_sha, not 'detailed' — the dir name carries the
        identifying signal."""
        with tempfile.TemporaryDirectory() as td:
            combo_dir = pathlib.Path(td) / "combo_abc123"
            combo_dir.mkdir()
            target = combo_dir / "detailed.json"
            target.write_text("{}")
            self.assertEqual(ptc._default_label(target), "combo_abc123")

    def test_dir_uses_dir_name(self):
        with tempfile.TemporaryDirectory() as td:
            dpath = pathlib.Path(td) / "my_bench_dir"
            dpath.mkdir()
            self.assertEqual(ptc._default_label(dpath), "my_bench_dir")


@unittest.skipUnless(
    _HAS_NUMPY and _HAS_MPL,
    "matplotlib/numpy not installed in this env",
)
class TestPlotCdfs(unittest.TestCase):
    """End-to-end: produces a file at the expected path. We don't
    assert on pixels — only on side effects we'll rely on
    downstream (file exists, percentiles logged)."""

    def _make_bench_dir(self, parent, name, ttfts):
        d = parent / name
        d.mkdir()
        (d / "detailed.json").write_text(json.dumps({"ttfts": ttfts}))
        return d

    def test_writes_png_with_two_curves(self):
        with tempfile.TemporaryDirectory() as td:
            tdpath = pathlib.Path(td)
            a = self._make_bench_dir(tdpath, "run_a", [0.5, 1.0, 2.0])
            b = self._make_bench_dir(tdpath, "run_b", [5.0, 10.0, 30.0])
            out = tdpath / "out.png"
            ptc.plot_cdfs(
                inputs=[a, b],
                out_path=out,
                labels=None,
                title="test",
                log_x=True,
            )
            self.assertTrue(out.is_file())
            self.assertGreater(out.stat().st_size, 0)

    def test_percentile_table_emitted_to_log(self):
        with tempfile.TemporaryDirectory() as td:
            tdpath = pathlib.Path(td)
            a = self._make_bench_dir(
                tdpath, "run_a", [float(i) for i in range(1, 101)]
            )
            out = tdpath / "out.png"
            with self.assertLogs(ptc.logger, level=logging.INFO) as cm:
                ptc.plot_cdfs(
                    inputs=[a],
                    out_path=out,
                    labels=None,
                    title="test",
                    log_x=True,
                )
            joined = "\n".join(cm.output)
            self.assertIn("P10", joined)
            self.assertIn("P50", joined)
            self.assertIn("P99", joined)

    def test_creates_parent_dir_if_missing(self):
        """Users will sometimes point --out at a nested path that
        doesn't exist yet; we should mkdir -p, not blow up."""
        with tempfile.TemporaryDirectory() as td:
            tdpath = pathlib.Path(td)
            a = self._make_bench_dir(tdpath, "run_a", [1.0, 2.0])
            out = tdpath / "nested" / "subdir" / "x.png"
            ptc.plot_cdfs(
                inputs=[a],
                out_path=out,
                labels=None,
                title="t",
                log_x=False,
            )
            self.assertTrue(out.is_file())


@unittest.skipUnless(
    _HAS_NUMPY and _HAS_MPL,
    "matplotlib/numpy not installed in this env",
)
class TestMain(unittest.TestCase):
    """argparse layer — validation and dispatch."""

    def test_labels_count_mismatch_errors(self):
        """Mismatched --labels would silently mislabel curves; we
        want the user told upfront."""
        with tempfile.TemporaryDirectory() as td:
            tdpath = pathlib.Path(td)
            a = tdpath / "a"
            a.mkdir()
            (a / "detailed.json").write_text(
                json.dumps({"ttfts": [1.0]})
            )
            b = tdpath / "b"
            b.mkdir()
            (b / "detailed.json").write_text(
                json.dumps({"ttfts": [2.0]})
            )
            with self.assertRaises(SystemExit):
                ptc.main([str(a), str(b), "--labels", "only-one"])

    def test_returns_zero_on_success(self):
        with tempfile.TemporaryDirectory() as td:
            tdpath = pathlib.Path(td)
            a = tdpath / "a"
            a.mkdir()
            (a / "detailed.json").write_text(
                json.dumps({"ttfts": [0.1, 0.2, 0.3]})
            )
            out = tdpath / "x.png"
            rc = ptc.main([str(a), "--out", str(out)])
            self.assertEqual(rc, 0)
            self.assertTrue(out.is_file())


if __name__ == "__main__":
    unittest.main()
