# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for tools.benchmark.compare.

Coverage target: 100% lines + branches of compare.py except the
`if __name__ == \"__main__\": sys.exit(main())` shim.
"""

import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

from tools.benchmark import compare


def _make_combo_dir(parent: Path, cid: str,
                    metrics: dict | None,
                    meta: dict | None) -> Path:
    """Create <parent>/<cid>/ with optional metrics.txt and meta.txt."""
    d = parent / cid
    d.mkdir(parents=True)
    if metrics is not None:
        d.joinpath("metrics.txt").write_text(
            "".join(f"{k}={v}\n" for k, v in metrics.items()))
    if meta is not None:
        d.joinpath("meta.txt").write_text(
            "".join(f"{k}={v}\n" for k, v in meta.items()))
    return d


class TestParseKvFile(unittest.TestCase):

    def test_basic(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt",
                                         delete=False) as f:
            f.write("a=1\nb=2\nc=3\n")
            p = f.name
        try:
            self.assertEqual(compare.parse_kv_file(p),
                             {"a": "1", "b": "2", "c": "3"})
        finally:
            os.unlink(p)

    def test_skips_blank_and_malformed(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt",
                                         delete=False) as f:
            f.write("a=1\n\nno_equals_here\nb=\n")
            p = f.name
        try:
            self.assertEqual(compare.parse_kv_file(p),
                             {"a": "1", "b": ""})
        finally:
            os.unlink(p)

    def test_value_with_equals(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt",
                                         delete=False) as f:
            f.write("a=key=val\n")
            p = f.name
        try:
            self.assertEqual(compare.parse_kv_file(p), {"a": "key=val"})
        finally:
            os.unlink(p)

    def test_handles_crlf_endings(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                         newline="") as f:
            f.write("a=1\r\nb=2\r\n")
            p = f.name
        try:
            self.assertEqual(compare.parse_kv_file(p),
                             {"a": "1", "b": "2"})
        finally:
            os.unlink(p)


class TestCollectResults(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_empty_dir_yields_nothing(self):
        self.assertEqual(compare.collect_results(self.dir), [])

    def test_nonexistent_dir_yields_nothing(self):
        self.assertEqual(compare.collect_results(self.dir / "nope"), [])

    def test_collects_combos_with_metrics(self):
        _make_combo_dir(self.dir, "abc",
                        metrics={"RequestThroughput": "1.2"},
                        meta={"K": "128"})
        _make_combo_dir(self.dir, "def",
                        metrics={"RequestThroughput": "3.4"},
                        meta={"K": "256"})
        results = compare.collect_results(self.dir)
        self.assertEqual(len(results), 2)
        ids = {r["combo_id"] for r in results}
        self.assertEqual(ids, {"abc", "def"})

    def test_skips_combos_without_metrics(self):
        # No metrics.txt -> not yet completed -> skipped.
        _make_combo_dir(self.dir, "incomplete",
                        metrics=None, meta={"K": "128"})
        _make_combo_dir(self.dir, "done",
                        metrics={"RequestThroughput": "1.0"},
                        meta={"K": "128"})
        results = compare.collect_results(self.dir)
        self.assertEqual([r["combo_id"] for r in results], ["done"])

    def test_tolerates_missing_meta(self):
        _make_combo_dir(self.dir, "no_meta",
                        metrics={"RequestThroughput": "1.0"}, meta=None)
        results = compare.collect_results(self.dir)
        self.assertEqual(results[0]["meta"], {})

    def test_skips_files_at_top_level(self):
        # A stray file at sweep_dir top level should not be parsed as a combo.
        (self.dir / "stray.txt").write_text("hello")
        _make_combo_dir(self.dir, "real",
                        metrics={"RequestThroughput": "1.0"}, meta={})
        results = compare.collect_results(self.dir)
        self.assertEqual([r["combo_id"] for r in results], ["real"])


class TestToFloat(unittest.TestCase):

    def test_numeric_strings(self):
        self.assertEqual(compare._to_float("3.14"), 3.14)
        self.assertEqual(compare._to_float("42"), 42.0)
        self.assertEqual(compare._to_float("-1"), -1.0)

    def test_none_and_empty(self):
        self.assertIsNone(compare._to_float(None))
        self.assertIsNone(compare._to_float(""))

    def test_non_numeric(self):
        self.assertIsNone(compare._to_float("not a number"))

    def test_unsupported_type(self):
        # `float([1, 2])` raises TypeError -> caught -> None.
        self.assertIsNone(compare._to_float([1, 2]))

    def test_nan_returns_none(self):
        # NaN breaks sorted()'s ordering contract; treat as missing.
        self.assertIsNone(compare._to_float("nan"))
        self.assertIsNone(compare._to_float("NaN"))

    def test_inf_returns_none(self):
        # ±inf is well-ordered but more likely a bench bug than a real
        # datapoint. Reject so it doesn't silently sort to top.
        self.assertIsNone(compare._to_float("inf"))
        self.assertIsNone(compare._to_float("-inf"))
        self.assertIsNone(compare._to_float("Infinity"))


class TestRankBy(unittest.TestCase):

    def _r(self, mid, val):
        return {"combo_id": mid, "result_dir": Path("/x"),
                "metrics": {"RequestThroughput": val}, "meta": {}}

    def test_sorts_descending_by_default(self):
        results = [self._r("a", "1"), self._r("b", "3"), self._r("c", "2")]
        out = compare.rank_by(results, "RequestThroughput")
        self.assertEqual([r["combo_id"] for r in out], ["b", "c", "a"])

    def test_ascending(self):
        results = [self._r("a", "1"), self._r("b", "3"), self._r("c", "2")]
        out = compare.rank_by(results, "RequestThroughput", descending=False)
        self.assertEqual([r["combo_id"] for r in out], ["a", "c", "b"])

    def test_missing_values_go_last(self):
        results = [self._r("a", ""), self._r("b", "1.0"), self._r("c", "2.0")]
        out = compare.rank_by(results, "RequestThroughput")
        # "a" missing -> end. "c" > "b".
        self.assertEqual([r["combo_id"] for r in out], ["c", "b", "a"])

    def test_missing_values_ascending_still_last(self):
        results = [self._r("a", ""), self._r("b", "5"), self._r("c", "1")]
        out = compare.rank_by(results, "RequestThroughput", descending=False)
        self.assertEqual([r["combo_id"] for r in out], ["c", "b", "a"])

    def test_missing_metric_key(self):
        # No 'RequestThroughput' key at all -> missing -> end.
        results = [
            {"combo_id": "a", "metrics": {}, "meta": {}, "result_dir": Path("/")},
            self._r("b", "9"),
        ]
        out = compare.rank_by(results, "RequestThroughput")
        self.assertEqual([r["combo_id"] for r in out], ["b", "a"])

    def test_nan_value_sorts_as_missing(self):
        # Regression: a NaN value used to destabilize sort because all
        # NaN comparisons return False. Now treated as missing -> last.
        results = [self._r("a", "nan"), self._r("b", "1.0"),
                   self._r("c", "2.0")]
        out = compare.rank_by(results, "RequestThroughput")
        self.assertEqual([r["combo_id"] for r in out], ["c", "b", "a"])


class TestBestPerAxis(unittest.TestCase):

    def _r(self, mid, k, val):
        return {"combo_id": mid, "result_dir": Path("/x"),
                "metrics": {"RequestThroughput": val},
                "meta": {"K": k}}

    def test_picks_best_within_each_group(self):
        results = [
            self._r("a", "128", "5"),
            self._r("b", "128", "8"),  # winner for K=128
            self._r("c", "256", "9"),  # winner for K=256
            self._r("d", "256", "7"),
        ]
        best = compare.best_per_axis(results, "K", "RequestThroughput")
        self.assertEqual(best["128"]["combo_id"], "b")
        self.assertEqual(best["256"]["combo_id"], "c")

    def test_drops_combos_missing_axis_key(self):
        results = [self._r("a", "128", "5")]
        results.append({"combo_id": "b", "metrics": {"RequestThroughput": "9"},
                        "meta": {}, "result_dir": Path("/")})
        best = compare.best_per_axis(results, "K", "RequestThroughput")
        self.assertEqual(set(best.keys()), {"128"})

    def test_drops_axis_value_empty(self):
        results = [
            self._r("a", "", "5"),
            self._r("b", "128", "9"),
        ]
        best = compare.best_per_axis(results, "K", "RequestThroughput")
        self.assertEqual(set(best.keys()), {"128"})

    def test_drops_group_with_no_numeric_metric(self):
        results = [
            self._r("a", "128", ""),
            self._r("b", "128", "not a number"),
        ]
        best = compare.best_per_axis(results, "K", "RequestThroughput")
        self.assertEqual(best, {})

    def test_ascending(self):
        results = [self._r("a", "128", "5"), self._r("b", "128", "8")]
        best = compare.best_per_axis(results, "K", "RequestThroughput",
                                     descending=False)
        self.assertEqual(best["128"]["combo_id"], "a")  # smallest wins


class TestExtractField(unittest.TestCase):

    def test_combo_id(self):
        r = {"combo_id": "abc", "result_dir": Path("/"),
             "metrics": {}, "meta": {}}
        self.assertEqual(compare._extract_field(r, "combo_id"), "abc")

    def test_result_dir(self):
        r = {"combo_id": "abc", "result_dir": Path("/x"),
             "metrics": {}, "meta": {}}
        self.assertEqual(compare._extract_field(r, "result_dir"), "/x")

    def test_dotted(self):
        r = {"combo_id": "abc", "result_dir": Path("/"),
             "metrics": {"RequestThroughput": "1.2"},
             "meta": {"K": "128"}}
        self.assertEqual(compare._extract_field(r, "metrics.RequestThroughput"),
                         "1.2")
        self.assertEqual(compare._extract_field(r, "meta.K"), "128")

    def test_missing_section_or_field(self):
        r = {"combo_id": "abc", "result_dir": Path("/"),
             "metrics": {}, "meta": {}}
        self.assertEqual(compare._extract_field(r, "metrics.NoSuch"), "")
        self.assertEqual(compare._extract_field(r, "missing.NoSuch"), "")

    def test_top_level_key(self):
        r = {"combo_id": "abc", "extra": "something",
             "result_dir": Path("/"), "metrics": {}, "meta": {}}
        self.assertEqual(compare._extract_field(r, "extra"), "something")
        self.assertEqual(compare._extract_field(r, "missing"), "")

    def test_pipe_in_value_is_escaped(self):
        # A pipe in cell content would break Markdown layout. Escape it.
        r = {"combo_id": "x|y", "result_dir": Path("/"),
             "metrics": {"k": "a|b"}, "meta": {"flag": "--foo a|b"}}
        self.assertEqual(compare._extract_field(r, "combo_id"), r"x\|y")
        self.assertEqual(compare._extract_field(r, "metrics.k"), r"a\|b")
        self.assertEqual(compare._extract_field(r, "meta.flag"),
                         r"--foo a\|b")


class TestFormatMarkdownTable(unittest.TestCase):

    def test_empty(self):
        self.assertEqual(
            compare.format_markdown_table([], compare.DEFAULT_COLUMNS),
            "(no results)")

    def test_basic(self):
        results = [{
            "combo_id": "abc",
            "result_dir": Path("/x"),
            "metrics": {"RequestThroughput": "1.2"},
            "meta": {"max_num_seqs": "128"},
        }]
        cols = [("combo_id", "id"),
                ("metrics.RequestThroughput", "req/s"),
                ("meta.max_num_seqs", "MNS")]
        out = compare.format_markdown_table(results, cols)
        # Should have header, separator, and one data row.
        lines = out.splitlines()
        self.assertEqual(len(lines), 3)
        self.assertIn("id", lines[0])
        self.assertIn("req/s", lines[0])
        self.assertIn("---", lines[1])
        self.assertIn("abc", lines[2])
        self.assertIn("1.2", lines[2])
        self.assertIn("128", lines[2])

    def test_columns_widen_to_longest(self):
        # Long value forces column width to expand.
        results = [{
            "combo_id": "shortid",
            "result_dir": Path("/"),
            "metrics": {"RequestThroughput": "12345.67890"},
            "meta": {},
        }]
        cols = [("combo_id", "id"),
                ("metrics.RequestThroughput", "x")]
        out = compare.format_markdown_table(results, cols)
        for line in out.splitlines():
            # Same width across rows.
            self.assertEqual(line.count("|"), 3)


class TestColumnsFromArg(unittest.TestCase):

    def test_default_when_empty(self):
        self.assertEqual(compare._columns_from_arg(None),
                         compare.DEFAULT_COLUMNS)
        self.assertEqual(compare._columns_from_arg(""),
                         compare.DEFAULT_COLUMNS)

    def test_parses_csv(self):
        result = compare._columns_from_arg(
            "combo_id,metrics.RequestThroughput,meta.K")
        self.assertEqual(result, [
            ("combo_id", "combo_id"),
            ("metrics.RequestThroughput", "RequestThroughput"),
            ("meta.K", "K"),
        ])

    def test_strips_whitespace_and_blanks(self):
        result = compare._columns_from_arg(" combo_id , , meta.K ")
        self.assertEqual(result, [
            ("combo_id", "combo_id"),
            ("meta.K", "K"),
        ])


class TestSortAxisKeys(unittest.TestCase):

    def test_numeric_sorts_ascending(self):
        self.assertEqual(compare._sort_axis_keys(["1024", "128", "256"]),
                         ["128", "256", "1024"])

    def test_strings_after_numerics(self):
        self.assertEqual(
            compare._sort_axis_keys(["zoo", "128", "abc", "1024"]),
            ["128", "1024", "abc", "zoo"])


class TestMainCli(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_empty_dir_returns_one(self):
        buf_err = io.StringIO()
        with redirect_stderr(buf_err):
            rc = compare.main([str(self.dir)])
        self.assertEqual(rc, 1)
        self.assertIn("No results", buf_err.getvalue())

    def test_basic_table(self):
        _make_combo_dir(self.dir, "abc",
                        metrics={"RequestThroughput": "1.2",
                                 "MeanTTFT": "100"},
                        meta={"K": "128", "case_name": "x"})
        _make_combo_dir(self.dir, "def",
                        metrics={"RequestThroughput": "3.4",
                                 "MeanTTFT": "120"},
                        meta={"K": "256", "case_name": "x"})
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = compare.main([str(self.dir),
                               "--columns", "combo_id,meta.K,metrics.RequestThroughput"])
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        self.assertIn("abc", out)
        self.assertIn("def", out)
        # def has higher throughput, should be first row in body.
        body = out.splitlines()[2:]
        self.assertIn("def", body[0])

    def test_ascending(self):
        _make_combo_dir(self.dir, "abc",
                        metrics={"RequestThroughput": "1.2"}, meta={})
        _make_combo_dir(self.dir, "def",
                        metrics={"RequestThroughput": "3.4"}, meta={})
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = compare.main([str(self.dir), "--ascending",
                               "--columns", "combo_id,metrics.RequestThroughput"])
        self.assertEqual(rc, 0)
        body = buf.getvalue().splitlines()[2:]
        self.assertIn("abc", body[0])  # 1.2 first when ascending

    def test_best_per(self):
        _make_combo_dir(self.dir, "k128_a",
                        metrics={"RequestThroughput": "1.0"},
                        meta={"K": "128"})
        _make_combo_dir(self.dir, "k128_b",
                        metrics={"RequestThroughput": "2.0"},
                        meta={"K": "128"})
        _make_combo_dir(self.dir, "k256_a",
                        metrics={"RequestThroughput": "5.0"},
                        meta={"K": "256"})
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = compare.main([str(self.dir),
                               "--best-per", "K",
                               "--columns", "combo_id,meta.K,metrics.RequestThroughput"])
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        # k128_b is winner for K=128 (2.0 > 1.0); k256_a for K=256.
        self.assertIn("k128_b", out)
        self.assertNotIn("k128_a", out)
        self.assertIn("k256_a", out)
        # K=128 should appear before K=256 (numeric ascending sort of axis values).
        i_128 = out.index("128")
        i_256 = out.index("256")
        self.assertLess(i_128, i_256)


if __name__ == "__main__":
    unittest.main()
