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
"""Unit tests for tools.benchmark.parse_bench_log.

Coverage target: 100% lines + branches of parse_bench_log.py.
"""

import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from tools.benchmark import parse_bench_log as pbl

# Realistic snippet of vllm bench serve output. Numbers are arbitrary;
# what matters is the line shapes match the real format.
SAMPLE_LOG = """\
============ Serving Benchmark Result ============
Successful requests:                     1000
Benchmark duration (s):                  120.50
Total input tokens:                      1024000
Total generated tokens:                  128000
Request throughput (req/s):              8.30
Output token throughput (tok/s):         1062.40
Total Token throughput (tok/s):          9559.30
---------------Time to First Token----------------
Mean TTFT (ms):                          250.10
Median TTFT (ms):                        220.00
P99 TTFT (ms):                           1100.50
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          15.60
Median TPOT (ms):                        15.00
P99 TPOT (ms):                           45.20
---------------Inter-token Latency----------------
Mean ITL (ms):                           14.80
Median ITL (ms):                         14.50
P99 ITL (ms):                            41.10
---------------End-to-end Latency-----------------
Mean E2EL (ms):                          2200.00
Median E2EL (ms):                        2050.00
P99 E2EL (ms):                           4500.30
==================================================
"""


class TestLastToken(unittest.TestCase):

    def test_extracts_last_token(self):
        self.assertEqual(pbl._last_token("Mean TTFT (ms): 12.3"), "12.3")

    def test_strips_trailing_whitespace(self):
        self.assertEqual(pbl._last_token("foo bar 99   \n"), "99")

    def test_empty_line(self):
        self.assertEqual(pbl._last_token(""), "")

    def test_whitespace_only(self):
        self.assertEqual(pbl._last_token("   \t  "), "")

    def test_returns_non_numeric_token_verbatim(self):
        # Documents the helper's honest contract: it returns the last
        # token, period — numeric or not. Callers validate (or trust).
        self.assertEqual(pbl._last_token("foo bar baz"), "baz")
        self.assertEqual(pbl._last_token("99% complete"), "complete")


class TestParseLatencyMetrics(unittest.TestCase):

    def test_full_sample(self):
        result = pbl.parse_latency_metrics(SAMPLE_LOG.splitlines())
        self.assertEqual(result["MeanTTFT"], "250.10")
        self.assertEqual(result["MedianTTFT"], "220.00")
        self.assertEqual(result["P99TTFT"], "1100.50")
        self.assertEqual(result["MeanTPOT"], "15.60")
        self.assertEqual(result["P99TPOT"], "45.20")
        self.assertEqual(result["MeanITL"], "14.80")
        self.assertEqual(result["P99ITL"], "41.10")
        self.assertEqual(result["MeanE2EL"], "2200.00")
        self.assertEqual(result["P99E2EL"], "4500.30")

    def test_empty_input_returns_all_blank(self):
        result = pbl.parse_latency_metrics([])
        # Every (stat, section) key present, all empty.
        for sec in pbl.LATENCY_SECTIONS:
            for stat in pbl.LATENCY_STATS:
                self.assertEqual(result[f"{stat}{sec}"], "")

    def test_missing_section_leaves_blank(self):
        # Drop all TPOT lines; TPOT keys should stay empty.
        lines = [ln for ln in SAMPLE_LOG.splitlines() if "TPOT" not in ln]
        result = pbl.parse_latency_metrics(lines)
        self.assertEqual(result["MeanTPOT"], "")
        self.assertEqual(result["P99TPOT"], "")
        # TTFT should still be there.
        self.assertEqual(result["MeanTTFT"], "250.10")

    def test_section_header_alone_does_not_match_stat(self):
        # A bare "(Title) TTFT (ms):" header should NOT capture as a stat.
        lines = ["TTFT (ms):", "Mean TTFT (ms):  100.0"]
        result = pbl.parse_latency_metrics(lines)
        self.assertEqual(result["MeanTTFT"], "100.0")
        # P99TTFT not present in input -> empty.
        self.assertEqual(result["P99TTFT"], "")

    def test_custom_sections_and_stats(self):
        # Exercise the parameter-driven path.
        lines = ["P50 ABC (ms):  7.7", "P50 DEF (ms):  9.9"]
        result = pbl.parse_latency_metrics(
            lines, sections=("ABC", "DEF"), stats=("P50",))
        self.assertEqual(result, {"P50ABC": "7.7", "P50DEF": "9.9"})

    def test_section_marker_present_but_no_known_stat(self):
        # Line has "TTFT (ms):" but the leading word isn't one of our
        # tracked stats (Mean/Median/P99). Should leave the slot empty.
        lines = ["P95 TTFT (ms): 12.5"]
        result = pbl.parse_latency_metrics(lines)
        self.assertEqual(result["MeanTTFT"], "")
        self.assertEqual(result["MedianTTFT"], "")
        self.assertEqual(result["P99TTFT"], "")


class TestParseThroughputMetrics(unittest.TestCase):

    def test_full_sample(self):
        result = pbl.parse_throughput_metrics(SAMPLE_LOG.splitlines())
        self.assertEqual(result["RequestThroughput"], "8.30")
        self.assertEqual(result["OutputTokenThroughput"], "1062.40")
        self.assertEqual(result["TotalTokenThroughput"], "9559.30")

    def test_empty_input(self):
        result = pbl.parse_throughput_metrics([])
        self.assertEqual(result, {
            "RequestThroughput": "",
            "OutputTokenThroughput": "",
            "TotalTokenThroughput": "",
        })

    def test_only_first_match_wins(self):
        # Two lines containing the marker — first should win.
        lines = [
            "Request throughput (req/s):              5.50",
            "Request throughput (req/s):              9.90",
        ]
        result = pbl.parse_throughput_metrics(lines)
        self.assertEqual(result["RequestThroughput"], "5.50")

    def test_custom_metrics(self):
        lines = ["fooBAR  42"]
        result = pbl.parse_throughput_metrics(lines, metrics=[("foo", "fooBAR")])
        self.assertEqual(result, {"foo": "42"})

    def test_diagnostic_prefix_does_not_match(self):
        # If vllm ever logs a diagnostic with the marker mid-line, a
        # naive substring match would steal the slot. The start-anchor
        # rejects such lines.
        lines = [
            "WARNING: Request throughput (req/s): too few samples — unreliable",
            "Request throughput (req/s):              7.77",
        ]
        result = pbl.parse_throughput_metrics(lines)
        self.assertEqual(result["RequestThroughput"], "7.77")

    def test_leading_whitespace_tolerated(self):
        # Indented lines still match (lstrip before startswith).
        lines = ["   Request throughput (req/s):              3.30"]
        result = pbl.parse_throughput_metrics(lines)
        self.assertEqual(result["RequestThroughput"], "3.30")


class TestParseBenchLog(unittest.TestCase):

    def test_combines_latency_and_throughput(self):
        result = pbl.parse_bench_log(SAMPLE_LOG)
        # Spot-check one of each kind.
        self.assertEqual(result["MeanTTFT"], "250.10")
        self.assertEqual(result["RequestThroughput"], "8.30")
        # No extraneous keys beyond the documented set.
        expected_keys = (
            {f"{s}{sec}" for sec in pbl.LATENCY_SECTIONS
             for s in pbl.LATENCY_STATS}
            | {k for k, _ in pbl.THROUGHPUT_METRICS}
        )
        self.assertEqual(set(result.keys()), expected_keys)

    def test_empty_input(self):
        result = pbl.parse_bench_log("")
        self.assertTrue(all(v == "" for v in result.values()))

    def test_key_order_is_section_major_then_throughput(self):
        # Pin the documented order: for each section in (TTFT, TPOT,
        # ITL, E2EL) we emit (Mean, Median, P99), then the throughput
        # keys in declared order. Section-major (NOT stat-major).
        keys = list(pbl.parse_bench_log("").keys())
        expected_latency = [
            f"{stat}{sec}"
            for sec in pbl.LATENCY_SECTIONS
            for stat in pbl.LATENCY_STATS
        ]
        expected_throughput = [k for k, _ in pbl.THROUGHPUT_METRICS]
        self.assertEqual(keys, expected_latency + expected_throughput)
        # Sanity: first six keys are the three TTFT stats followed by
        # the three TPOT stats — section-major.
        self.assertEqual(keys[:6],
                         ["MeanTTFT", "MedianTTFT", "P99TTFT",
                          "MeanTPOT", "MedianTPOT", "P99TPOT"])


class TestFormatMetrics(unittest.TestCase):

    def test_renders_key_value_lines(self):
        out = pbl.format_metrics({"a": "1", "b": "2"})
        self.assertEqual(out, "a=1\nb=2\n")

    def test_preserves_insertion_order(self):
        out = pbl.format_metrics({"z": "9", "a": "1"})
        self.assertEqual(out, "z=9\na=1\n")

    def test_empty(self):
        self.assertEqual(pbl.format_metrics({}), "")


class TestMainCli(unittest.TestCase):

    def test_reads_file_and_writes_stdout(self):
        with tempfile.NamedTemporaryFile("w", suffix=".log",
                                         delete=False) as f:
            f.write(SAMPLE_LOG)
            tmp = f.name
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = pbl.main([tmp])
            self.assertEqual(rc, 0)
            out = buf.getvalue()
            self.assertIn("RequestThroughput=8.30", out)
            self.assertIn("MeanTTFT=250.10", out)
            self.assertIn("P99E2EL=4500.30", out)
        finally:
            os.unlink(tmp)

    def test_reads_stdin_when_dash(self):
        buf = io.StringIO()
        with patch("sys.stdin", io.StringIO(SAMPLE_LOG)):
            with redirect_stdout(buf):
                rc = pbl.main(["-"])
        self.assertEqual(rc, 0)
        self.assertIn("RequestThroughput=8.30", buf.getvalue())

    def test_argparse_requires_positional_arg(self):
        # No args -> argparse exits 2 with usage on stderr.
        with self.assertRaises(SystemExit) as cm:
            pbl.main([])
        self.assertEqual(cm.exception.code, 2)

    def test_module_main_block(self):
        # Cover the `if __name__ == "__main__"` line.
        import runpy
        with tempfile.NamedTemporaryFile("w", suffix=".log",
                                         delete=False) as f:
            f.write("Request throughput (req/s):  3.14\n")
            tmp = f.name
        try:
            buf = io.StringIO()
            with patch("sys.argv", ["parse_bench_log.py", tmp]):
                with redirect_stdout(buf):
                    with self.assertRaises(SystemExit) as cm:
                        runpy.run_module(
                            "tools.benchmark.parse_bench_log",
                            run_name="__main__")
                    self.assertEqual(cm.exception.code, 0)
            self.assertIn("RequestThroughput=3.14", buf.getvalue())
        finally:
            os.unlink(tmp)


if __name__ == "__main__":
    unittest.main()
