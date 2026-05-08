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
"""Unit tests for the production .service registry exporter.

Coverage emphasises the data-correctness invariants:

  - Workload-key includes (kernel, service, model, TP, in, out) so
    distinct kernels/services/models do NOT silently overwrite each
    other.
  - Existing-file load failures are loud, not silent (a corrupt prior
    file must not nuke history).
  - Writes are atomic via os.replace from a sibling tmp file.
  - 0.0 throughput is treated as a real value, not as "missing".
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

from tools.benchmark.build_service_registry import (
    _make_workload_key, _normalize_metric, _to_float,
    export_production_registry, rank_by,
)


def _result(model="m", tp="1", in_len="2048", out_len="2048",
            max_num_seqs="128", max_num_batched_tokens="10000",
            throughput="10.5", ttft=None):
    metrics: dict[str, str] = {"RequestThroughput": throughput}
    if ttft is not None:
        metrics["MeanTTFT"] = ttft
    return {
        "meta": {
            "model": model,
            "tensor_parallel_size": tp,
            "input_len": in_len,
            "output_len": out_len,
            "max_num_seqs": max_num_seqs,
            "max_num_batched_tokens": max_num_batched_tokens,
            "max_model_len": "8192",
            "block_size": "128",
        },
        "metrics": metrics,
    }


class TestWorkloadKey(unittest.TestCase):

    def test_distinct_kernels_get_distinct_keys(self):
        # Same workload shape, different kernel — must NOT collide.
        meta = _result()["meta"]
        k1 = _make_workload_key(meta, kernel_id="rpa_v3", service_id="vllm")
        k2 = _make_workload_key(meta, kernel_id="rpa_v4", service_id="vllm")
        self.assertNotEqual(k1, k2)

    def test_distinct_services_get_distinct_keys(self):
        meta = _result()["meta"]
        k1 = _make_workload_key(meta, kernel_id="rpa_v3", service_id="vllm")
        k2 = _make_workload_key(meta, kernel_id="rpa_v3", service_id="sglang")
        self.assertNotEqual(k1, k2)

    def test_distinct_models_get_distinct_keys(self):
        m1 = _result(model="llama_3_8b")["meta"]
        m2 = _result(model="qwen_3_coder_30b")["meta"]
        k1 = _make_workload_key(m1, "rpa_v3", "vllm")
        k2 = _make_workload_key(m2, "rpa_v3", "vllm")
        self.assertNotEqual(k1, k2)

    def test_distinct_tp_sizes_get_distinct_keys(self):
        m1 = _result(tp="1")["meta"]
        m2 = _result(tp="4")["meta"]
        k1 = _make_workload_key(m1, "rpa_v3", "vllm")
        k2 = _make_workload_key(m2, "rpa_v3", "vllm")
        self.assertNotEqual(k1, k2)


class TestExportProductionRegistry(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)
        self.out_path = self.tmp_path / "production.service"

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_export_creates_file_with_namespaced_workload_key(self):
        export_production_registry(
            [_result()], self.out_path,
            kernel_id="rpa_v3", service_id="vllm")
        with open(self.out_path) as f:
            data = json.load(f)
        keys = list(data["best_configs_by_workload"].keys())
        self.assertEqual(len(keys), 1)
        self.assertIn("rpa_v3", keys[0])
        self.assertIn("vllm", keys[0])
        self.assertIn("2048_in_2048_out", keys[0])

    def test_distinct_kernels_coexist_in_same_file(self):
        export_production_registry(
            [_result()], self.out_path,
            kernel_id="rpa_v3", service_id="vllm")
        export_production_registry(
            [_result(throughput="20.0")], self.out_path,
            kernel_id="rpa_v4", service_id="vllm")
        with open(self.out_path) as f:
            data = json.load(f)
        # Two entries — second did NOT overwrite the first.
        self.assertEqual(len(data["best_configs_by_workload"]), 2)

    def test_overwrites_only_if_strictly_better(self):
        export_production_registry(
            [_result(throughput="20.0")], self.out_path,
            kernel_id="rpa_v3", service_id="vllm")
        # Worse result — must NOT overwrite.
        export_production_registry(
            [_result(throughput="15.0", max_num_seqs="999")],
            self.out_path,
            kernel_id="rpa_v3", service_id="vllm")
        with open(self.out_path) as f:
            data = json.load(f)
        entry = list(data["best_configs_by_workload"].values())[0]
        self.assertEqual(entry["MAX_NUM_SEQS"], "128")  # original kept
        # Strictly-better result — must overwrite.
        export_production_registry(
            [_result(throughput="25.0", max_num_seqs="256")],
            self.out_path,
            kernel_id="rpa_v3", service_id="vllm")
        with open(self.out_path) as f:
            data = json.load(f)
        entry = list(data["best_configs_by_workload"].values())[0]
        self.assertEqual(entry["MAX_NUM_SEQS"], "256")

    def test_zero_throughput_is_a_real_value_not_missing(self):
        # Pre-seed with a 0.0 throughput entry. A new 0.0 must NOT
        # overwrite (not strictly better). Earlier code's
        # `_to_float() or 0.0` masked 0.0 as missing and over-eagerly
        # accepted any positive over a true 0.0 baseline.
        export_production_registry(
            [_result(throughput="0.0", max_num_seqs="ALPHA")], self.out_path,
            kernel_id="rpa_v3", service_id="vllm")
        export_production_registry(
            [_result(throughput="0.0", max_num_seqs="BETA")], self.out_path,
            kernel_id="rpa_v3", service_id="vllm")
        with open(self.out_path) as f:
            data = json.load(f)
        entry = list(data["best_configs_by_workload"].values())[0]
        self.assertEqual(entry["MAX_NUM_SEQS"], "ALPHA")  # not overwritten

    def test_corrupt_existing_file_raises_not_silently_overwritten(self):
        # If the existing file is corrupt JSON, the previous behaviour
        # was `try/except: pass` — which then OVERWROTE the corrupt
        # file with a single new entry, losing whatever was salvageable.
        # Now: surface the error and refuse to overwrite.
        self.out_path.write_text("not valid json {{{")
        with self.assertRaises(json.JSONDecodeError):
            export_production_registry(
                [_result()], self.out_path,
                kernel_id="rpa_v3", service_id="vllm")
        # Original (corrupt) content untouched.
        self.assertEqual(self.out_path.read_text(), "not valid json {{{")

    def test_write_is_atomic_no_partial_files_on_crash(self):
        # Smoke-check the atomic-write path: after a successful export,
        # there should be NO leftover .tmp files in the directory.
        export_production_registry(
            [_result()], self.out_path,
            kernel_id="rpa_v3", service_id="vllm")
        leftovers = [p for p in self.tmp_path.iterdir()
                     if p.suffix == ".tmp"
                     or ".tmp" in p.name and p.name != "production.service"]
        self.assertEqual(leftovers, [])
        self.assertTrue(self.out_path.is_file())

    def test_export_picks_winner_from_unsorted_list_descending(self):
        # The export now picks the winner internally rather than
        # trusting the caller to have sorted. Pass a list in arbitrary
        # order and confirm the highest-throughput row gets exported.
        results = [
            _result(throughput="10.0", max_num_seqs="A"),
            _result(throughput="30.0", max_num_seqs="C"),
            _result(throughput="20.0", max_num_seqs="B"),
        ]
        export_production_registry(
            results, self.out_path,
            metric="RequestThroughput", descending=True,
            kernel_id="rpa_v3", service_id="vllm")
        with open(self.out_path) as f:
            data = json.load(f)
        entry = list(data["best_configs_by_workload"].values())[0]
        self.assertEqual(entry["MAX_NUM_SEQS"], "C")

    def test_export_picks_winner_from_unsorted_list_ascending(self):
        # Latency: ascending direction; smallest TTFT wins.
        results = [
            _result(ttft="100.0", max_num_seqs="A"),
            _result(ttft="10.0",  max_num_seqs="B"),
            _result(ttft="50.0",  max_num_seqs="C"),
        ]
        export_production_registry(
            results, self.out_path,
            metric="MeanTTFT", descending=False,
            kernel_id="rpa_v3", service_id="vllm_latency")
        with open(self.out_path) as f:
            data = json.load(f)
        entry = list(data["best_configs_by_workload"].values())[0]
        self.assertEqual(entry["MAX_NUM_SEQS"], "B")  # TTFT=10.0

    def test_export_accepts_qualified_metric_form(self):
        # Belt-and-suspenders: even though main() normalizes, calling
        # export directly with the qualified "metrics.X" form must not
        # silently mis-rank. This is the regression for the bug where
        # rank_by saw "metrics.MeanTTFT", failed to find it on every
        # result, and returned them in collect-order.
        results = [
            _result(ttft="100.0", max_num_seqs="A"),
            _result(ttft="10.0",  max_num_seqs="B"),
        ]
        # Bare form (canonical inside export):
        export_production_registry(
            results, self.out_path,
            metric="MeanTTFT", descending=False,
            kernel_id="rpa_v3", service_id="vllm_latency")
        with open(self.out_path) as f:
            entry_bare = list(json.load(f)["best_configs_by_workload"].values())[0]
        self.out_path.unlink()
        # Qualified form via the same _normalize_metric callers should use:
        export_production_registry(
            results, self.out_path,
            metric=_normalize_metric("metrics.MeanTTFT"), descending=False,
            kernel_id="rpa_v3", service_id="vllm_latency")
        with open(self.out_path) as f:
            entry_qual = list(json.load(f)["best_configs_by_workload"].values())[0]
        self.assertEqual(entry_bare["MAX_NUM_SEQS"], entry_qual["MAX_NUM_SEQS"])
        self.assertEqual(entry_bare["MAX_NUM_SEQS"], "B")

    def test_export_skips_when_no_parseable_metric(self):
        # All combos missing the requested metric: export must NOT
        # freeze a bogus winner into the file. Should print a WARN
        # and leave the output absent.
        results = [
            _result(ttft="100.0"),
            _result(ttft="10.0"),
        ]
        export_production_registry(
            results, self.out_path,
            metric="NoSuchMetric", descending=False,
            kernel_id="rpa_v3", service_id="vllm")
        self.assertFalse(self.out_path.exists())


class TestRankByPrefixHandling(unittest.TestCase):
    """Regression coverage for the rank_by silent-wrong-winner bug.

    Cause: rank_by called r["metrics"].get(metric); when callers passed
    metric="metrics.MeanTTFT" (the recipe / column convention), the
    bare-keyed metrics dict returned None for every row, every row
    fell into the (1, 0.0) sort bucket, and Python's stable sort
    returned them in collect-order. results_to_print[0] then anchored
    the export to the first-collected row, NOT the lowest-TTFT (or
    highest-throughput) winner.
    """

    def _r(self, ttft):
        return {"meta": {}, "metrics": {"MeanTTFT": ttft}, "combo_id": ttft}

    def test_rank_by_with_bare_metric_name_orders_correctly(self):
        # Sanity: with the bare name (post-normalization) rank_by has
        # always worked. Lock the contract.
        ranked = rank_by(
            [self._r("100.0"), self._r("10.0"), self._r("50.0")],
            metric="MeanTTFT", descending=False)
        self.assertEqual(
            [r["combo_id"] for r in ranked], ["10.0", "50.0", "100.0"])

    def test_normalize_metric_strips_prefix(self):
        self.assertEqual(_normalize_metric("metrics.MeanTTFT"), "MeanTTFT")
        self.assertEqual(_normalize_metric("MeanTTFT"), "MeanTTFT")
        self.assertEqual(
            _normalize_metric("metrics.RequestThroughput"),
            "RequestThroughput")
        # Edge: only the leading "metrics." is stripped.
        self.assertEqual(
            _normalize_metric("metrics.metrics.X"), "metrics.X")

    def test_rank_by_with_normalized_qualified_form_orders_correctly(self):
        # The recipe uses "metrics.MeanTTFT"; main() normalizes once at
        # the boundary; rank_by sees "MeanTTFT". This test exercises the
        # full path: take the recipe form, run it through _normalize_metric,
        # call rank_by — assert the lowest-TTFT row lands first.
        ranked = rank_by(
            [self._r("100.0"), self._r("10.0"), self._r("50.0")],
            metric=_normalize_metric("metrics.MeanTTFT"),
            descending=False)
        self.assertEqual(ranked[0]["combo_id"], "10.0")


class TestMainEndToEnd(unittest.TestCase):
    """Full-CLI path: main() with the qualified recipe-form --metric flag
    must rank correctly and export the right winner.

    Stubs collect_results so the test doesn't depend on an on-disk
    sweep dir. This is the integration test that would have caught the
    silent wrong-winner bug at review time — it asserts the
    end-to-end CLI -> rank -> export wiring, not just any single
    component in isolation.
    """

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)
        self.out_path = self.tmp_path / "production.service"

    def tearDown(self):
        self.tmpdir.cleanup()

    def _run_main_with_stubbed_results(self, results, *extra_args):
        from tools.benchmark import build_service_registry as bsr
        original = bsr.collect_results
        bsr.collect_results = lambda _path: results
        try:
            argv = [
                str(self.tmp_path),  # sweep_dir (unused; collect_results stubbed)
                "--export-production", str(self.out_path),
                "--kernel-id", "rpa_v3",
                *extra_args,
            ]
            rc = bsr.main(argv)
        finally:
            bsr.collect_results = original
        return rc

    def test_main_with_qualified_latency_metric_picks_lowest_ttft(self):
        # Recipe-form metric, ascending direction. The exported winner
        # must be the lowest-TTFT row, not the first-collected.
        results = [
            _result(ttft="100.0", max_num_seqs="A"),
            _result(ttft="10.0",  max_num_seqs="B"),
            _result(ttft="50.0",  max_num_seqs="C"),
        ]
        rc = self._run_main_with_stubbed_results(
            results,
            "--metric", "metrics.MeanTTFT",
            "--ascending",
            "--service-id", "vllm_latency",
        )
        self.assertEqual(rc, 0)
        with open(self.out_path) as f:
            entry = list(json.load(f)["best_configs_by_workload"].values())[0]
        self.assertEqual(entry["MAX_NUM_SEQS"], "B")

    def test_main_with_qualified_throughput_metric_picks_highest(self):
        # Recipe-form metric, descending (default). Original throughput
        # path must still pick the highest req/s. This is the contract
        # the rpa_v3+vllm recipe relies on.
        results = [
            _result(throughput="10.0", max_num_seqs="A"),
            _result(throughput="30.0", max_num_seqs="C"),
            _result(throughput="20.0", max_num_seqs="B"),
        ]
        rc = self._run_main_with_stubbed_results(
            results,
            "--metric", "metrics.RequestThroughput",
            "--service-id", "vllm",
        )
        self.assertEqual(rc, 0)
        with open(self.out_path) as f:
            entry = list(json.load(f)["best_configs_by_workload"].values())[0]
        self.assertEqual(entry["MAX_NUM_SEQS"], "C")


if __name__ == "__main__":
    unittest.main()
