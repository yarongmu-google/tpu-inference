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
    _make_workload_key, export_production_registry, _to_float,
)


def _result(model="m", tp="1", in_len="2048", out_len="2048",
            max_num_seqs="128", max_num_batched_tokens="10000",
            throughput="10.5"):
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
        "metrics": {"RequestThroughput": throughput},
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
            _result(), self.out_path,
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
            _result(), self.out_path,
            kernel_id="rpa_v3", service_id="vllm")
        export_production_registry(
            _result(throughput="20.0"), self.out_path,
            kernel_id="rpa_v4", service_id="vllm")
        with open(self.out_path) as f:
            data = json.load(f)
        # Two entries — second did NOT overwrite the first.
        self.assertEqual(len(data["best_configs_by_workload"]), 2)

    def test_overwrites_only_if_strictly_better(self):
        export_production_registry(
            _result(throughput="20.0"), self.out_path,
            kernel_id="rpa_v3", service_id="vllm")
        # Worse result — must NOT overwrite.
        export_production_registry(
            _result(throughput="15.0", max_num_seqs="999"),
            self.out_path,
            kernel_id="rpa_v3", service_id="vllm")
        with open(self.out_path) as f:
            data = json.load(f)
        entry = list(data["best_configs_by_workload"].values())[0]
        self.assertEqual(entry["MAX_NUM_SEQS"], "128")  # original kept
        # Strictly-better result — must overwrite.
        export_production_registry(
            _result(throughput="25.0", max_num_seqs="256"),
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
            _result(throughput="0.0", max_num_seqs="ALPHA"), self.out_path,
            kernel_id="rpa_v3", service_id="vllm")
        export_production_registry(
            _result(throughput="0.0", max_num_seqs="BETA"), self.out_path,
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
                _result(), self.out_path,
                kernel_id="rpa_v3", service_id="vllm")
        # Original (corrupt) content untouched.
        self.assertEqual(self.out_path.read_text(), "not valid json {{{")

    def test_write_is_atomic_no_partial_files_on_crash(self):
        # Smoke-check the atomic-write path: after a successful export,
        # there should be NO leftover .tmp files in the directory.
        export_production_registry(
            _result(), self.out_path,
            kernel_id="rpa_v3", service_id="vllm")
        leftovers = [p for p in self.tmp_path.iterdir()
                     if p.suffix == ".tmp"
                     or ".tmp" in p.name and p.name != "production.service"]
        self.assertEqual(leftovers, [])
        self.assertTrue(self.out_path.is_file())


if __name__ == "__main__":
    unittest.main()
