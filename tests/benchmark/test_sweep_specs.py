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
"""Smoke-test that the checked-in sweep specs parse and produce the
expected combo counts.

Functions as a regression guard: if a future change to the spec format
or to a checked-in spec file desyncs them, the smoke tests at the top
of a sweep dev cycle catch it before TPU time is spent.
"""

import unittest
from pathlib import Path

from tools.benchmark import sweep


REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEPS_DIR = REPO_ROOT / "tools" / "benchmark" / "sweeps"


class TestSmokeSpecs(unittest.TestCase):

    def _load(self, name: str) -> dict:
        return sweep.load_spec(str(SWEEPS_DIR / name))

    def test_baseline_smoke_parses_and_enumerates(self):
        spec = self._load("llama3_8b_v7x_baseline_smoke.json")
        combos = sweep.enumerate_combos(spec)
        self.assertEqual(len(combos), 2,
                         "baseline_smoke spec should produce 2 combos "
                         "(2 MAX_NUM_BATCHED_TOKENS values × empty coupled)")
        # Every combo has the chunk-prefill optimization OFF.
        for c in combos:
            self.assertEqual(c["LONG_PREFILL_TOKEN_THRESHOLD"], "0")
        # The two combos differ only in MAX_NUM_BATCHED_TOKENS.
        self.assertEqual(
            sorted(c["MAX_NUM_BATCHED_TOKENS"] for c in combos),
            ["2048", "4096"])

    def test_optimized_smoke_parses_and_enumerates(self):
        spec = self._load("llama3_8b_v7x_optimized_smoke.json")
        combos = sweep.enumerate_combos(spec)
        self.assertEqual(len(combos), 2,
                         "optimized_smoke spec should produce 2 combos "
                         "(1 MAX_NUM_BATCHED_TOKENS × 2 coupled K values)")
        ks = sorted(c["LONG_PREFILL_TOKEN_THRESHOLD"] for c in combos)
        self.assertEqual(ks, ["2048", "512"])
        # Coupled (K, RPA_P_BLOCK_SIZES) pairing — the K=512 combo
        # gets its specific block sizes, K=2048 gets its own.
        by_k = {c["LONG_PREFILL_TOKEN_THRESHOLD"]: c for c in combos}
        self.assertEqual(by_k["512"]["RPA_P_BLOCK_SIZES"], "256,1024,128,1024")
        self.assertEqual(by_k["2048"]["RPA_P_BLOCK_SIZES"], "256,512,256,512")

    def test_smoke_specs_have_a_safe_timeout(self):
        # Both smoke specs should pin a per-combo timeout below the
        # generous default so a hung run doesn't waste the budget.
        for name in ("llama3_8b_v7x_baseline_smoke.json",
                     "llama3_8b_v7x_optimized_smoke.json"):
            spec = self._load(name)
            self.assertIn("timeout_seconds", spec, msg=name)
            self.assertLessEqual(spec["timeout_seconds"],
                                 sweep.DEFAULT_TIMEOUT_SECONDS, msg=name)
            self.assertGreater(spec["timeout_seconds"], 0, msg=name)

    def test_smoke_specs_resolve_case_file(self):
        # case_file is encoded as '../cases/...' — the load_spec
        # resolution should produce an absolute path that points
        # at an existing file.
        for name in ("llama3_8b_v7x_baseline_smoke.json",
                     "llama3_8b_v7x_optimized_smoke.json"):
            spec = self._load(name)
            cf = Path(spec["case_file"])
            self.assertTrue(cf.is_absolute(), msg=name)
            self.assertTrue(cf.is_file(),
                            msg=f"{name}: case_file does not exist: {cf}")


if __name__ == "__main__":
    unittest.main()
