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
        # Sort numerically — string values come out of enumerate_combos
        # but ascending integer order is what a reader expects.
        ks = sorted((c["LONG_PREFILL_TOKEN_THRESHOLD"] for c in combos),
                    key=int)
        self.assertEqual(ks, ["512", "2048"])
        # Coupled (K, RPA_P_BLOCK_SIZES) pairing — both K values pull
        # from the all-three-flavors tune (page=128 winners). After that
        # round, K=512 and K=2048 happen to share the same PREFILL block
        # configuration (bq=256 bkv=2048 bq_c=256 bkv_c=512); the prior
        # PREFILL-only round had them differ. Assertion checks both
        # values explicitly so a future tune that re-differentiates them
        # surfaces here, not in a silent serving regression.
        by_k = {c["LONG_PREFILL_TOKEN_THRESHOLD"]: c for c in combos}
        self.assertEqual(by_k["512"]["RPA_P_BLOCK_SIZES"], "256,2048,256,512")
        self.assertEqual(by_k["2048"]["RPA_P_BLOCK_SIZES"], "256,2048,256,512")

    def test_baseline_full_parses_and_enumerates(self):
        # 3 combos: MAX_NUM_BATCHED_TOKENS in {2048, 4096, 8192}, no
        # coupled axes, K pinned to 0 (baseline).
        spec = self._load("llama3_8b_v7x_baseline_full.json")
        combos = sweep.enumerate_combos(spec)
        self.assertEqual(len(combos), 3,
                         "baseline_full should produce 3 combos "
                         "(MAX_NUM_BATCHED_TOKENS x 3, no coupled axes)")
        # Every combo must carry K=0 (chunk-prefill OFF) and the
        # tuned D/M block sizes from the fixed block. Catches a stray
        # K override or missing fixed-block plumbing.
        for c in combos:
            self.assertEqual(c["LONG_PREFILL_TOKEN_THRESHOLD"], "0")
            self.assertEqual(c["RPA_D_BLOCK_SIZES"], "1,4096,1,1024")
            self.assertEqual(c["RPA_M_BLOCK_SIZES"], "64,512,64,256")

    def test_optimized_full_parses_and_enumerates(self):
        # 15 combos: MAX_NUM_BATCHED_TOKENS x K, with K coupled to its
        # own RPA_P_BLOCK_SIZES from the tuning table.
        spec = self._load("llama3_8b_v7x_optimized_full.json")
        combos = sweep.enumerate_combos(spec)
        self.assertEqual(len(combos), 15,
                         "optimized_full should produce 15 combos "
                         "(3 MAX_NUM_BATCHED_TOKENS x 5 coupled K values)")
        # Spot-check the K=2048 / K=128 winners — both have distinct
        # RPA_P_BLOCK_SIZES from the all-three-flavors tune. If a
        # future re-tune shifts these, the test surfaces it.
        by_k = {c["LONG_PREFILL_TOKEN_THRESHOLD"]: c for c in combos}
        self.assertEqual(by_k["128"]["RPA_P_BLOCK_SIZES"], "128,2048,128,1024")
        self.assertEqual(by_k["1024"]["RPA_P_BLOCK_SIZES"], "256,1024,256,512")
        self.assertEqual(by_k["2048"]["RPA_P_BLOCK_SIZES"], "256,2048,256,512")

    def test_baseline_full_4k_parses_and_enumerates(self):
        # 4 combos: 2 MAX_NUM_SEQS x 2 MAX_NUM_BATCHED_TOKENS, no
        # coupled axes. The MAX_NUM_SEQS=1000 row is the load-bearing
        # one — its meant to expose v7x's HBM-backed concurrency
        # advantage that 128 leaves on the table.
        spec = self._load("llama3_8b_v7x_baseline_full_4k.json")
        combos = sweep.enumerate_combos(spec)
        self.assertEqual(len(combos), 4,
                         "baseline_full_4k should produce 4 combos "
                         "(2 MAX_NUM_SEQS x 2 MAX_NUM_BATCHED_TOKENS)")
        pairs = {(c["MAX_NUM_SEQS"], c["MAX_NUM_BATCHED_TOKENS"])
                 for c in combos}
        self.assertEqual(
            pairs,
            {("128", "4096"), ("128", "10275"),
             ("1000", "4096"), ("1000", "10275")})

    def test_optimized_full_4k_parses_and_enumerates(self):
        # 20 combos: (2 MAX_NUM_SEQS x 2 MAX_NUM_BATCHED_TOKENS) x 5 K.
        # The cartesian-with-coupled-K is the load-bearing structure
        # — if enumerate_combos ever drops the cross-product behavior
        # this will surface here, not silently in a half-sized sweep.
        spec = self._load("llama3_8b_v7x_optimized_full_4k.json")
        combos = sweep.enumerate_combos(spec)
        self.assertEqual(len(combos), 20,
                         "optimized_full_4k should produce 20 combos "
                         "(2 x 2 x 5)")
        # Each K should appear exactly 4 times (once per
        # (max_num_seqs, max_num_batched_tokens) pair).
        from collections import Counter
        k_counts = Counter(c["LONG_PREFILL_TOKEN_THRESHOLD"]
                           for c in combos)
        self.assertEqual(dict(k_counts),
                         {"128": 4, "256": 4, "512": 4,
                          "1024": 4, "2048": 4})

    def test_all_specs_have_a_safe_timeout(self):
        # All specs should pin a per-combo timeout below the generous
        # default so a hung run does not waste the budget.
        for name in ("llama3_8b_v7x_baseline_smoke.json",
                     "llama3_8b_v7x_optimized_smoke.json",
                     "llama3_8b_v7x_baseline_full.json",
                     "llama3_8b_v7x_optimized_full.json",
                     "llama3_8b_v7x_baseline_full_4k.json",
                     "llama3_8b_v7x_optimized_full_4k.json",
                     "llama3_8b_v7x_baseline_full_4k.json"):
            spec = self._load(name)
            self.assertIn("timeout_seconds", spec, msg=name)
            self.assertLessEqual(spec["timeout_seconds"],
                                 sweep.DEFAULT_TIMEOUT_SECONDS, msg=name)
            self.assertGreater(spec["timeout_seconds"], 0, msg=name)

    def test_all_specs_resolve_case_file(self):
        # case_file is encoded as '../cases/...' — the load_spec
        # resolution should produce an absolute path that points
        # at an existing file.
        for name in ("llama3_8b_v7x_baseline_smoke.json",
                     "llama3_8b_v7x_optimized_smoke.json",
                     "llama3_8b_v7x_baseline_full.json",
                     "llama3_8b_v7x_optimized_full.json",
                     "llama3_8b_v7x_baseline_full_4k.json",
                     "llama3_8b_v7x_optimized_full_4k.json",
                     "llama3_8b_v7x_baseline_full_4k.json"):
            spec = self._load(name)
            cf = Path(spec["case_file"])
            self.assertTrue(cf.is_absolute(), msg=name)
            self.assertTrue(cf.is_file(),
                            msg=f"{name}: case_file does not exist: {cf}")


if __name__ == "__main__":
    unittest.main()
