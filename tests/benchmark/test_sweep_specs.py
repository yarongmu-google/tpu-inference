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
        spec = self._load("llama3_8b_v7x_baseline_smoke.service")
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
        spec = self._load("llama3_8b_v7x_optimized_smoke.service")
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
        spec = self._load("llama3_8b_v7x_baseline_full.service")
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
        # 15 combos: MAX_NUM_BATCHED_TOKENS x K. Block sizes come from
        # the kernel_registry via auto-link (production.kernel), NOT
        # hardcoded coupled_axes — so this test asserts the auto-link
        # semantics (each K gets *some* PREFILL block sizes injected,
        # plus D/M block sizes) without pinning to specific tune
        # results. Pinning to values would re-break this test on every
        # re-tune; the values themselves are exercised by
        # test_sweep_auto_link.
        spec = self._load("llama3_8b_v7x_optimized_full.service")
        # Skip if the registry is missing (e.g. this is a fresh checkout
        # before the first tune).
        if "_loaded_kernel_registry" not in spec:
            self.skipTest(
                "kernel_registry not loaded — run tools/run_pipeline.sh "
                "or the kernel tuner first to produce production.kernel")
        # enumerate_combos raises SpecError if any spec K is missing
        # from the registry (since b1c8d4f4 / silent-skip fix). So a
        # successful call here means the registry covers every K in
        # the spec — which is exactly the precondition for firing the
        # full sweep without wasting TPU on un-tuned defaults.
        combos = sweep.enumerate_combos(spec)
        self.assertEqual(len(combos), 15,
                         "optimized_full should produce 15 combos "
                         "(3 MAX_NUM_BATCHED_TOKENS x 5 K values)")
        ks = {c["LONG_PREFILL_TOKEN_THRESHOLD"] for c in combos}
        self.assertEqual(ks, {"128", "256", "512", "1024", "2048"})
        # Auto-link populated D, M, and P blocks on every combo. Format
        # is "bq,bkv,bq_csz,bkv_csz" — four comma-separated positive
        # ints. Don't pin specific values; just verify the auto-link
        # fired for all three flavors.
        for c in combos:
            for key in ("RPA_D_BLOCK_SIZES", "RPA_M_BLOCK_SIZES",
                        "RPA_P_BLOCK_SIZES"):
                self.assertIn(key, c)
                parts = c[key].split(",")
                self.assertEqual(len(parts), 4, f"{key}={c[key]}")
                for p in parts:
                    self.assertTrue(p.isdigit(), f"{key}={c[key]}")

    def test_baseline_full_4k_parses_and_enumerates(self):
        # 4 combos: 2 MAX_NUM_SEQS x 2 MAX_NUM_BATCHED_TOKENS, no
        # coupled axes. The MAX_NUM_SEQS=1000 row is the load-bearing
        # one — its meant to expose v7x's HBM-backed concurrency
        # advantage that 128 leaves on the table.
        spec = self._load("llama3_8b_v7x_baseline_full_4k.service")
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
        spec = self._load("llama3_8b_v7x_optimized_full_4k.service")
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
        for name in ("llama3_8b_v7x_baseline_smoke.service",
                     "llama3_8b_v7x_optimized_smoke.service",
                     "llama3_8b_v7x_baseline_full.service",
                     "llama3_8b_v7x_optimized_full.service",
                     "llama3_8b_v7x_baseline_full_4k.service",
                     "llama3_8b_v7x_optimized_full_4k.service",
                     "llama3_8b_v7x_baseline_full_4k.service"):
            spec = self._load(name)
            self.assertIn("timeout_seconds", spec, msg=name)
            self.assertLessEqual(spec["timeout_seconds"],
                                 sweep.DEFAULT_TIMEOUT_SECONDS, msg=name)
            self.assertGreater(spec["timeout_seconds"], 0, msg=name)

    def test_all_specs_resolve_case_file(self):
        # case_file is encoded as '../cases/...' — the load_spec
        # resolution should produce an absolute path that points
        # at an existing file.
        for name in ("llama3_8b_v7x_baseline_smoke.service",
                     "llama3_8b_v7x_optimized_smoke.service",
                     "llama3_8b_v7x_baseline_full.service",
                     "llama3_8b_v7x_optimized_full.service",
                     "llama3_8b_v7x_baseline_full_4k.service",
                     "llama3_8b_v7x_optimized_full_4k.service",
                     "llama3_8b_v7x_baseline_full_4k.service"):
            spec = self._load(name)
            cf = Path(spec["case_file"])
            self.assertTrue(cf.is_absolute(), msg=name)
            self.assertTrue(cf.is_file(),
                            msg=f"{name}: case_file does not exist: {cf}")


if __name__ == "__main__":
    unittest.main()
