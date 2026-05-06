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

There are no longer dedicated `*_smoke.service` files — the orchestrator
takes a `--smoke` flag that natively truncates any .service to 1 combo,
so a separate copy is redundant. These tests cover the four production
specs (baseline_full, baseline_full_4k, optimized_full, optimized_full_4k).
"""

import unittest
from collections import Counter
from pathlib import Path

from tools.benchmark import sweep


REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEPS_DIR = REPO_ROOT / "tools" / "benchmark" / "sweeps"

ALL_SPECS = (
    "llama3_8b_v7x_baseline_full.service",
    "llama3_8b_v7x_baseline_full_4k.service",
    "llama3_8b_v7x_optimized_full.service",
    "llama3_8b_v7x_optimized_full_4k.service",
)


def _block_sizes_str_valid(value):
    """Format check: four comma-separated positive integers."""
    parts = value.split(",")
    if len(parts) != 4:
        return False
    return all(p.isdigit() and int(p) > 0 for p in parts)


class TestSmokeSpecs(unittest.TestCase):

    def _load(self, name: str) -> dict:
        return sweep.load_spec(str(SWEEPS_DIR / name))

    # ------------------------- baseline_full -------------------------

    def test_baseline_full_parses_and_enumerates(self):
        # 3 combos: MAX_NUM_BATCHED_TOKENS in {2048, 4096, 8192}, no
        # coupled axes, K pinned to 0 (chunk-prefill OFF).
        spec = self._load("llama3_8b_v7x_baseline_full.service")
        if "_loaded_kernel_registry" not in spec:
            self.skipTest(
                "kernel_registry not loaded — run tools/run_pipeline.sh "
                "or the kernel tuner first to produce production.kernel")
        combos = sweep.enumerate_combos(spec)
        self.assertEqual(len(combos), 3,
                         "baseline_full should produce 3 combos "
                         "(MAX_NUM_BATCHED_TOKENS x 3, no coupled axes)")
        # Every combo must carry K=0 and the auto-linked D + M block
        # sizes from the registry. PREFILL is NOT injected since K=0.
        for c in combos:
            self.assertEqual(c["LONG_PREFILL_TOKEN_THRESHOLD"], "0")
            self.assertIn("RPA_D_BLOCK_SIZES", c)
            self.assertIn("RPA_M_BLOCK_SIZES", c)
            self.assertNotIn("RPA_P_BLOCK_SIZES", c)
            self.assertTrue(_block_sizes_str_valid(c["RPA_D_BLOCK_SIZES"]))
            self.assertTrue(_block_sizes_str_valid(c["RPA_M_BLOCK_SIZES"]))

    # ----------------------- baseline_full_4k ------------------------

    def test_baseline_full_4k_parses_and_enumerates(self):
        # 4 combos: 2 MAX_NUM_SEQS x 2 MAX_NUM_BATCHED_TOKENS, K=0.
        # The MAX_NUM_SEQS=1000 row is the load-bearing one — its meant
        # to expose v7xs HBM-backed concurrency advantage.
        spec = self._load("llama3_8b_v7x_baseline_full_4k.service")
        if "_loaded_kernel_registry" not in spec:
            self.skipTest(
                "kernel_registry not loaded — run tools/run_pipeline.sh "
                "or the kernel tuner first to produce production.kernel")
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
        for c in combos:
            self.assertEqual(c["LONG_PREFILL_TOKEN_THRESHOLD"], "0")
            self.assertIn("RPA_D_BLOCK_SIZES", c)
            self.assertIn("RPA_M_BLOCK_SIZES", c)
            self.assertNotIn("RPA_P_BLOCK_SIZES", c)

    # ------------------------ optimized_full -------------------------

    def test_optimized_full_parses_and_enumerates(self):
        # 15 combos: 3 MAX_NUM_BATCHED_TOKENS x 5 K. All blocks (D, M,
        # P) auto-linked from production.kernel keyed on (case, page,
        # K). enumerate_combos itself raises if the registry is missing
        # any K, so a successful call here means the registry covers
        # every K in the spec — exactly the precondition for firing
        # the full sweep without wasting TPU on un-tuned defaults.
        spec = self._load("llama3_8b_v7x_optimized_full.service")
        if "_loaded_kernel_registry" not in spec:
            self.skipTest(
                "kernel_registry not loaded — run tools/run_pipeline.sh "
                "or the kernel tuner first to produce production.kernel")
        combos = sweep.enumerate_combos(spec)
        self.assertEqual(len(combos), 15,
                         "optimized_full should produce 15 combos "
                         "(3 MAX_NUM_BATCHED_TOKENS x 5 K values)")
        ks = {c["LONG_PREFILL_TOKEN_THRESHOLD"] for c in combos}
        self.assertEqual(ks, {"128", "256", "512", "1024", "2048"})
        for c in combos:
            for key in ("RPA_D_BLOCK_SIZES", "RPA_M_BLOCK_SIZES",
                        "RPA_P_BLOCK_SIZES"):
                self.assertIn(key, c)
                self.assertTrue(_block_sizes_str_valid(c[key]),
                                f"{key}={c[key]}")

    # ----------------------- optimized_full_4k -----------------------

    def test_optimized_full_4k_parses_and_enumerates(self):
        # 20 combos: 2 MAX_NUM_SEQS x 2 MAX_NUM_BATCHED_TOKENS x 5 K
        # (full cartesian; coupled_axes is now empty after migration to
        # kernel_registry auto-link).
        spec = self._load("llama3_8b_v7x_optimized_full_4k.service")
        if "_loaded_kernel_registry" not in spec:
            self.skipTest(
                "kernel_registry not loaded — run tools/run_pipeline.sh "
                "or the kernel tuner first to produce production.kernel")
        combos = sweep.enumerate_combos(spec)
        self.assertEqual(len(combos), 20,
                         "optimized_full_4k should produce 20 combos "
                         "(2 x 2 x 5)")
        # Each K should appear exactly 4 times (once per
        # (max_num_seqs, max_num_batched_tokens) pair).
        k_counts = Counter(c["LONG_PREFILL_TOKEN_THRESHOLD"]
                           for c in combos)
        self.assertEqual(dict(k_counts),
                         {"128": 4, "256": 4, "512": 4,
                          "1024": 4, "2048": 4})
        for c in combos:
            for key in ("RPA_D_BLOCK_SIZES", "RPA_M_BLOCK_SIZES",
                        "RPA_P_BLOCK_SIZES"):
                self.assertIn(key, c)

    # -------------------------- meta tests ---------------------------

    def test_all_specs_have_a_safe_timeout(self):
        # All specs should pin a per-combo timeout below the generous
        # default so a hung run does not waste the budget.
        for name in ALL_SPECS:
            spec = self._load(name)
            self.assertIn("timeout_seconds", spec, msg=name)
            self.assertLessEqual(spec["timeout_seconds"],
                                 sweep.DEFAULT_TIMEOUT_SECONDS, msg=name)
            self.assertGreater(spec["timeout_seconds"], 0, msg=name)

    def test_all_specs_resolve_case_file(self):
        # case_file is encoded as '../cases/...' — the load_spec
        # resolution should produce an absolute path that points
        # at an existing file.
        for name in ALL_SPECS:
            spec = self._load(name)
            cf = Path(spec["case_file"])
            self.assertTrue(cf.is_absolute(), msg=name)
            self.assertTrue(cf.is_file(),
                            msg=f"{name}: case_file does not exist: {cf}")

    def test_all_specs_use_kernel_registry(self):
        # After the migration, every checked-in spec sources block sizes
        # from production.kernel via auto-link rather than hardcoding.
        # If a future PR adds a spec without kernel_registry, this test
        # surfaces the architectural drift.
        for name in ALL_SPECS:
            spec_path = SWEEPS_DIR / name
            import json
            with open(spec_path) as f:
                raw = json.load(f)
            self.assertIn("kernel_registry", raw,
                          f"{name} must reference kernel_registry — "
                          "do not re-introduce hardcoded RPA_*_BLOCK_SIZES.")
            fixed = raw.get("fixed", {}) or {}
            for k in ("RPA_D_BLOCK_SIZES",
                      "RPA_M_BLOCK_SIZES",
                      "RPA_P_BLOCK_SIZES"):
                self.assertNotIn(
                    k, fixed,
                    f"{name}: {k} hardcoded in fixed; remove and let "
                    "auto-link from kernel_registry inject it.")


if __name__ == "__main__":
    unittest.main()
