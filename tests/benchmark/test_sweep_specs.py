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
"""Tests for the sweep_recipes module + the orchestrator's synthesized
spec.

After the architectural refactor (the .service files were deleted; the
orchestrator now synthesizes specs from .workload + a per-(kernel,
service) recipe), there are no checked-in `.service` files to test
against. These tests cover:

  - The recipe table is well-shaped for each registered (kernel,
    service) entry.
  - Synthesis from a real .workload produces a spec sweep.py can
    parse.
  - The synthesized spec preserves the architectural invariants:
    case_file points to the .workload, kernel_registry points to
    production.kernel beside the workload, and block sizes are NOT
    hardcoded in fixed (they auto-link from the registry).
"""

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from tools.benchmark import sweep
from tools.benchmark.sweep_recipes import RECIPES, synthesize_service_spec


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKLOAD_DIR = REPO_ROOT / "tools" / "benchmark" / "cases"


class TestRecipes(unittest.TestCase):

    def test_recipes_have_required_fields(self):
        # Each registered (kernel, service) recipe must have the
        # fields the synthesizer expects. Catches a typo'd recipe at
        # test time instead of at orchestrator-runtime.
        for key, recipe in RECIPES.items():
            self.assertIsInstance(key, tuple, msg=key)
            self.assertEqual(len(key), 2, msg=key)
            self.assertIn("sweep_axes", recipe, msg=key)
            self.assertIn("fixed", recipe, msg=key)
            self.assertIn("timeout_seconds", recipe, msg=key)
            self.assertIsInstance(recipe["sweep_axes"], dict, msg=key)
            self.assertIsInstance(recipe["fixed"], dict, msg=key)
            self.assertIsInstance(recipe["timeout_seconds"], int, msg=key)
            # rank_metric / rank_descending are optional but, if present,
            # must be well-typed. The orchestrator forwards them to
            # build_service_registry; a typo'd type here would surface
            # as a CLI parse error mid-pipeline.
            if "rank_metric" in recipe:
                self.assertIsInstance(recipe["rank_metric"], str, msg=key)
            if "rank_descending" in recipe:
                self.assertIsInstance(recipe["rank_descending"], bool,
                                      msg=key)

    def test_no_block_sizes_hardcoded_in_fixed(self):
        # RPA_*_BLOCK_SIZES come from production.kernel via auto-link.
        # If a recipe ever pins them in `fixed`, that defeats the
        # auto-link and re-creates the staleness problem the registry
        # was supposed to solve.
        for key, recipe in RECIPES.items():
            for k in recipe["fixed"]:
                self.assertFalse(
                    k.startswith("RPA_") and k.endswith("_BLOCK_SIZES"),
                    f"Recipe {key} pins {k} in fixed; remove and let "
                    "auto-link from production.kernel inject it.")


class TestSynthesizeFromRealWorkload(unittest.TestCase):
    """End-to-end: take a real .workload, synthesize a spec, parse it
    via sweep.load_spec, enumerate combos. Catches mismatches between
    the synthesizer and what sweep.py expects."""

    def setUp(self):
        # Pick the prefill_heavy workload as the canonical input —
        # it's the one tonight's run uses.
        self.workload = (
            WORKLOAD_DIR / "v7x" / "llama3_8b" / "prefill_heavy.workload")
        self.assertTrue(self.workload.is_file(),
                        f"Sanity: {self.workload} should exist")

    def test_synthesize_for_rpa_v3_vllm(self):
        spec = synthesize_service_spec(
            str(self.workload), kernel_id="rpa_v3", service_id="vllm")
        # Architectural invariants
        self.assertEqual(Path(spec["case_file"]).resolve(),
                         self.workload.resolve())
        self.assertEqual(
            Path(spec["kernel_registry"]).name, "production.kernel")
        self.assertIn("sweep_axes", spec)
        self.assertIn("fixed", spec)
        self.assertNotIn("RPA_D_BLOCK_SIZES", spec["fixed"])
        self.assertNotIn("RPA_M_BLOCK_SIZES", spec["fixed"])
        self.assertNotIn("RPA_P_BLOCK_SIZES", spec["fixed"])
        # Spec must round-trip through JSON without losing anything
        roundtripped = json.loads(json.dumps(spec))
        self.assertEqual(roundtripped["sweep_axes"], spec["sweep_axes"])

    def test_synthesized_spec_parses_via_sweep(self):
        # Write to a temp file, then load_spec it. If sweep.load_spec
        # rejects the synthesized shape, the orchestrator would crash
        # at runtime; this catches it at test-time.
        spec = synthesize_service_spec(
            str(self.workload), kernel_id="rpa_v3", service_id="vllm")
        with tempfile.NamedTemporaryFile(
                "w", suffix=".service", delete=False) as f:
            json.dump(spec, f)
            spec_path = f.name
        try:
            loaded = sweep.load_spec(spec_path)
        finally:
            import os
            os.unlink(spec_path)
        # Registry might not exist (fresh checkout) — sweep.load_spec
        # raises in that case. Skip the enumeration check if so.
        if "_loaded_kernel_registry" not in loaded:
            self.skipTest(
                "production.kernel not present yet — Layer 1 has not run.")
        # Registry MAY exist but not cover every K in the recipe.
        # enumerate_combos raises in that case (the strict-auto-link
        # pre-flight). That is the desired behavior at runtime — fail
        # before TPU cycles are spent — but as a unit test it just
        # means we are mid-tune. Skip with a clear message.
        try:
            combos = sweep.enumerate_combos(loaded)
        except sweep.SpecError as e:
            self.skipTest(
                f"production.kernel is incomplete relative to the recipe; "
                f"finish the tune first. Underlying message: {e}")
        recipe = RECIPES[("rpa_v3", "vllm")]
        expected = 1
        for v in recipe["sweep_axes"].values():
            expected *= len(v)
        self.assertEqual(len(combos), expected,
                         f"Combo count = {len(combos)}; recipe expects "
                         f"{expected}")
        # Each K value should appear exactly (combos / len(K)) times,
        # one for each (max_num_batched_tokens, max_num_seqs) pair.
        ks_in_recipe = recipe["sweep_axes"]["LONG_PREFILL_TOKEN_THRESHOLD"]
        k_counts = Counter(c["LONG_PREFILL_TOKEN_THRESHOLD"] for c in combos)
        per_k = expected // len(ks_in_recipe)
        for k in ks_in_recipe:
            self.assertEqual(k_counts[str(k)], per_k,
                             f"K={k} appears {k_counts[str(k)]} times, "
                             f"expected {per_k}")

    def test_unknown_recipe_raises(self):
        with self.assertRaisesRegex(ValueError, "No sweep recipe"):
            synthesize_service_spec(
                str(self.workload),
                kernel_id="nonexistent_kernel", service_id="vllm")

    def test_synthesize_for_rpa_v3_vllm_latency(self):
        # The latency recipe pins LONG_PREFILL_TOKEN_THRESHOLD=0
        # (MIXED-only) and MAX_NUM_SEQS=1 in `fixed`, sweeps only
        # MAX_NUM_BATCHED_TOKENS, and ranks ascending by Mean TTFT.
        # Lock those invariants — they are the contract the workload
        # files (cases/v7x/*/*_latency.workload) depend on.
        latency_workload = (
            WORKLOAD_DIR / "v7x" / "llama3_8b" / "prefill_heavy_latency.workload")
        self.assertTrue(latency_workload.is_file(),
                        f"Sanity: {latency_workload} should exist")
        spec = synthesize_service_spec(
            str(latency_workload),
            kernel_id="rpa_v3", service_id="vllm_latency")
        self.assertEqual(spec["fixed"]["LONG_PREFILL_TOKEN_THRESHOLD"], 0)
        self.assertEqual(spec["fixed"]["MAX_NUM_SEQS"], 1)
        self.assertNotIn("LONG_PREFILL_TOKEN_THRESHOLD", spec["sweep_axes"])
        self.assertNotIn("MAX_NUM_SEQS", spec["sweep_axes"])
        self.assertIn("MAX_NUM_BATCHED_TOKENS", spec["sweep_axes"])
        self.assertEqual(spec["rank_metric"], "metrics.MeanTTFT")
        self.assertFalse(spec["rank_descending"])

    def test_default_rank_fields_when_recipe_omits_them(self):
        # synthesize_service_spec must default rank_metric /
        # rank_descending to throughput-style values when a recipe
        # leaves them unset. Otherwise older recipes (or new ones the
        # author forgot to annotate) would silently break the
        # orchestrator's --metric forwarding.
        from tools.benchmark import sweep_recipes as sr
        original = sr.RECIPES
        try:
            sr.RECIPES = {
                ("dummy_kernel", "dummy_service"): {
                    "sweep_axes": {"MAX_NUM_BATCHED_TOKENS": [8192]},
                    "fixed": {"BLOCK_SIZE": 128},
                    "timeout_seconds": 600,
                    # rank_metric / rank_descending intentionally omitted.
                },
            }
            spec = sr.synthesize_service_spec(
                str(self.workload),
                kernel_id="dummy_kernel", service_id="dummy_service")
            self.assertEqual(spec["rank_metric"], "metrics.RequestThroughput")
            self.assertTrue(spec["rank_descending"])
        finally:
            sr.RECIPES = original


if __name__ == "__main__":
    unittest.main()
