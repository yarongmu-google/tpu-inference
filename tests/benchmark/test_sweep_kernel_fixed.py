# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for `build_kernel_fixed` and its helpers in sweep_recipes.

Coverage targets:
  - All four routings (D, M, P, L) — base env emission, LPTT formula,
    BLOCK_SIZE source, RPA_*_BLOCK_SIZES content, min-latency winner
    selection across multiple registry rows.
  - Optional-case emission (Option Y, symmetric env surface): when the
    registry has e.g. a logical winner at the chosen page_size, an M
    routing still emits RPA_L_BLOCK_SIZES.
  - SKIP_BUCKET_AUTOGEN always 1.
  - Pin schema (strict per-routing): extras raise ValueError.
  - Invalid kernel raises ValueError.
  - Missing primary winner raises SpecError.
  - Missing required supporting winner at primary's page_size raises SpecError.
  - LOGICAL primary missing max_num_subseqs raises SpecError (catches
    pre-refactor registries).
"""

import unittest

from tools.benchmark.sweep import SpecError
from tools.benchmark.sweep_recipes import (
    PIN_SCHEMA,
    build_kernel_fixed,
)


# ---------------------------------------------------------------------------
# Registry fixture builders
# ---------------------------------------------------------------------------


def _winner(
    *,
    bq_sz: int = 32,
    bkv_sz: int = 512,
    bq_csz: int = 32,
    bkv_csz: int = 256,
    page_size: int,
    chunk_prefill_size: int = 0,
    max_num_subseqs: int | None = None,
    latency_us: float = 1000.0,
) -> dict:
    """Build one registry entry. Defaults are arbitrary but consistent."""
    tp: dict = {
        "bq_sz": bq_sz, "bkv_sz": bkv_sz,
        "bq_csz": bq_csz, "bkv_csz": bkv_csz,
        "page_size": page_size,
        "chunk_prefill_size": chunk_prefill_size,
    }
    if max_num_subseqs is not None:
        tp["max_num_subseqs"] = max_num_subseqs
    return {"tuning_key": {}, "tunable_params": tp, "latency_us": latency_us}


def _minimal_registry(*, include_optional: bool = False) -> dict:
    """A 4-case registry with one winner per case at page_size=128.
    PREFILL/LOGICAL have K=256; LOGICAL has mnss=33.
    """
    reg = {
        "results": {
            "decode":  [_winner(page_size=128, latency_us=100.0)],
            "mixed":   [_winner(page_size=128, latency_us=200.0)],
        }
    }
    if include_optional:
        reg["results"]["prefill"] = [
            _winner(page_size=128, chunk_prefill_size=256, latency_us=300.0)
        ]
        reg["results"]["logical"] = [
            _winner(page_size=128, chunk_prefill_size=256,
                    max_num_subseqs=33, latency_us=400.0)
        ]
    return reg


# ---------------------------------------------------------------------------
# 1. D routing
# ---------------------------------------------------------------------------


class TestDRouting(unittest.TestCase):
    """Decode-only routing. Primary case = decode; no required supporting."""

    def setUp(self):
        # D routing minimum: just a decode winner.
        self.reg = {
            "results": {
                "decode": [_winner(page_size=128, latency_us=100.0)],
            }
        }

    def test_minimum_returns_expected_keys(self):
        env = build_kernel_fixed("D", self.reg)
        self.assertEqual(env["BLOCK_SIZE"], 128)
        self.assertEqual(env["SKIP_BUCKET_AUTOGEN"], 1)
        self.assertEqual(env["LONG_PREFILL_TOKEN_THRESHOLD"], 0)
        self.assertIn("RPA_D_BLOCK_SIZES", env)
        # No required M, P, L for D-only routing.
        self.assertNotIn("RPA_M_BLOCK_SIZES", env)
        self.assertNotIn("RPA_P_BLOCK_SIZES", env)
        self.assertNotIn("RPA_L_BLOCK_SIZES", env)
        # Cat 4-extras don't apply to D.
        self.assertNotIn("RPA_KERNEL_K", env)
        self.assertNotIn("RPA_MAX_NUM_SUBSEQS", env)

    def test_lptt_is_zero(self):
        env = build_kernel_fixed("D", self.reg)
        self.assertEqual(env["LONG_PREFILL_TOKEN_THRESHOLD"], 0)

    def test_block_size_from_decode_winner(self):
        reg = {"results": {"decode": [_winner(page_size=256)]}}
        env = build_kernel_fixed("D", reg)
        self.assertEqual(env["BLOCK_SIZE"], 256)

    def test_d_block_sizes_format(self):
        reg = {"results": {"decode": [_winner(
            bq_sz=1, bkv_sz=2, bq_csz=3, bkv_csz=4, page_size=128)]}}
        env = build_kernel_fixed("D", reg)
        self.assertEqual(env["RPA_D_BLOCK_SIZES"], "1,2,3,4")

    def test_picks_min_latency_winner(self):
        reg = {"results": {"decode": [
            _winner(page_size=128, bq_sz=10, latency_us=999.0),
            _winner(page_size=128, bq_sz=20, latency_us=100.0),  # min
            _winner(page_size=128, bq_sz=30, latency_us=500.0),
        ]}}
        env = build_kernel_fixed("D", reg)
        # Min-latency winner had bq_sz=20.
        self.assertTrue(env["RPA_D_BLOCK_SIZES"].startswith("20,"))


# ---------------------------------------------------------------------------
# 2. M routing
# ---------------------------------------------------------------------------


class TestMRouting(unittest.TestCase):
    """MIXED routing. Primary = mixed; requires decode at same page_size."""

    def test_minimum_returns_expected_keys(self):
        reg = _minimal_registry()
        env = build_kernel_fixed("M", reg)
        self.assertEqual(env["BLOCK_SIZE"], 128)
        self.assertEqual(env["LONG_PREFILL_TOKEN_THRESHOLD"], 0)
        self.assertIn("RPA_D_BLOCK_SIZES", env)
        self.assertIn("RPA_M_BLOCK_SIZES", env)
        # No P/L in this minimal registry.
        self.assertNotIn("RPA_P_BLOCK_SIZES", env)
        self.assertNotIn("RPA_L_BLOCK_SIZES", env)

    def test_lptt_is_zero(self):
        env = build_kernel_fixed("M", _minimal_registry())
        self.assertEqual(env["LONG_PREFILL_TOKEN_THRESHOLD"], 0)

    def test_block_size_from_mixed_winner(self):
        reg = {"results": {
            "mixed":  [_winner(page_size=256, latency_us=200.0)],
            "decode": [_winner(page_size=256, latency_us=100.0)],
        }}
        env = build_kernel_fixed("M", reg)
        self.assertEqual(env["BLOCK_SIZE"], 256)

    def test_picks_min_latency_winner(self):
        reg = {"results": {
            "mixed": [
                _winner(page_size=128, bq_sz=10, latency_us=999.0),
                _winner(page_size=128, bq_sz=20, latency_us=100.0),  # min
            ],
            "decode": [_winner(page_size=128, latency_us=50.0)],
        }}
        env = build_kernel_fixed("M", reg)
        self.assertTrue(env["RPA_M_BLOCK_SIZES"].startswith("20,"))


# ---------------------------------------------------------------------------
# 3. P routing
# ---------------------------------------------------------------------------


class TestPRouting(unittest.TestCase):
    """Static-K PREFILL routing. Primary = prefill; needs decode + mixed."""

    def setUp(self):
        self.reg = {
            "results": {
                "decode":  [_winner(page_size=128, latency_us=100.0)],
                "mixed":   [_winner(page_size=128, latency_us=200.0)],
                "prefill": [_winner(page_size=128, chunk_prefill_size=256,
                                    latency_us=300.0)],
            }
        }

    def test_minimum_returns_expected_keys(self):
        env = build_kernel_fixed("P", self.reg)
        self.assertEqual(env["BLOCK_SIZE"], 128)
        self.assertEqual(env["RPA_KERNEL_K"], 256)
        self.assertEqual(env["LONG_PREFILL_TOKEN_THRESHOLD"], 256)
        self.assertIn("RPA_D_BLOCK_SIZES", env)
        self.assertIn("RPA_M_BLOCK_SIZES", env)
        self.assertIn("RPA_P_BLOCK_SIZES", env)
        self.assertNotIn("RPA_MAX_NUM_SUBSEQS", env)
        self.assertNotIn("RPA_L_BLOCK_SIZES", env)

    def test_lptt_equals_K(self):
        env = build_kernel_fixed("P", self.reg)
        self.assertEqual(env["LONG_PREFILL_TOKEN_THRESHOLD"], 256)

    def test_kernel_K_set(self):
        reg = dict(self.reg)
        reg["results"]["prefill"] = [_winner(
            page_size=128, chunk_prefill_size=512, latency_us=300.0)]
        env = build_kernel_fixed("P", reg)
        self.assertEqual(env["RPA_KERNEL_K"], 512)
        self.assertEqual(env["LONG_PREFILL_TOKEN_THRESHOLD"], 512)

    def test_picks_min_latency_primary(self):
        reg = {
            "results": {
                "decode":  [_winner(page_size=128, latency_us=100.0)],
                "mixed":   [_winner(page_size=128, latency_us=200.0)],
                "prefill": [
                    _winner(page_size=128, chunk_prefill_size=512,
                            bq_sz=10, latency_us=999.0),
                    _winner(page_size=128, chunk_prefill_size=256,
                            bq_sz=20, latency_us=300.0),  # min
                ],
            }
        }
        env = build_kernel_fixed("P", reg)
        self.assertEqual(env["RPA_KERNEL_K"], 256)
        self.assertTrue(env["RPA_P_BLOCK_SIZES"].startswith("20,"))


# ---------------------------------------------------------------------------
# 4. L routing
# ---------------------------------------------------------------------------


class TestLRouting(unittest.TestCase):
    """LOGICAL routing. Primary = logical; needs decode + mixed."""

    def setUp(self):
        self.reg = {
            "results": {
                "decode":  [_winner(page_size=128, latency_us=100.0)],
                "mixed":   [_winner(page_size=128, latency_us=200.0)],
                "logical": [_winner(
                    page_size=128, chunk_prefill_size=256,
                    max_num_subseqs=33, latency_us=400.0)],
            }
        }

    def test_minimum_returns_expected_keys(self):
        env = build_kernel_fixed("L", self.reg)
        self.assertEqual(env["BLOCK_SIZE"], 128)
        self.assertEqual(env["RPA_KERNEL_K"], 256)
        self.assertEqual(env["RPA_MAX_NUM_SUBSEQS"], 33)
        self.assertEqual(env["LONG_PREFILL_TOKEN_THRESHOLD"], 256 * 33)
        self.assertIn("RPA_D_BLOCK_SIZES", env)
        self.assertIn("RPA_M_BLOCK_SIZES", env)
        self.assertIn("RPA_L_BLOCK_SIZES", env)
        self.assertNotIn("RPA_P_BLOCK_SIZES", env)

    def test_lptt_equals_K_times_mnss(self):
        reg = {
            "results": {
                "decode":  [_winner(page_size=128, latency_us=100.0)],
                "mixed":   [_winner(page_size=128, latency_us=200.0)],
                "logical": [_winner(
                    page_size=128, chunk_prefill_size=512,
                    max_num_subseqs=64, latency_us=400.0)],
            }
        }
        env = build_kernel_fixed("L", reg)
        self.assertEqual(env["LONG_PREFILL_TOKEN_THRESHOLD"], 512 * 64)

    def test_L_block_sizes_from_logical_winner(self):
        reg = {
            "results": {
                "decode":  [_winner(page_size=128, latency_us=100.0)],
                "mixed":   [_winner(page_size=128, latency_us=200.0)],
                "logical": [_winner(
                    page_size=128, chunk_prefill_size=256,
                    max_num_subseqs=33,
                    bq_sz=11, bkv_sz=22, bq_csz=33, bkv_csz=44,
                    latency_us=400.0)],
            }
        }
        env = build_kernel_fixed("L", reg)
        self.assertEqual(env["RPA_L_BLOCK_SIZES"], "11,22,33,44")

    def test_picks_min_latency_primary(self):
        reg = {
            "results": {
                "decode":  [_winner(page_size=128, latency_us=100.0)],
                "mixed":   [_winner(page_size=128, latency_us=200.0)],
                "logical": [
                    _winner(page_size=128, chunk_prefill_size=256,
                            max_num_subseqs=33, bq_sz=10, latency_us=999.0),
                    _winner(page_size=128, chunk_prefill_size=512,
                            max_num_subseqs=66, bq_sz=20, latency_us=400.0),
                ]
            }
        }
        env = build_kernel_fixed("L", reg)
        # Min-latency winner had K=512, mnss=66, bq=20.
        self.assertEqual(env["RPA_KERNEL_K"], 512)
        self.assertEqual(env["RPA_MAX_NUM_SUBSEQS"], 66)
        self.assertEqual(env["LONG_PREFILL_TOKEN_THRESHOLD"], 512 * 66)


# ---------------------------------------------------------------------------
# 5. Optional case emission (Option Y — symmetric env surface)
# ---------------------------------------------------------------------------


class TestOptionalCaseEmission(unittest.TestCase):
    """Non-required cases get their block sizes emitted IF present in
    registry at the chosen page_size (Option Y)."""

    def test_M_routing_emits_optional_L_when_present(self):
        reg = _minimal_registry(include_optional=True)
        env = build_kernel_fixed("M", reg)
        # M doesn't REQUIRE L or P, but registry has them → emit.
        self.assertIn("RPA_P_BLOCK_SIZES", env)
        self.assertIn("RPA_L_BLOCK_SIZES", env)

    def test_D_routing_emits_all_optional_when_present(self):
        reg = _minimal_registry(include_optional=True)
        env = build_kernel_fixed("D", reg)
        self.assertIn("RPA_M_BLOCK_SIZES", env)
        self.assertIn("RPA_P_BLOCK_SIZES", env)
        self.assertIn("RPA_L_BLOCK_SIZES", env)

    def test_optional_case_at_different_page_size_not_emitted(self):
        # L exists at page=256, not page=128. Primary picks 128, so L
        # at 256 shouldn't be emitted.
        reg = {
            "results": {
                "decode":  [_winner(page_size=128, latency_us=100.0)],
                "mixed":   [_winner(page_size=128, latency_us=200.0)],
                "logical": [_winner(page_size=256, chunk_prefill_size=256,
                                    max_num_subseqs=33, latency_us=400.0)],
            }
        }
        env = build_kernel_fixed("M", reg)
        self.assertEqual(env["BLOCK_SIZE"], 128)
        self.assertNotIn("RPA_L_BLOCK_SIZES", env)


# ---------------------------------------------------------------------------
# 6. SKIP_BUCKET_AUTOGEN invariant
# ---------------------------------------------------------------------------


class TestSkipBucketAutogen(unittest.TestCase):
    def test_always_set_to_one_for_all_routings(self):
        for kernel, reg in [
            ("D", {"results": {"decode": [_winner(page_size=128)]}}),
            ("M", _minimal_registry()),
            ("P", {"results": {
                "decode":  [_winner(page_size=128)],
                "mixed":   [_winner(page_size=128)],
                "prefill": [_winner(page_size=128, chunk_prefill_size=256)],
            }}),
            ("L", {"results": {
                "decode":  [_winner(page_size=128)],
                "mixed":   [_winner(page_size=128)],
                "logical": [_winner(page_size=128, chunk_prefill_size=256,
                                    max_num_subseqs=33)],
            }}),
        ]:
            env = build_kernel_fixed(kernel, reg)
            self.assertEqual(env["SKIP_BUCKET_AUTOGEN"], 1,
                             f"kernel={kernel}")


# ---------------------------------------------------------------------------
# 7. Pin schema (strict per-routing)
# ---------------------------------------------------------------------------


class TestPinSchema(unittest.TestCase):
    def test_pin_selects_specific_winner(self):
        reg = {
            "results": {
                "decode":  [_winner(page_size=128, latency_us=100.0)],
                "mixed":   [_winner(page_size=128, latency_us=200.0)],
                "logical": [
                    _winner(page_size=128, chunk_prefill_size=256,
                            max_num_subseqs=33, latency_us=400.0),
                    _winner(page_size=128, chunk_prefill_size=512,
                            max_num_subseqs=66, latency_us=999.0),  # worse
                ],
            }
        }
        # No pin: picks min-latency (K=256).
        env = build_kernel_fixed("L", reg)
        self.assertEqual(env["RPA_KERNEL_K"], 256)
        # Pin to K=512: forces the worse winner.
        env = build_kernel_fixed("L", reg, pin={"K": 512})
        self.assertEqual(env["RPA_KERNEL_K"], 512)
        self.assertEqual(env["RPA_MAX_NUM_SUBSEQS"], 66)

    def test_pin_no_match_raises_spec_error(self):
        reg = _minimal_registry(include_optional=True)
        with self.assertRaises(SpecError):
            build_kernel_fixed("L", reg, pin={"K": 9999})

    def test_M_routing_pin_with_K_raises_value_error(self):
        # M's pin schema doesn't allow K (M has no chunk_prefill_size meaning).
        reg = _minimal_registry()
        with self.assertRaises(ValueError):
            build_kernel_fixed("M", reg, pin={"K": 256})

    def test_P_routing_pin_with_mnss_raises_value_error(self):
        reg = {"results": {
            "decode":  [_winner(page_size=128)],
            "mixed":   [_winner(page_size=128)],
            "prefill": [_winner(page_size=128, chunk_prefill_size=256)],
        }}
        with self.assertRaises(ValueError):
            build_kernel_fixed("P", reg, pin={"mnss": 33})

    def test_pin_schema_constants_match_expected(self):
        # Sanity-check the static map so a future edit can't silently widen.
        self.assertEqual(PIN_SCHEMA["D"], {"page_size"})
        self.assertEqual(PIN_SCHEMA["M"], {"page_size"})
        self.assertEqual(PIN_SCHEMA["P"], {"page_size", "K"})
        self.assertEqual(PIN_SCHEMA["L"], {"page_size", "K", "mnss"})


# ---------------------------------------------------------------------------
# 8. Error cases
# ---------------------------------------------------------------------------


class TestErrorCases(unittest.TestCase):
    def test_invalid_kernel_raises_value_error(self):
        with self.assertRaises(ValueError):
            build_kernel_fixed("X", {"results": {}})

    def test_no_primary_winner_raises_spec_error(self):
        # L routing, registry has no logical winners.
        reg = _minimal_registry()
        with self.assertRaises(SpecError):
            build_kernel_fixed("L", reg)

    def test_no_supporting_winner_at_primary_page_size_raises(self):
        # L routing: logical winner at page=128, but decode only at page=256.
        reg = {
            "results": {
                "decode":  [_winner(page_size=256, latency_us=100.0)],
                "mixed":   [_winner(page_size=128, latency_us=200.0)],
                "logical": [_winner(page_size=128, chunk_prefill_size=256,
                                    max_num_subseqs=33, latency_us=400.0)],
            }
        }
        with self.assertRaises(SpecError):
            build_kernel_fixed("L", reg)

    def test_L_routing_missing_mnss_raises(self):
        # Old-format LOGICAL entry without max_num_subseqs in tunable_params.
        reg = {
            "results": {
                "decode":  [_winner(page_size=128)],
                "mixed":   [_winner(page_size=128)],
                "logical": [{
                    "tuning_key": {},
                    "tunable_params": {
                        "bq_sz": 1, "bkv_sz": 2,
                        "bq_csz": 3, "bkv_csz": 4,
                        "page_size": 128, "chunk_prefill_size": 256,
                        # Note: no max_num_subseqs.
                    },
                    "latency_us": 400.0,
                }],
            }
        }
        with self.assertRaises(SpecError):
            build_kernel_fixed("L", reg)


if __name__ == "__main__":
    unittest.main()
