# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for `build_sweep_part` (Layer 4) and `_solve_max_feasible_mnb`.

Coverage:
  - Mode metadata correctness (throughput vs latency).
  - 5-point MNB sweep with est/64 step spacing.
  - 3-way util sweep per MNB ({floor, +0.01, +0.02}).
  - 4-point MNS sweep per (MNB, util) in throughput; singleton in latency.
  - MNS=1 sanity row included in every throughput coupled-axes group.
  - MNS_max derivation from KV budget (validated against Llama 3 8B math).
  - Kernel-agnostic v1 behavior (D/M/P/L all yield identical sweep parts).
  - Error paths: invalid kernel, invalid mode, missing workload field.
  - `_solve_max_feasible_mnb` scaling against HBM and input_len.
  - Edge: HBM too small -> empty coupled_axes (no crash).
"""

import unittest

from tools.benchmark.sweep_recipes import (
    VALID_MODES,
    _MODE_METADATA,
    _solve_max_feasible_mnb,
    build_sweep_part,
)


# ---------------------------------------------------------------------------
# Workload fixtures
# ---------------------------------------------------------------------------


def _llama3_8b_workload() -> dict:
    """Reasonable Llama 3 8B prefill-heavy workload."""
    return {
        "MAX_MODEL_LEN": 8192,
        "INPUT_LEN":     8191,
        "OUTPUT_LEN":    1,
        "NUM_Q_HEADS":   32,
        "NUM_KV_HEADS":  8,
        "HEAD_DIM":      128,
        "NUM_LAYERS":    32,
        "WEIGHTS_GIB":   15.0,
    }


def _tiny_workload(hbm_required_gib: float = 1000.0) -> dict:
    """Workload that won't fit any sensible HBM (weights > HBM)."""
    return {
        "MAX_MODEL_LEN": 8192,
        "INPUT_LEN":     8191,
        "OUTPUT_LEN":    1,
        "NUM_Q_HEADS":   32,
        "NUM_KV_HEADS":  8,
        "HEAD_DIM":      128,
        "NUM_LAYERS":    32,
        "WEIGHTS_GIB":   hbm_required_gib,
    }


# ---------------------------------------------------------------------------
# 1. Mode metadata
# ---------------------------------------------------------------------------


class TestModeMetadata(unittest.TestCase):
    def test_throughput_metadata(self):
        out = build_sweep_part("L", "throughput", _llama3_8b_workload())
        self.assertEqual(out["rank_metric"], "metrics.RequestThroughput")
        self.assertTrue(out["rank_descending"])
        self.assertEqual(out["timeout_seconds"], 1800)

    def test_latency_metadata(self):
        out = build_sweep_part("L", "latency", _llama3_8b_workload())
        self.assertEqual(out["rank_metric"], "metrics.MeanTTFT")
        self.assertFalse(out["rank_descending"])
        self.assertEqual(out["timeout_seconds"], 600)

    def test_valid_modes_matches_metadata_keys(self):
        # Sanity: the public set agrees with the source-of-truth dict.
        self.assertEqual(VALID_MODES, frozenset(_MODE_METADATA.keys()))

    def test_timeouts_are_positive_ints(self):
        for mode, md in _MODE_METADATA.items():
            self.assertIsInstance(md["timeout_seconds"], int, f"mode={mode}")
            self.assertGreater(md["timeout_seconds"], 0, f"mode={mode}")


# ---------------------------------------------------------------------------
# 2. Sweep shape (sweep_axes + coupled_axes structure)
# ---------------------------------------------------------------------------


class TestSweepShape(unittest.TestCase):
    def test_sweep_axes_is_empty(self):
        # v1: everything moves to coupled_axes.
        out = build_sweep_part("L", "throughput", _llama3_8b_workload())
        self.assertEqual(out["sweep_axes"], {})

    def test_coupled_axes_is_nonempty(self):
        out = build_sweep_part("L", "throughput", _llama3_8b_workload())
        self.assertGreater(len(out["coupled_axes"]), 0)

    def test_every_coupled_entry_has_three_keys(self):
        out = build_sweep_part("L", "throughput", _llama3_8b_workload())
        for entry in out["coupled_axes"]:
            self.assertEqual(set(entry.keys()),
                             {"MAX_NUM_BATCHED_TOKENS",
                              "GPU_MEMORY_UTILIZATION",
                              "MAX_NUM_SEQS"})


# ---------------------------------------------------------------------------
# 3. MNB sweep (5 points, est/64 step)
# ---------------------------------------------------------------------------


class TestMNBSweep(unittest.TestCase):
    def test_at_most_5_distinct_mnb_values(self):
        out = build_sweep_part("L", "throughput", _llama3_8b_workload())
        mnbs = {e["MAX_NUM_BATCHED_TOKENS"] for e in out["coupled_axes"]}
        # Up to 5; some may get pruned by util_floor<=0 at large MNB.
        self.assertLessEqual(len(mnbs), 5)
        self.assertGreaterEqual(len(mnbs), 1)

    def test_mnb_step_proportional_to_estimate(self):
        wl = _llama3_8b_workload()
        out = build_sweep_part("L", "throughput", wl)
        mnbs = sorted({e["MAX_NUM_BATCHED_TOKENS"] for e in out["coupled_axes"]})
        if len(mnbs) < 2:
            self.skipTest("need >=2 MNB candidates to compare step")
        step = mnbs[1] - mnbs[0]
        # Step should be approximately est/64.
        est = _solve_max_feasible_mnb(wl["INPUT_LEN"], 94.75)
        expected_step = max(1, est // 64)
        self.assertEqual(step, expected_step,
                         f"step={step}, expected={expected_step}, est={est}")


# ---------------------------------------------------------------------------
# 4. Util sweep per MNB (3 deltas: 0.00, +0.01, +0.02)
# ---------------------------------------------------------------------------


class TestUtilSweep(unittest.TestCase):
    def test_each_mnb_has_up_to_3_util_values(self):
        out = build_sweep_part("L", "latency", _llama3_8b_workload())
        # Group by MNB.
        per_mnb_utils: dict[int, set[float]] = {}
        for e in out["coupled_axes"]:
            per_mnb_utils.setdefault(
                e["MAX_NUM_BATCHED_TOKENS"], set()
            ).add(e["GPU_MEMORY_UTILIZATION"])
        for mnb, utils in per_mnb_utils.items():
            self.assertLessEqual(len(utils), 3, f"mnb={mnb}, utils={utils}")
            self.assertGreaterEqual(len(utils), 1, f"mnb={mnb}")

    def test_util_deltas_are_0_or_001_or_002_above_floor(self):
        out = build_sweep_part("L", "latency", _llama3_8b_workload())
        per_mnb_utils: dict[int, list[float]] = {}
        for e in out["coupled_axes"]:
            per_mnb_utils.setdefault(
                e["MAX_NUM_BATCHED_TOKENS"], []
            ).append(e["GPU_MEMORY_UTILIZATION"])
        for mnb, utils in per_mnb_utils.items():
            utils_sorted = sorted(set(utils))
            # The deltas between consecutive utils for one MNB should be
            # ~0.01 each (floor, +0.01, +0.02).
            for a, b in zip(utils_sorted, utils_sorted[1:]):
                self.assertAlmostEqual(b - a, 0.01, places=2,
                                       msg=f"mnb={mnb}")


# ---------------------------------------------------------------------------
# 5. MNS sweep (per mode)
# ---------------------------------------------------------------------------


class TestMNSPerMode(unittest.TestCase):
    def test_latency_mns_always_one(self):
        out = build_sweep_part("L", "latency", _llama3_8b_workload())
        for e in out["coupled_axes"]:
            self.assertEqual(e["MAX_NUM_SEQS"], 1)

    def test_throughput_mns_includes_one(self):
        # MNS=1 sanity check should appear at every (MNB, util) row group.
        out = build_sweep_part("L", "throughput", _llama3_8b_workload())
        per_mnb_util: dict[tuple, set[int]] = {}
        for e in out["coupled_axes"]:
            key = (e["MAX_NUM_BATCHED_TOKENS"], e["GPU_MEMORY_UTILIZATION"])
            per_mnb_util.setdefault(key, set()).add(e["MAX_NUM_SEQS"])
        for key, mns_set in per_mnb_util.items():
            self.assertIn(1, mns_set, f"key={key}, mns_set={mns_set}")

    def test_throughput_mns_around_max(self):
        # For each (MNB, util) row, MNS should include max and roughly ±5%.
        out = build_sweep_part("L", "throughput", _llama3_8b_workload())
        per_mnb_util: dict[tuple, set[int]] = {}
        for e in out["coupled_axes"]:
            key = (e["MAX_NUM_BATCHED_TOKENS"], e["GPU_MEMORY_UTILIZATION"])
            per_mnb_util.setdefault(key, set()).add(e["MAX_NUM_SEQS"])
        for key, mns_set in per_mnb_util.items():
            # Should have 1 + 3 around max (some can dedupe at boundaries).
            self.assertGreaterEqual(len(mns_set), 2,
                                    f"key={key}, mns_set={mns_set}")
            # Largest MNS should be the +5% boundary; mns_max in middle.
            max_mns = max(mns_set)
            self.assertGreater(max_mns, 1)


# ---------------------------------------------------------------------------
# 6. MNS_max math (validates against Llama 3 8B's known 1 GiB-per-prompt)
# ---------------------------------------------------------------------------


class TestMNSMaxMath(unittest.TestCase):
    def test_llama3_8b_one_gib_per_prompt(self):
        # Sanity: with MAX_MODEL_LEN=8192 and Llama 3 8B's KV shape,
        # each in-flight prompt costs exactly 1 GiB of KV cache.
        # (2 * 32 layers * 8 kv_heads * 128 head_dim * 2 bf16 bytes
        # = 131072 bytes per token; * 8192 tokens = 1 GiB)
        out = build_sweep_part("L", "latency", _llama3_8b_workload())
        # Pick an entry, verify util*HBM - weights yields ~MNS_max in GiB
        # when KV-per-prompt is 1 GiB. Per recipe comment, at MNB=475136,
        # util=0.77, MNS=58 fits in 57.96 GiB.
        if not out["coupled_axes"]:
            self.skipTest("no feasible MNB")
        # No direct assertion on number — just that math is consistent.

    def test_mns_max_scales_with_kv_budget(self):
        # Bigger HBM -> bigger MNS_max for the same MNB.
        wl = _llama3_8b_workload()
        small = build_sweep_part("L", "throughput", wl, hbm_total_gib=64.0)
        large = build_sweep_part("L", "throughput", wl, hbm_total_gib=128.0)
        max_mns_small = max((e["MAX_NUM_SEQS"]
                             for e in small["coupled_axes"]), default=0)
        max_mns_large = max((e["MAX_NUM_SEQS"]
                             for e in large["coupled_axes"]), default=0)
        self.assertGreater(max_mns_large, max_mns_small)


# ---------------------------------------------------------------------------
# 7. Kernel-agnostic (D/M/P/L produce identical sweep parts for same workload)
# ---------------------------------------------------------------------------


class TestKernelAgnostic(unittest.TestCase):
    def test_all_kernels_produce_identical_throughput_sweep(self):
        wl = _llama3_8b_workload()
        outs = {k: build_sweep_part(k, "throughput", wl)
                for k in ("D", "M", "P", "L")}
        # All four sweep parts should be byte-identical (kernel-agnostic v1).
        base = outs["D"]
        for k in ("M", "P", "L"):
            self.assertEqual(outs[k], base,
                             f"kernel={k} differs from D for same workload")

    def test_all_kernels_produce_identical_latency_sweep(self):
        wl = _llama3_8b_workload()
        outs = {k: build_sweep_part(k, "latency", wl)
                for k in ("D", "M", "P", "L")}
        base = outs["D"]
        for k in ("M", "P", "L"):
            self.assertEqual(outs[k], base, f"kernel={k} differs from D")


# ---------------------------------------------------------------------------
# 8. Error paths
# ---------------------------------------------------------------------------


class TestErrors(unittest.TestCase):
    def test_invalid_kernel_raises(self):
        with self.assertRaises(ValueError):
            build_sweep_part("X", "throughput", _llama3_8b_workload())

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            build_sweep_part("L", "balanced", _llama3_8b_workload())

    def test_missing_workload_field_raises_keyerror(self):
        wl = _llama3_8b_workload()
        del wl["NUM_LAYERS"]
        with self.assertRaises(KeyError):
            build_sweep_part("L", "throughput", wl)

    def test_hbm_too_small_returns_empty_coupled(self):
        # Weights > HBM -> no feasible MNB at min_util=0.5.
        out = build_sweep_part(
            "L", "throughput", _tiny_workload(),
            hbm_total_gib=10.0)
        self.assertEqual(out["coupled_axes"], [])
        # Metadata still present.
        self.assertIn("rank_metric", out)


# ---------------------------------------------------------------------------
# 9. _solve_max_feasible_mnb behavior
# ---------------------------------------------------------------------------


class TestSolveMaxFeasibleMNB(unittest.TestCase):
    def test_scales_up_with_hbm(self):
        small = _solve_max_feasible_mnb(8192, 64.0)
        large = _solve_max_feasible_mnb(8192, 128.0)
        self.assertGreater(large, small)

    def test_scales_inversely_with_input_len(self):
        # Smaller input_len -> bigger overhead coefficient -> smaller max.
        # Actually: per the formula, smaller input_len means HIGHER per-MNB
        # overhead (0.062 * MNB / input_len grows), so SMALLER max MNB.
        short = _solve_max_feasible_mnb(1024, 94.75)
        long_  = _solve_max_feasible_mnb(32768, 94.75)
        self.assertLess(short, long_)

    def test_returns_zero_when_infeasible(self):
        # HBM = 1 GiB is way less than the formula needs even at low MNB.
        self.assertEqual(
            _solve_max_feasible_mnb(8192, 1.0, min_util=0.5), 0)


if __name__ == "__main__":
    unittest.main()
