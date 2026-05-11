# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for tools.tuning.v2.kernel.enumerate_logical.

Verifies constraint application and tuning_key/tunable_params shapes.
Uses small search spaces so the cartesian product is exhaustively
checkable.
"""

import unittest

from tools.tuning.v2.kernel.enumerate_logical import (
    enumerate_logical_combos,
)


MODEL_SHAPE = {
    "num_q_heads":    32,
    "num_kv_heads":   8,
    "head_dim":       128,
    "max_model_len":  8192,
    "q_dtype":        "bfloat16",
    "kv_dtype":       "bfloat16",
    "sliding_window": None,
}


def _enum(**overrides) -> list[tuple[dict, dict]]:
    """Helper: enumerate with minimal default args, override as needed."""
    base = dict(
        max_num_seqs=128,
        max_num_batched_tokens=8192,
        model_shape=MODEL_SHAPE,
        code_revision="abc12345",
        search_space={
            "page_size": [128],
            "kernel_K":  [256],
            "mnss":      [4224],
            "bq_sz":     [256],
            "bkv_sz":    [2048],
            "bq_csz":    [256],
            "bkv_csz":   [512],
        },
    )
    base.update(overrides)
    return list(enumerate_logical_combos(**base))


class TestEnumerateLogical(unittest.TestCase):

    # ----- happy path -----

    def test_single_valid_combo_yields_one_pair(self):
        combos = _enum()
        self.assertEqual(len(combos), 1)
        tk, tp = combos[0]
        self.assertEqual(tk["case"], "logical")
        self.assertEqual(tk["page_size"], 128)
        self.assertEqual(tk["kernel_K"], 256)
        self.assertEqual(tk["max_num_seqs"], 128)
        self.assertEqual(tk["code_revision"], "abc12345")
        self.assertEqual(tk["num_q_heads"], 32)
        self.assertEqual(tp, {
            "bq_sz":   256,
            "bkv_sz":  2048,
            "bq_csz":  256,
            "bkv_csz": 512,
            "mnss":    4224,
        })

    def test_cartesian_product_count_with_constraint_pruning(self):
        """Cartesian product after constraint pruning.

        At max_num_seqs=128, mnss=4224, K=256: per_phys_q = 8192 (==MNB) OK.
        At max_num_seqs=128, mnss=4224, K=512: per_phys_q = 16384 > MNB=8192, PRUNED.
        So only K=256 yields combos:
          2 (page) × 1 (K=256) × 1 (mnss) × 2 (bq_sz) × 1 × 1 × 1 = 4.
        """
        combos = _enum(
            search_space={
                "page_size": [64, 128],
                "kernel_K":  [256, 512],
                "mnss":      [4224],
                "bq_sz":     [128, 256],
                "bkv_sz":    [2048],
                "bq_csz":    [128],
                "bkv_csz":   [512],
            },
        )
        self.assertEqual(len(combos), 4)
        # No K=512 combos survived the per_phys_q prune.
        ks = {tk["kernel_K"] for tk, _tp in combos}
        self.assertEqual(ks, {256})

    # ----- constraint #1: mnss >= max_num_seqs -----

    def test_skips_mnss_below_max_num_seqs(self):
        combos = _enum(
            max_num_seqs=128,
            search_space={
                "page_size": [128], "kernel_K": [256],
                "mnss":      [64, 128, 256],   # 64 < 128 should drop
                "bq_sz":     [256], "bkv_sz": [2048],
                "bq_csz":    [256], "bkv_csz": [512],
            },
        )
        # mnss=128 = mns: m_per_phys = max(1, 128//128 - 1) = max(1, 0) = 1
        #   per_phys_q = 256 * 1 = 256, fits MNB=8192 ✓
        # mnss=256: m_per_phys = max(1, 256//128 - 1) = max(1, 1) = 1
        #   per_phys_q = 256 ✓
        # mnss=64: pruned by constraint #1
        mnsses = {tp["mnss"] for _tk, tp in combos}
        self.assertNotIn(64, mnsses)
        self.assertIn(128, mnsses)
        self.assertIn(256, mnsses)

    # ----- constraint #2: per_phys_q <= MNB -----

    def test_skips_per_phys_q_overflow(self):
        """At mns=1, K=256, MNB=8192: only mnss <= 33 are valid (per_phys_q
        ≤ 8192). Larger mnss yield per_phys_q > MNB and must be dropped."""
        combos = _enum(
            max_num_seqs=1,
            max_num_batched_tokens=8192,
            search_space={
                "page_size": [128], "kernel_K": [256],
                "mnss":      [2, 33, 65, 129],
                "bq_sz":     [256], "bkv_sz": [2048],
                "bq_csz":    [256], "bkv_csz": [512],
            },
        )
        mnsses = {tp["mnss"] for _tk, tp in combos}
        # mnss=2: m_per_phys = max(1, 2-1) = 1, per_phys_q = 256, OK
        # mnss=33: m_per_phys = max(1, 33-1) = 32, per_phys_q = 8192, OK (==MNB)
        # mnss=65: per_phys_q = 64*256 = 16384 > 8192, PRUNE
        # mnss=129: per_phys_q = 128*256 = 32768 > 8192, PRUNE
        self.assertEqual(mnsses, {2, 33})

    # ----- constraint #3: bq_sz <= kernel_K -----

    def test_skips_bq_sz_above_kernel_K(self):
        combos = _enum(
            search_space={
                "page_size": [128], "kernel_K": [256],
                "mnss":      [4224],
                "bq_sz":     [128, 256, 512, 1024],  # > K=256 pruned
                "bkv_sz":    [2048],
                "bq_csz":    [128], "bkv_csz": [512],
            },
        )
        bq_szs = {tp["bq_sz"] for _tk, tp in combos}
        self.assertEqual(bq_szs, {128, 256})

    # ----- constraint #4: bq_csz <= bq_sz, bkv_csz <= bkv_sz -----

    def test_skips_bq_csz_above_bq_sz(self):
        combos = _enum(
            search_space={
                "page_size": [128], "kernel_K": [256],
                "mnss":      [4224],
                "bq_sz":     [128],
                "bkv_sz":    [2048],
                "bq_csz":    [64, 128, 256, 512],   # > bq_sz=128 pruned
                "bkv_csz":   [512],
            },
        )
        bq_cszs = {tp["bq_csz"] for _tk, tp in combos}
        self.assertEqual(bq_cszs, {64, 128})

    def test_skips_bkv_csz_above_bkv_sz(self):
        combos = _enum(
            search_space={
                "page_size": [128], "kernel_K": [256],
                "mnss":      [4224],
                "bq_sz":     [256], "bkv_sz": [1024],
                "bq_csz":    [256],
                "bkv_csz":   [256, 1024, 2048],   # > bkv_sz=1024 pruned
            },
        )
        bkv_cszs = {tp["bkv_csz"] for _tk, tp in combos}
        self.assertEqual(bkv_cszs, {256, 1024})

    # ----- tuning_key shape -----

    def test_tuning_key_includes_model_shape(self):
        combos = _enum()
        tk, _ = combos[0]
        for k in ("num_q_heads", "num_kv_heads", "head_dim",
                  "max_model_len", "q_dtype", "kv_dtype",
                  "sliding_window"):
            self.assertIn(k, tk)

    def test_tuning_key_case_is_logical(self):
        combos = _enum()
        for tk, _ in combos:
            self.assertEqual(tk["case"], "logical")

    def test_tunable_params_keys(self):
        _, tp = _enum()[0]
        self.assertEqual(
            set(tp.keys()),
            {"bq_sz", "bkv_sz", "bq_csz", "bkv_csz", "mnss"},
        )

    # ----- empty cases -----

    def test_empty_search_space_axis_yields_no_combos(self):
        combos = _enum(
            search_space={
                "page_size": [], "kernel_K": [256], "mnss": [4224],
                "bq_sz":     [256], "bkv_sz": [2048],
                "bq_csz":    [256], "bkv_csz": [512],
            },
        )
        self.assertEqual(combos, [])

    def test_all_combos_pruned_yields_no_combos(self):
        """Every combo violates per_phys_q constraint."""
        combos = _enum(
            max_num_seqs=1,
            max_num_batched_tokens=128,   # tiny budget
            search_space={
                "page_size": [128], "kernel_K": [256],
                "mnss":      [10],   # M=9, per_phys_q=2304 > MNB
                "bq_sz":     [256], "bkv_sz": [2048],
                "bq_csz":    [256], "bkv_csz": [512],
            },
        )
        self.assertEqual(combos, [])

    # ----- deterministic iteration order -----

    def test_iteration_order_deterministic(self):
        first = _enum()
        second = _enum()
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
