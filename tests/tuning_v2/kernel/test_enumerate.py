# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for tools.tuning.v2.kernel.enumerate (D/M/P/L)."""

import unittest

from tools.tuning.v2.kernel.enumerate import (
    ALL_CASES,
    CASE_DECODE,
    CASE_LOGICAL,
    CASE_MIXED,
    CASE_PREFILL,
    _common_blocks_ok,
    enumerate_all_combos,
    enumerate_decode_combos,
    enumerate_mixed_combos,
    enumerate_prefill_combos,
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


def _kwargs(**overrides):
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
    return base


class TestCommonBlocksOk(unittest.TestCase):

    def _ok(self, **overrides):
        kw = dict(
            page_size=128, bq_sz=256, bkv_sz=2048,
            bq_csz=256, bkv_csz=512, max_model_len=8192,
        )
        kw.update(overrides)
        return _common_blocks_ok(**kw)

    def test_modulo_violation_bq(self):
        self.assertFalse(self._ok(bq_sz=200, bq_csz=128))

    def test_modulo_violation_bkv(self):
        self.assertFalse(self._ok(bkv_sz=1000, bkv_csz=512))

    def test_modulo_violation_page_on_bkv_sz(self):
        # bkv_sz not divisible by page_size.
        self.assertFalse(self._ok(
            bkv_sz=200, bkv_csz=200, page_size=128,
        ))

    def test_modulo_violation_page_on_bkv_csz(self):
        # bkv_csz not divisible by page_size. Must pick a bkv_sz
        # that DOES satisfy bkv_sz % bkv_csz == 0 and
        # bkv_sz % page_size == 0 so this rule is the discriminator.
        self.assertFalse(self._ok(
            bkv_sz=512, bkv_csz=64, page_size=128,
        ))

    def test_pages_per_seq_cap(self):
        """bkv_sz / page_size > pages_per_seq is over-sized."""
        # max_model_len=8192, page_size=128 → pages_per_seq=64.
        # bkv_sz=16384 → 128 pages > 64.
        self.assertFalse(self._ok(
            bkv_sz=16384, max_model_len=8192,
        ))

    def test_accepts_valid_combo(self):
        self.assertTrue(self._ok())


class TestEnumerateDecode(unittest.TestCase):

    def test_pins_bq_sz_and_bq_csz_to_one(self):
        combos = list(enumerate_decode_combos(**_kwargs()))
        self.assertEqual(len(combos), 1)
        _tk, tp = combos[0]
        self.assertEqual(tp["bq_sz"], 1)
        self.assertEqual(tp["bq_csz"], 1)

    def test_tuning_key_case_is_decode_with_kernel_K_zero(self):
        _tk, _ = next(iter(enumerate_decode_combos(**_kwargs())))
        self.assertEqual(_tk["case"], CASE_DECODE)
        self.assertEqual(_tk["kernel_K"], 0)

    def test_tunable_params_omits_mnss(self):
        _, tp = next(iter(enumerate_decode_combos(**_kwargs())))
        self.assertNotIn("mnss", tp)

    def test_drops_combos_that_violate_common_constraints(self):
        """DECODE applies the same modulo prune as other cases.
        bkv_sz=300 isn't divisible by page_size=128 -> dropped."""
        kw = _kwargs(search_space={
            "page_size": [128], "kernel_K": [256], "mnss": [4224],
            "bq_sz": [256], "bkv_sz": [300],   # bad modulo
            "bq_csz": [256], "bkv_csz": [128],
        })
        combos = list(enumerate_decode_combos(**kw))
        self.assertEqual(combos, [])

    def test_iterates_over_bkv_axes(self):
        kw = _kwargs(search_space={
            "page_size": [128], "kernel_K": [256], "mnss": [4224],
            "bq_sz": [256], "bkv_sz": [2048, 4096],
            "bq_csz": [256], "bkv_csz": [512, 1024],
        })
        combos = list(enumerate_decode_combos(**kw))
        # 2 bkv_sz × 2 bkv_csz = 4; all should pass common
        # constraints at page_size=128.
        self.assertEqual(len(combos), 4)


class TestEnumerateMixed(unittest.TestCase):

    def test_case_is_mixed_kernel_K_zero(self):
        combos = list(enumerate_mixed_combos(**_kwargs()))
        self.assertEqual(len(combos), 1)
        tk, tp = combos[0]
        self.assertEqual(tk["case"], CASE_MIXED)
        self.assertEqual(tk["kernel_K"], 0)
        self.assertNotIn("mnss", tp)

    def test_drops_bq_csz_larger_than_bq_sz(self):
        kw = _kwargs(search_space={
            "page_size": [128], "kernel_K": [256], "mnss": [4224],
            "bq_sz": [64], "bkv_sz": [2048],
            "bq_csz": [128],   # > bq_sz, dropped
            "bkv_csz": [512],
        })
        self.assertEqual(list(enumerate_mixed_combos(**kw)), [])

    def test_drops_bkv_csz_larger_than_bkv_sz(self):
        kw = _kwargs(search_space={
            "page_size": [128], "kernel_K": [256], "mnss": [4224],
            "bq_sz": [256], "bkv_sz": [512],
            "bq_csz": [256], "bkv_csz": [1024],   # > bkv_sz
        })
        self.assertEqual(list(enumerate_mixed_combos(**kw)), [])

    def test_drops_combos_that_violate_common_constraints(self):
        kw = _kwargs(search_space={
            "page_size": [128], "kernel_K": [256], "mnss": [4224],
            "bq_sz": [256], "bkv_sz": [300],   # bad modulo
            "bq_csz": [256], "bkv_csz": [128],
        })
        self.assertEqual(list(enumerate_mixed_combos(**kw)), [])

    def test_sweeps_full_bq_sz_range(self):
        kw = _kwargs(search_space={
            "page_size": [128], "kernel_K": [256], "mnss": [4224],
            "bq_sz": [256, 512],
            "bkv_sz":    [2048],
            "bq_csz":    [256],
            "bkv_csz":   [512],
        })
        combos = list(enumerate_mixed_combos(**kw))
        bq_sizes = sorted({tp["bq_sz"] for _, tp in combos})
        self.assertEqual(bq_sizes, [256, 512])


class TestEnumeratePrefill(unittest.TestCase):

    def test_case_is_prefill_kernel_K_carried(self):
        combos = list(enumerate_prefill_combos(**_kwargs()))
        self.assertEqual(len(combos), 1)
        tk, tp = combos[0]
        self.assertEqual(tk["case"], CASE_PREFILL)
        self.assertEqual(tk["kernel_K"], 256)
        self.assertNotIn("mnss", tp)

    def test_drops_bq_sz_larger_than_kernel_K(self):
        kw = _kwargs(search_space={
            "page_size": [128], "kernel_K": [128], "mnss": [4224],
            "bq_sz": [64, 256],   # 256 > 128, should be dropped
            "bkv_sz": [2048], "bq_csz": [64], "bkv_csz": [512],
        })
        combos = list(enumerate_prefill_combos(**kw))
        self.assertEqual(len(combos), 1)
        self.assertEqual(combos[0][1]["bq_sz"], 64)

    def test_drops_bq_csz_larger_than_bq_sz(self):
        kw = _kwargs(search_space={
            "page_size": [128], "kernel_K": [256], "mnss": [4224],
            "bq_sz": [64], "bkv_sz": [2048],
            "bq_csz": [128],   # > bq_sz
            "bkv_csz": [512],
        })
        self.assertEqual(list(enumerate_prefill_combos(**kw)), [])

    def test_drops_bkv_csz_larger_than_bkv_sz(self):
        kw = _kwargs(search_space={
            "page_size": [128], "kernel_K": [256], "mnss": [4224],
            "bq_sz": [256], "bkv_sz": [512],
            "bq_csz": [256], "bkv_csz": [1024],   # > bkv_sz
        })
        self.assertEqual(list(enumerate_prefill_combos(**kw)), [])

    def test_drops_combos_that_violate_common_constraints(self):
        kw = _kwargs(search_space={
            "page_size": [128], "kernel_K": [256], "mnss": [4224],
            "bq_sz": [256], "bkv_sz": [300],   # bad modulo
            "bq_csz": [256], "bkv_csz": [128],
        })
        self.assertEqual(list(enumerate_prefill_combos(**kw)), [])

    def test_drops_kernel_K_not_divisible_by_bq_sz(self):
        kw = _kwargs(search_space={
            "page_size": [128], "kernel_K": [200],   # 200 % 64 != 0
            "mnss": [4224],
            "bq_sz": [64], "bkv_sz": [2048],
            "bq_csz": [64], "bkv_csz": [512],
        })
        combos = list(enumerate_prefill_combos(**kw))
        self.assertEqual(combos, [])


class TestEnumerateAllCombos(unittest.TestCase):

    def test_emits_all_four_cases_by_default(self):
        combos = list(enumerate_all_combos(**_kwargs()))
        cases = sorted({tk["case"] for tk, _ in combos})
        self.assertEqual(
            cases, ["decode", "logical", "mixed", "prefill"],
        )

    def test_default_case_order_d_m_p_l(self):
        """Operator-readable order: DECODE → MIXED → PREFILL →
        LOGICAL (cheap-first, complex-last)."""
        combos = list(enumerate_all_combos(**_kwargs()))
        seen_order = [tk["case"] for tk, _ in combos]
        # Each case is contiguous in the iteration.
        self.assertEqual(seen_order[0], "decode")
        self.assertEqual(seen_order[-1], "logical")

    def test_case_filter_selects_subset(self):
        combos = list(enumerate_all_combos(
            **_kwargs(), cases=("decode", "mixed"),
        ))
        cases = sorted({tk["case"] for tk, _ in combos})
        self.assertEqual(cases, ["decode", "mixed"])

    def test_unknown_case_raises(self):
        with self.assertRaisesRegex(ValueError, "bogus"):
            list(enumerate_all_combos(
                **_kwargs(), cases=("bogus",),
            ))

    def test_all_cases_constant_is_four(self):
        self.assertEqual(set(ALL_CASES),
                         {"decode", "mixed", "prefill", "logical"})


if __name__ == "__main__":
    unittest.main()
