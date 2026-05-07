# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for tpu_inference.runner.subseq_planner.

The planner is a pure-stdlib helper, so these tests run without JAX / TPU.
Coverage focuses on:

  - DECODE / PREFILL / MIXED routing for each (n_R, K_kernel) shape.
  - Multiple real-request interleaving and ordering invariants.
  - Backwards-compat: K_kernel=None matches todays single-MIXED-per-req
    behaviour exactly.
  - Sizing bound computation.

Wire-up correctness (planner output -> kernel prefetch arrays) is out of
scope here; that lands in the follow-up PR with its own integration tests.
"""

import importlib.util
import os
import unittest

# Load subseq_planner directly via importlib to avoid pulling
# tpu_inference/__init__.py (which transitively imports vllm/jax/torch).
# The planner is intentionally pure stdlib, so this works on any box —
# including dev machines without the TPU runtime stack.
_PLANNER_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "tpu_inference", "runner", "subseq_planner.py",
)
_spec = importlib.util.spec_from_file_location(
    "subseq_planner_under_test", _PLANNER_PATH)
_mod = importlib.util.module_from_spec(_spec)
# Register in sys.modules BEFORE exec — dataclasses looks up
# cls.__module__ in sys.modules during class construction. Without this,
# AttributeError: 'NoneType' object has no attribute '__dict__'.
import sys
sys.modules["subseq_planner_under_test"] = _mod
_spec.loader.exec_module(_mod)
StepPlan = _mod.StepPlan
SubSeqEntry = _mod.SubSeqEntry
SubSeqKind = _mod.SubSeqKind
plan_step = _mod.plan_step
required_max_num_subseqs = _mod.required_max_num_subseqs
SMEM_LIMIT_BYTES = _mod.SMEM_LIMIT_BYTES
compute_smem_bytes = _mod.compute_smem_bytes
validate_smem_fits = _mod.validate_smem_fits
max_M_under_smem = _mod.max_M_under_smem


class TestPlanStepDecodeOnly(unittest.TestCase):

    def test_single_decode(self):
        plan = plan_step(
            num_scheduled_tokens={"R1": 1},
            prior_kv_lens={"R1": 100},
            req_id_order=["R1"],
            K_kernel=256,
        )
        self.assertEqual(plan.num_decode, 1)
        self.assertEqual(plan.num_prefill, 0)
        self.assertEqual(plan.num_mixed, 0)
        self.assertEqual(plan.request_distribution, (1, 1, 1))
        self.assertEqual(plan.subseqs[0].kind, SubSeqKind.DECODE)
        self.assertEqual(plan.subseqs[0].q_len, 1)
        self.assertEqual(plan.subseqs[0].kv_len_at_end, 101)

    def test_many_decodes_ordered_by_req_id_order(self):
        plan = plan_step(
            num_scheduled_tokens={"R1": 1, "R2": 1, "R3": 1},
            prior_kv_lens={"R1": 50, "R2": 100, "R3": 200},
            req_id_order=["R3", "R1", "R2"],   # custom order
            K_kernel=256,
        )
        self.assertEqual(plan.num_decode, 3)
        self.assertEqual(
            [s.real_req_id for s in plan.subseqs],
            ["R3", "R1", "R2"],
        )


class TestPlanStepPrefillChunking(unittest.TestCase):

    def test_exact_multiple_of_K_kernel(self):
        # 768 = 3 * 256, exact — no MIXED remainder.
        plan = plan_step(
            num_scheduled_tokens={"R1": 768},
            prior_kv_lens={"R1": 0},
            req_id_order=["R1"],
            K_kernel=256,
        )
        self.assertEqual(plan.num_decode, 0)
        self.assertEqual(plan.num_prefill, 3)
        self.assertEqual(plan.num_mixed, 0)
        self.assertEqual(plan.request_distribution, (0, 3, 3))
        for i, s in enumerate(plan.subseqs):
            self.assertEqual(s.kind, SubSeqKind.PREFILL)
            self.assertEqual(s.q_len, 256)
            self.assertEqual(s.q_offset_in_real_req, i * 256)
            self.assertEqual(s.kv_len_at_end, (i + 1) * 256)

    def test_with_remainder(self):
        # 800 = 3 * 256 + 32; expect 3 PREFILL + 1 MIXED with q_len=32.
        plan = plan_step(
            num_scheduled_tokens={"R1": 800},
            prior_kv_lens={"R1": 0},
            req_id_order=["R1"],
            K_kernel=256,
        )
        self.assertEqual(plan.num_prefill, 3)
        self.assertEqual(plan.num_mixed, 1)
        # kv_len_at_end values increase monotonically across PREFILL chunks
        # so the @pl.loops sequential KV writes are correctly ordered.
        prefill_subseqs = plan.subseqs[:3]
        for i, s in enumerate(prefill_subseqs):
            self.assertEqual(s.q_len, 256)
            self.assertEqual(s.kv_len_at_end, (i + 1) * 256)
        # MIXED entry picks up the 32-token tail.
        mixed = plan.subseqs[3]
        self.assertEqual(mixed.kind, SubSeqKind.MIXED)
        self.assertEqual(mixed.q_offset_in_real_req, 768)
        self.assertEqual(mixed.q_len, 32)
        self.assertEqual(mixed.kv_len_at_end, 800)

    def test_short_prefill_below_K_kernel(self):
        # 100 < K_kernel=256 — entire n_R lands in MIXED, no PREFILL chunks.
        plan = plan_step(
            num_scheduled_tokens={"R1": 100},
            prior_kv_lens={"R1": 0},
            req_id_order=["R1"],
            K_kernel=256,
        )
        self.assertEqual(plan.num_prefill, 0)
        self.assertEqual(plan.num_mixed, 1)
        self.assertEqual(plan.subseqs[0].kind, SubSeqKind.MIXED)
        self.assertEqual(plan.subseqs[0].q_len, 100)

    def test_kv_len_includes_prior(self):
        # Mid-prefill request: prior_kv_len > 0. kv_len_at_end accumulates
        # off the prior, not from 0.
        plan = plan_step(
            num_scheduled_tokens={"R1": 512},
            prior_kv_lens={"R1": 1000},   # already had 1000 KVs
            req_id_order=["R1"],
            K_kernel=256,
        )
        self.assertEqual(plan.num_prefill, 2)
        self.assertEqual(plan.subseqs[0].kv_len_at_end, 1256)
        self.assertEqual(plan.subseqs[1].kv_len_at_end, 1512)


class TestPlanStepMultiRequest(unittest.TestCase):

    def test_decode_prefill_mixed_interleave(self):
        # R1: 800 prefill (3 chunks + 32 remainder)
        # R2: 1024 prefill (4 chunks, no remainder)
        # R3: 1 decode
        # R4: 100 short prefill (no chunks; entirely MIXED)
        plan = plan_step(
            num_scheduled_tokens={"R1": 800, "R2": 1024, "R3": 1, "R4": 100},
            prior_kv_lens={"R1": 0, "R2": 0, "R3": 50, "R4": 0},
            req_id_order=["R1", "R2", "R3", "R4"],
            K_kernel=256,
        )
        # Counts.
        self.assertEqual(plan.num_decode, 1)
        self.assertEqual(plan.num_prefill, 3 + 4)   # R1 chunks + R2 chunks
        self.assertEqual(plan.num_mixed, 1 + 1)     # R1 remainder + R4 entire
        self.assertEqual(plan.num_total, 1 + 7 + 2)
        self.assertEqual(plan.request_distribution, (1, 8, 10))

        # Ordering: D, then P, then M. Within P, R1s 3 chunks come first
        # (matching req_id_order), then R2s 4 chunks. Within M, R1s
        # remainder before R4s short prefill.
        kinds = [s.kind for s in plan.subseqs]
        self.assertEqual(kinds, [
            SubSeqKind.DECODE,                                    # R3
            SubSeqKind.PREFILL, SubSeqKind.PREFILL, SubSeqKind.PREFILL,  # R1 chunks
            SubSeqKind.PREFILL, SubSeqKind.PREFILL, SubSeqKind.PREFILL, SubSeqKind.PREFILL,  # R2 chunks
            SubSeqKind.MIXED, SubSeqKind.MIXED,                  # R1 tail, R4
        ])

        # Verify chunks of the same request are CONTIGUOUS in PREFILL
        # bucket (so the kernels @pl.loop processes them sequentially
        # and chunk N+1 sees chunk Ns KV writes).
        prefill = [s for s in plan.subseqs if s.kind == SubSeqKind.PREFILL]
        # First 3 are R1, next 4 are R2.
        self.assertEqual([s.real_req_id for s in prefill],
                         ["R1", "R1", "R1", "R2", "R2", "R2", "R2"])
        # And their q_offset_in_real_req increases monotonically per real req:
        self.assertEqual([s.q_offset_in_real_req for s in prefill[:3]],
                         [0, 256, 512])
        self.assertEqual([s.q_offset_in_real_req for s in prefill[3:]],
                         [0, 256, 512, 768])

    def test_zero_scheduled_tokens_emits_no_subseqs(self):
        plan = plan_step(
            num_scheduled_tokens={"R1": 0, "R2": 256},
            prior_kv_lens={"R1": 50, "R2": 0},
            req_id_order=["R1", "R2"],
            K_kernel=256,
        )
        self.assertEqual(plan.num_total, 1)
        self.assertEqual(plan.subseqs[0].real_req_id, "R2")


class TestPlanStepBackwardsCompat(unittest.TestCase):
    """When K_kernel is None, the planner must reproduce todays behaviour:
    decode goes to DECODE, everything else lands in a single MIXED sub-seq
    per request — no chunking."""

    def test_K_kernel_none_disables_chunking(self):
        plan = plan_step(
            num_scheduled_tokens={"R1": 800, "R2": 1, "R3": 100},
            prior_kv_lens={"R1": 0, "R2": 50, "R3": 0},
            req_id_order=["R1", "R2", "R3"],
            K_kernel=None,
        )
        self.assertEqual(plan.num_decode, 1)
        self.assertEqual(plan.num_prefill, 0)
        self.assertEqual(plan.num_mixed, 2)
        # Whole 800 of R1 is a single MIXED sub-seq, not 3 PREFILL chunks.
        r1_mixed = next(s for s in plan.subseqs
                        if s.real_req_id == "R1")
        self.assertEqual(r1_mixed.kind, SubSeqKind.MIXED)
        self.assertEqual(r1_mixed.q_len, 800)


class TestPlanStepValidation(unittest.TestCase):

    def test_K_kernel_le_1_rejected(self):
        for bad in (0, 1, -5):
            with self.assertRaises(ValueError):
                plan_step(
                    num_scheduled_tokens={"R1": 256},
                    prior_kv_lens={"R1": 0},
                    req_id_order=["R1"],
                    K_kernel=bad,
                )

    def test_missing_prior_kv_len_rejected(self):
        with self.assertRaises(ValueError):
            plan_step(
                num_scheduled_tokens={"R1": 256},
                prior_kv_lens={},   # missing
                req_id_order=["R1"],
                K_kernel=256,
            )

    def test_missing_in_scheduled_rejected(self):
        with self.assertRaises(ValueError):
            plan_step(
                num_scheduled_tokens={},
                prior_kv_lens={"R1": 0},
                req_id_order=["R1"],
                K_kernel=256,
            )


class TestRequiredMaxNumSubseqs(unittest.TestCase):

    def test_exact_multiple(self):
        # MAX_NUM_SEQS=128, K_sched_cap=2048, K_kernel=256.
        # ceil(2048/256) = 8 chunks per req max, plus 1 MIXED remainder slot.
        # Total: 128 * 8 + 128 = 1152.
        self.assertEqual(
            required_max_num_subseqs(
                max_num_seqs=128, K_sched_cap=2048, K_kernel=256),
            1152,
        )

    def test_non_multiple(self):
        # K_sched_cap=2000, K_kernel=256: ceil(2000/256) = 8 chunks max
        # (since the last chunk could be partial; we still bound at 8 to
        # cover the "n_R = 8 * K_kernel - epsilon" case).
        # Plus 1 MIXED slot per req.
        # 128 * 8 + 128 = 1152.
        self.assertEqual(
            required_max_num_subseqs(
                max_num_seqs=128, K_sched_cap=2000, K_kernel=256),
            1152,
        )

    def test_K_sched_smaller_than_K_kernel(self):
        # If someone misconfigures K_sched_cap < K_kernel, the formula
        # still bounds at 1 chunk per req (ceil(K_sched_cap/K_kernel) = 1)
        # plus 1 MIXED remainder slot — giving 2x max_num_seqs.
        self.assertEqual(
            required_max_num_subseqs(
                max_num_seqs=128, K_sched_cap=100, K_kernel=256),
            128 + 128,
        )

    def test_invalid_args_rejected(self):
        with self.assertRaises(ValueError):
            required_max_num_subseqs(max_num_seqs=128, K_sched_cap=2048,
                                     K_kernel=1)
        with self.assertRaises(ValueError):
            required_max_num_subseqs(max_num_seqs=128, K_sched_cap=0,
                                     K_kernel=256)
        with self.assertRaises(ValueError):
            required_max_num_subseqs(max_num_seqs=0, K_sched_cap=2048,
                                     K_kernel=256)


class TestSmemBudget(unittest.TestCase):
    """SMEM is the binding constraint on max_num_subseqs because
    page_indices_ref dominates the prefetch-array footprint. These tests
    pin down the formula and the practical caps for the workloads we
    actually care about (8B/8K and 70B/32K)."""

    def test_smem_limit_value_matches_kernel_tuner(self):
        # Kept in sync with rpa_v3_kernel_tuner.py:96
        # SMEM_LIMIT_BYTES = 0.9 * 1024 * 1024
        self.assertEqual(SMEM_LIMIT_BYTES, int(0.9 * 1024 * 1024))

    def test_compute_smem_dominated_by_page_indices(self):
        # max_num_seqs=128, pages_per_seq=64 → page_indices is
        # 128*64 i32s ≈ 32 KB; other prefetches ≈ 1 KB.
        bytes_used = compute_smem_bytes(max_num_seqs=128, pages_per_seq=64)
        # Sanity: dominant term * 4 bytes = 32 KB; total within +20%.
        self.assertGreater(bytes_used, 32 * 1024)
        self.assertLess(bytes_used, 40 * 1024)

    def test_compute_smem_scales_linearly_in_max_num_seqs(self):
        # Doubling max_num_seqs roughly doubles SMEM.
        small = compute_smem_bytes(max_num_seqs=128, pages_per_seq=64)
        big = compute_smem_bytes(max_num_seqs=256, pages_per_seq=64)
        ratio = big / small
        self.assertGreater(ratio, 1.9)
        self.assertLess(ratio, 2.1)

    def test_today_8b_8k_fits(self):
        # Production 8B/8K config: max_num_seqs=128, pages_per_seq=64.
        # Should comfortably fit (~36 KB).
        validate_smem_fits(max_num_subseqs=128, pages_per_seq=64)

    def test_naive_8x_bump_overflows(self):
        # 8B/8K bumped to max_num_subseqs=128*32=4096 (M_max=32 nominal):
        # ~1 MB, over 0.9 MB limit.
        with self.assertRaisesRegex(ValueError, "SMEM budget exceeded"):
            validate_smem_fits(max_num_subseqs=128 * 32, pages_per_seq=64)

    def test_70b_32k_with_64_seqs_smaller_M_works(self):
        # 70B/32K: pages_per_seq = ceil(32768/128) = 256.
        # MAX_NUM_SEQS=64. M=8 → max_num_subseqs = 64*9 = 576 → ~580 KB.
        # Should fit.
        validate_smem_fits(max_num_subseqs=64 * 9, pages_per_seq=256)
        # But M=16 → 64*17 = 1088 → ~1.1 MB → should NOT fit.
        with self.assertRaises(ValueError):
            validate_smem_fits(max_num_subseqs=64 * 17, pages_per_seq=256)

    def test_max_M_under_smem_8b_8k(self):
        # 8B/8K, MAX_NUM_SEQS=128, pages_per_seq=64.
        # Practical M_max should be in the 20s (1MB / (128*64*4) ≈ 32,
        # minus headroom for other prefetch arrays).
        M = max_M_under_smem(max_num_seqs=128, pages_per_seq=64)
        self.assertGreaterEqual(M, 20)
        self.assertLessEqual(M, 32)
        # Verify the boundary: M fits, M+1 doesnt.
        validate_smem_fits(max_num_subseqs=128 * (M + 1),
                           pages_per_seq=64)
        with self.assertRaises(ValueError):
            validate_smem_fits(max_num_subseqs=128 * (M + 2),
                               pages_per_seq=64)

    def test_max_M_under_smem_70b_32k(self):
        # 70B/32K, MAX_NUM_SEQS=64, pages_per_seq=256.
        # Practical M_max should be in the low teens.
        M = max_M_under_smem(max_num_seqs=64, pages_per_seq=256)
        self.assertGreaterEqual(M, 8)
        self.assertLessEqual(M, 14)

    def test_max_M_under_smem_70b_32k_mns_128_more_constrained(self):
        # Same workload but MNS=128: max_num_seqs is 2x larger →
        # achievable M roughly halves.
        M = max_M_under_smem(max_num_seqs=128, pages_per_seq=256)
        self.assertGreaterEqual(M, 4)
        self.assertLessEqual(M, 7)

    def test_validate_smem_fits_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            validate_smem_fits(max_num_subseqs=0, pages_per_seq=64)
        with self.assertRaises(ValueError):
            validate_smem_fits(max_num_subseqs=128, pages_per_seq=0)


if __name__ == "__main__":
    unittest.main()
