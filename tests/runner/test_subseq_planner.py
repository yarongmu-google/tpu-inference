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

import dataclasses
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
DecoupledKConfig = _mod.DecoupledKConfig
evaluate_decoupled_k_config = _mod.evaluate_decoupled_k_config
build_prior_kv_lens = _mod.build_prior_kv_lens
build_iter_prefetches = _mod.build_iter_prefetches
IterPrefetches = _mod.IterPrefetches


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
        self.assertEqual(plan.subseqs[0].q_offset_in_real_req, 0)
        self.assertEqual(plan.subseqs[0].kv_len_at_end, 1256)
        self.assertEqual(plan.subseqs[1].q_offset_in_real_req, 256)
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

    def test_max_M_under_smem_safety_cap_warns(self):
        # If someone passes a huge smem_limit_bytes (e.g. sys.maxsize or
        # otherwise-misconfigured), the linear search would never find
        # the bytes_used > limit terminator. The safety cap stops the
        # search at 1024 and emits a warning. Verify both behaviours.
        import warnings as _warnings
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            M = max_M_under_smem(max_num_seqs=128, pages_per_seq=64,
                                 smem_limit_bytes=10**18)
        # Should hit the cap and return cap - 1.
        self.assertEqual(M, 1023)
        self.assertTrue(any("safety cap" in str(w.message)
                            for w in caught),
                        f"Expected 'safety cap' warning; got: "
                        f"{[str(w.message) for w in caught]}")


class TestEvaluateDecoupledKConfig(unittest.TestCase):
    """Tests for the runner-init resolution of the decoupled-K config.

    The function is the single place where K_sched/K_kernel/SMEM-budget
    inputs are validated and turned into a DecoupledKConfig. The runner
    plumbing just calls it; these tests cover the math without requiring
    the JAX/TPU stack."""

    def _8b_8k_inputs(self, *, K_kernel=256, K_sched=2048,
                       max_num_batched_tokens=8192, max_num_seqs=128,
                       max_model_len=8192, page_size=128):
        # Standard 8B/8K shape that today comfortably fits SMEM.
        return dict(K_kernel=K_kernel, K_sched=K_sched,
                    max_num_batched_tokens=max_num_batched_tokens,
                    max_num_seqs=max_num_seqs, max_model_len=max_model_len,
                    page_size=page_size)

    def test_typical_8b_8k_fits(self):
        # K_sched=2048, K_kernel=256 -> requested_M=8.
        # max_num_seqs=128, pages_per_seq=64 -> achievable_M in low-20s
        # to high-20s depending on the SMEM formulas alignment overhead.
        cfg = evaluate_decoupled_k_config(**self._8b_8k_inputs())
        self.assertEqual(cfg.K_kernel, 256)
        self.assertEqual(cfg.K_sched_effective, 2048)
        self.assertEqual(cfg.requested_M, 8)
        # Tight bound rather than just >=8: the formula is deterministic
        # and we want this test to flag SMEM-formula regressions.
        self.assertIn(cfg.achievable_M, range(20, 30))
        self.assertEqual(cfg.effective_M, 8)
        self.assertEqual(cfg.effective_K_sched_cap, 2048)
        self.assertEqual(cfg.pages_per_seq, 64)
        self.assertFalse(cfg.requested_exceeds_achievable)

    def test_clamps_when_K_sched_exceeds_achievable(self):
        # Force K_sched=K_kernel*64=16384, lift MNB to match so the MNB
        # clamp doesnt fire here (the MNB-clamp is exercised in its own
        # test below). At 8B/8K MNS=128, achievable_M ~26 < 64 ->
        # clamps to achievable.
        cfg = evaluate_decoupled_k_config(
            **self._8b_8k_inputs(K_sched=256 * 64,
                                 max_num_batched_tokens=256 * 64))
        self.assertEqual(cfg.K_sched_effective, 256 * 64)
        self.assertEqual(cfg.requested_M, 64)
        self.assertLess(cfg.achievable_M, 64)
        self.assertEqual(cfg.effective_M, cfg.achievable_M)
        self.assertEqual(cfg.effective_K_sched_cap,
                         cfg.achievable_M * 256)
        self.assertTrue(cfg.requested_exceeds_achievable)

    def test_K_sched_zero_falls_back_to_max_num_batched_tokens(self):
        # K_sched=0 (vLLM default for "no per-request cap"): bound by
        # MNB=8192. requested_M = ceil(8192/256) = 32.
        cfg = evaluate_decoupled_k_config(
            **self._8b_8k_inputs(K_sched=0))
        self.assertEqual(cfg.K_sched_effective, 8192)   # = MNB
        self.assertEqual(cfg.requested_M, 32)
        # 8B/8K: achievable_M is in mid-20s, so 32 should clamp.
        self.assertEqual(cfg.effective_M, cfg.achievable_M)
        self.assertEqual(cfg.effective_K_sched_cap,
                         cfg.achievable_M * 256)

    def test_K_sched_above_MNB_clamps_to_MNB(self):
        # User mis-sets K_sched above MNB. vLLM cant grant more than MNB
        # for a single requests step, so requested_M should reflect MNB,
        # not the user-provided K_sched.
        cfg = evaluate_decoupled_k_config(
            **self._8b_8k_inputs(K_sched=16384,
                                 max_num_batched_tokens=8192))
        self.assertEqual(cfg.K_sched_effective, 8192)   # clamped to MNB
        self.assertEqual(cfg.requested_M, 32)           # ceil(8192/256)
        # Below MNB: no clamp; uses K_sched directly.
        cfg = evaluate_decoupled_k_config(
            **self._8b_8k_inputs(K_sched=4096,
                                 max_num_batched_tokens=8192))
        self.assertEqual(cfg.K_sched_effective, 4096)
        self.assertEqual(cfg.requested_M, 16)

    def test_k_sched_below_k_kernel_yields_M_1(self):
        # K_sched=100 < K_kernel=256 -> requested_M = ceil(100/256) = 1.
        # Effectively no chunking benefit, but valid.
        cfg = evaluate_decoupled_k_config(
            **self._8b_8k_inputs(K_sched=100))
        self.assertEqual(cfg.requested_M, 1)
        self.assertEqual(cfg.effective_M, 1)
        self.assertEqual(cfg.effective_K_sched_cap, 256)

    def test_70b_32k_mns128_achievable_M_low(self):
        # 70B/32K with MNS=128 has very tight SMEM (pages_per_seq=256).
        cfg = evaluate_decoupled_k_config(
            K_kernel=256, K_sched=4096, max_num_batched_tokens=8192,
            max_num_seqs=128, max_model_len=32768, page_size=128)
        self.assertEqual(cfg.requested_M, 16)
        # SMEM budget caps achievable_M at ~6 here.
        self.assertGreaterEqual(cfg.achievable_M, 4)
        self.assertLessEqual(cfg.achievable_M, 7)
        self.assertEqual(cfg.effective_M, cfg.achievable_M)
        self.assertTrue(cfg.requested_exceeds_achievable)

    def test_pages_per_seq_ceiling(self):
        # max_model_len=8193 with page_size=128 -> 65 pages (ceiling).
        cfg = evaluate_decoupled_k_config(
            K_kernel=256, K_sched=512, max_num_batched_tokens=8192,
            max_num_seqs=4, max_model_len=8193, page_size=128)
        self.assertEqual(cfg.pages_per_seq, 65)

    def test_smem_budget_argument_changes_achievable(self):
        # Smaller SMEM budget -> smaller achievable_M.
        small = evaluate_decoupled_k_config(
            **self._8b_8k_inputs(), smem_budget=128 * 1024)  # 128 KB
        large = evaluate_decoupled_k_config(
            **self._8b_8k_inputs(), smem_budget=900 * 1024)  # ~900 KB
        self.assertLess(small.achievable_M, large.achievable_M)

    def test_hard_guard_M1_does_not_fit_raises(self):
        # Force a config where even M=1 (no chunking) fails SMEM: huge
        # max_num_seqs * huge pages_per_seq. The error message should be
        # the M=1-specific actionable wording (mentions max_num_seqs /
        # page_size / max_model_len), NOT the generic
        # validate_smem_fits message that suggests reducing K_sched_cap
        # (which is meaningless at this regime).
        with self.assertRaisesRegex(
                ValueError,
                "Workload geometry incompatible with hardware SMEM"):
            evaluate_decoupled_k_config(
                K_kernel=256, K_sched=512, max_num_batched_tokens=8192,
                max_num_seqs=1000, max_model_len=131072, page_size=128)

    def test_invalid_inputs_raise(self):
        for bad in ({"K_kernel": 0}, {"K_kernel": 1},
                    {"max_num_batched_tokens": 0},
                    {"max_num_seqs": 0},
                    {"max_model_len": 0},
                    {"page_size": 0}):
            with self.subTest(bad=bad):
                inputs = self._8b_8k_inputs()
                inputs.update(bad)
                with self.assertRaises(ValueError):
                    evaluate_decoupled_k_config(**inputs)


class TestBuildPriorKvLens(unittest.TestCase):
    """Tests for the data-marshalling helper extracted from
    PersistentBatchManager.compute_step_plan. The helper takes the
    inputs that come from the runtime stack (input_batch + scheduler
    output) and produces the prior_kv_lens dict plan_step expects.

    Inputs are deliberately stdlib types (dict, list, list-as-int-array)
    so this is fully testable without the JAX/vLLM stack."""

    def test_basic_three_reqs(self):
        prior = build_prior_kv_lens(
            req_ids_in_order=["R1", "R2", "R3"],
            num_scheduled_tokens={"R1": 256, "R2": 1, "R3": 800},
            req_id_to_index={"R1": 0, "R2": 1, "R3": 2},
            num_computed_tokens_cpu=[100, 200, 300],
        )
        self.assertEqual(prior, {"R1": 100, "R2": 200, "R3": 300})

    def test_skips_zero_scheduled(self):
        # Reqs with n_R == 0 must not appear (they wouldnt produce
        # sub-seqs anyway; their absence keeps plan_steps prior_kv_lens
        # contract — every key in num_scheduled_tokens with n_R > 0
        # has a corresponding key here).
        prior = build_prior_kv_lens(
            req_ids_in_order=["R1", "R2"],
            num_scheduled_tokens={"R1": 256, "R2": 0},
            req_id_to_index={"R1": 0, "R2": 1},
            num_computed_tokens_cpu=[100, 200],
        )
        self.assertEqual(prior, {"R1": 100})

    def test_skips_negative_scheduled(self):
        # Defensive: negative n_R never reaches us in practice but the
        # helper treats it as "skip" rather than crashing.
        prior = build_prior_kv_lens(
            req_ids_in_order=["R1"],
            num_scheduled_tokens={"R1": -5},
            req_id_to_index={"R1": 0},
            num_computed_tokens_cpu=[100],
        )
        self.assertEqual(prior, {})

    def test_skips_missing_in_scheduled(self):
        # req_ids_in_order may contain reqs not present in
        # scheduler_output.num_scheduled_tokens (race during update_states).
        # We skip them rather than KeyError.
        prior = build_prior_kv_lens(
            req_ids_in_order=["R1", "R2"],
            num_scheduled_tokens={"R1": 256},
            req_id_to_index={"R1": 0, "R2": 1},
            num_computed_tokens_cpu=[100, 200],
        )
        self.assertEqual(prior, {"R1": 100})

    def test_skips_unknown_index(self):
        # req_id_to_index race: a req in req_ids_in_order may not yet be
        # registered. Skip rather than IndexError downstream.
        prior = build_prior_kv_lens(
            req_ids_in_order=["R1", "R_new"],
            num_scheduled_tokens={"R1": 256, "R_new": 256},
            req_id_to_index={"R1": 0},                # R_new not yet here
            num_computed_tokens_cpu=[100],
        )
        self.assertEqual(prior, {"R1": 100})

    def test_int_conversion(self):
        # num_computed_tokens_cpu is typically a numpy array of int32;
        # the helper converts to Python int so plan_step gets clean
        # integers for arithmetic. Test with a non-int input to verify
        # the conversion happens.
        class FakeIndexable:
            def __getitem__(self, idx):
                # Returns a "numpy-int-like" — int() should coerce.
                return float(100 + idx * 50)
        prior = build_prior_kv_lens(
            req_ids_in_order=["R1", "R2"],
            num_scheduled_tokens={"R1": 256, "R2": 256},
            req_id_to_index={"R1": 0, "R2": 1},
            num_computed_tokens_cpu=FakeIndexable(),
        )
        self.assertEqual(prior, {"R1": 100, "R2": 150})
        # And the values must be int (plan_step does arithmetic on them).
        self.assertIsInstance(prior["R1"], int)

    def test_ordering_preserved_in_dict_keys(self):
        # Pythons dicts preserve insertion order; the helper iterates
        # req_ids_in_order, so the resulting dicts key order should
        # match. plan_step uses req_id_order separately for ordering,
        # but downstream consumers might iterate the dict directly.
        prior = build_prior_kv_lens(
            req_ids_in_order=["Z", "A", "M"],
            num_scheduled_tokens={"A": 1, "Z": 1, "M": 1},
            req_id_to_index={"A": 1, "Z": 0, "M": 2},
            num_computed_tokens_cpu=[10, 20, 30],
        )
        self.assertEqual(list(prior.keys()), ["Z", "A", "M"])

    def test_empty_inputs(self):
        prior = build_prior_kv_lens(
            req_ids_in_order=[],
            num_scheduled_tokens={},
            req_id_to_index={},
            num_computed_tokens_cpu=[],
        )
        self.assertEqual(prior, {})


class TestDecoupledKConfigDataclass(unittest.TestCase):

    def test_requested_exceeds_achievable_property(self):
        cfg = DecoupledKConfig(
            K_kernel=256,
            K_sched_effective=2048,
            requested_M=10, achievable_M=4, effective_M=4,
            effective_K_sched_cap=1024, smem_budget=900 * 1024,
            pages_per_seq=64)
        self.assertTrue(cfg.requested_exceeds_achievable)

        cfg = DecoupledKConfig(
            K_kernel=256,
            K_sched_effective=2048,
            requested_M=4, achievable_M=10, effective_M=4,
            effective_K_sched_cap=1024, smem_budget=900 * 1024,
            pages_per_seq=64)
        self.assertFalse(cfg.requested_exceeds_achievable)

    def test_frozen(self):
        cfg = DecoupledKConfig(
            K_kernel=256,
            K_sched_effective=2048,
            requested_M=8, achievable_M=8, effective_M=8,
            effective_K_sched_cap=2048, smem_budget=900 * 1024,
            pages_per_seq=64)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            cfg.requested_M = 99   # type: ignore[misc]


class TestBuildIterPrefetches(unittest.TestCase):
    """Tests for the iter-keyed prefetch builder. Pairs a StepPlan with
    the persistent batchs req_id_to_index and emits the
    (phys_seq_indices, q_offsets) lists the kernel reads on the LOGICAL
    dispatch."""

    def test_decode_plus_chunked_prefill_happy_path(self):
        # Two reqs: R1 is decoding (1 token), R2 has 12 prefill tokens
        # at K_kernel=4, so 3 PREFILL chunks (q_offsets 0, 4, 8).
        # Persistent-batch ordering puts R2 in slot 0 and R1 in slot 1
        # to verify phys_seq_indices is NOT just identity.
        plan = plan_step(
            num_scheduled_tokens={"R1": 1, "R2": 12},
            prior_kv_lens={"R1": 256, "R2": 0},
            req_id_order=["R2", "R1"],
            K_kernel=4,
        )
        self.assertEqual(plan.num_decode, 1)
        self.assertEqual(plan.num_prefill, 3)
        self.assertEqual(plan.num_mixed, 0)

        out = build_iter_prefetches(
            plan=plan,
            req_id_to_index={"R2": 0, "R1": 1},
            max_num_seqs=4,
            max_num_subseqs=8,
            static_q_len=4,
            num_scheduled_tokens={"R1": 1, "R2": 12},
        )
        # Iter order: DECODE (R1) -> PREFILL (R2 x3).
        self.assertEqual(out.phys_seq_indices[:4], [1, 0, 0, 0])
        self.assertEqual(out.q_offsets[:4], [0, 0, 4, 8])
        # Tail beyond num_total is zero-padded; never read by kernel.
        self.assertEqual(out.phys_seq_indices[4:], [0, 0, 0, 0])
        self.assertEqual(out.q_offsets[4:], [0, 0, 0, 0])

    def test_returns_iter_prefetches_with_correct_lengths(self):
        plan = plan_step(
            num_scheduled_tokens={"R1": 1},
            prior_kv_lens={"R1": 0},
            req_id_order=["R1"],
            K_kernel=4,
        )
        out = build_iter_prefetches(
            plan=plan,
            req_id_to_index={"R1": 0},
            max_num_seqs=2,
            max_num_subseqs=16,
            static_q_len=None,
        )
        self.assertIsInstance(out, IterPrefetches)
        self.assertEqual(len(out.phys_seq_indices), 16)
        self.assertEqual(len(out.q_offsets), 16)

    def test_coupled_k_path_static_q_len_none_skips_bounds_check(self):
        # On the coupled-K path the helper is still callable for the iter
        # arrays alone — bounds-check is skipped, num_scheduled_tokens
        # may be omitted.
        plan = plan_step(
            num_scheduled_tokens={"R1": 5},
            prior_kv_lens={"R1": 0},
            req_id_order=["R1"],
            K_kernel=None,   # coupled-K: one MIXED per non-decode req
        )
        self.assertEqual(plan.num_mixed, 1)

        out = build_iter_prefetches(
            plan=plan,
            req_id_to_index={"R1": 0},
            max_num_seqs=2,
            max_num_subseqs=4,
            static_q_len=None,
            num_scheduled_tokens=None,    # not required when static is None
        )
        self.assertEqual(out.phys_seq_indices[:1], [0])
        self.assertEqual(out.q_offsets[:1], [0])

    def test_empty_plan_returns_zero_padded(self):
        plan = StepPlan(subseqs=(), num_decode=0, num_prefill=0,
                        num_mixed=0)
        out = build_iter_prefetches(
            plan=plan,
            req_id_to_index={},
            max_num_seqs=4,
            max_num_subseqs=8,
            static_q_len=None,
        )
        self.assertEqual(out.phys_seq_indices, [0] * 8)
        self.assertEqual(out.q_offsets, [0] * 8)

    def test_oversized_plan_rejected(self):
        # 5 sub-seqs but max_num_subseqs=4 — would silently overflow the
        # prefetch arrays if not caught here. required_max_num_subseqs()
        # is supposed to size for the worst case; under-sizing means
        # the workload exceeded the runner-init bound and we should
        # fail loud.
        plan = plan_step(
            num_scheduled_tokens={"R1": 10},
            prior_kv_lens={"R1": 0},
            req_id_order=["R1"],
            K_kernel=2,   # 10 / 2 = 5 PREFILL chunks
        )
        self.assertEqual(plan.num_total, 5)
        with self.assertRaisesRegex(ValueError, "max_num_subseqs"):
            build_iter_prefetches(
                plan=plan,
                req_id_to_index={"R1": 0},
                max_num_seqs=2,
                max_num_subseqs=4,
                static_q_len=2,
                num_scheduled_tokens={"R1": 10},
            )

    def test_missing_req_id_rejected(self):
        plan = plan_step(
            num_scheduled_tokens={"R1": 1},
            prior_kv_lens={"R1": 0},
            req_id_order=["R1"],
            K_kernel=4,
        )
        with self.assertRaisesRegex(ValueError, "out of sync"):
            build_iter_prefetches(
                plan=plan,
                req_id_to_index={},   # R1 absent
                max_num_seqs=4,
                max_num_subseqs=4,
                static_q_len=None,
            )

    def test_phys_index_out_of_range_rejected(self):
        plan = plan_step(
            num_scheduled_tokens={"R1": 1},
            prior_kv_lens={"R1": 0},
            req_id_order=["R1"],
            K_kernel=4,
        )
        with self.assertRaisesRegex(ValueError, "outside"):
            build_iter_prefetches(
                plan=plan,
                req_id_to_index={"R1": 5},   # max_num_seqs=4 → out of range
                max_num_seqs=4,
                max_num_subseqs=4,
                static_q_len=None,
            )

    def test_negative_q_offset_rejected(self):
        # plan_step never emits a negative q_offset, so we hand-build a
        # corrupt SubSeqEntry to exercise the defensive guard.
        bad_entry = SubSeqEntry(
            real_req_id="R1",
            kind=SubSeqKind.PREFILL,
            q_offset_in_real_req=-1,
            q_len=4,
            kv_len_at_end=4,
            prior_kv_len=0,
        )
        plan = StepPlan(subseqs=(bad_entry,), num_decode=0, num_prefill=1,
                        num_mixed=0)
        with self.assertRaisesRegex(ValueError, "negative q_offset"):
            build_iter_prefetches(
                plan=plan,
                req_id_to_index={"R1": 0},
                max_num_seqs=2,
                max_num_subseqs=4,
                static_q_len=4,
                num_scheduled_tokens={"R1": 4},
            )

    def test_logical_bounds_check_rejects_overhang(self):
        # PREFILL sub-seq with q_offset + static_q_len > n_R would let
        # the kernel read past the phys reqs scheduled-token slice,
        # producing silent garbage. Hand-build a corrupt plan
        # (plan_step itself wouldnt emit this) to verify the guard
        # fires here rather than at the kernel.
        bad_entry = SubSeqEntry(
            real_req_id="R1",
            kind=SubSeqKind.PREFILL,
            q_offset_in_real_req=0,
            q_len=4,
            kv_len_at_end=4,
            prior_kv_len=0,
        )
        plan = StepPlan(subseqs=(bad_entry,), num_decode=0, num_prefill=1,
                        num_mixed=0)
        with self.assertRaisesRegex(ValueError, "would read past"):
            build_iter_prefetches(
                plan=plan,
                req_id_to_index={"R1": 0},
                max_num_seqs=2,
                max_num_subseqs=4,
                static_q_len=8,                       # 0 + 8 > n_R = 4
                num_scheduled_tokens={"R1": 4},
            )

    def test_logical_requires_num_scheduled_when_static_set(self):
        plan = plan_step(
            num_scheduled_tokens={"R1": 4},
            prior_kv_lens={"R1": 0},
            req_id_order=["R1"],
            K_kernel=4,
        )
        with self.assertRaisesRegex(ValueError, "num_scheduled_tokens"):
            build_iter_prefetches(
                plan=plan,
                req_id_to_index={"R1": 0},
                max_num_seqs=2,
                max_num_subseqs=4,
                static_q_len=4,
                num_scheduled_tokens=None,   # required for LOGICAL bounds check
            )


if __name__ == "__main__":
    unittest.main()
