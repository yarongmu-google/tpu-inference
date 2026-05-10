# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for tools.kernel.tuner.v1.logical_workload_sizing.

The helper exists to keep generate_cases pre-flight pruning and
_build_inputs workload construction in sync. The original bug
(commits 2536219e..c5ade1bb) was a paired-update inconsistency where
the bound check assumed `actual_num_seqs == max_num_seqs` but the
build site scaled it down. Result: 0 cases generated, silently
broken. These tests catch the inconsistency by asserting concrete
shape outcomes for the recipes we actually run.

Lives outside the rpa_v3_kernel_tuner module (no JAX/TPU import
dependency) so it runs on any box.
"""

import unittest

from tools.kernel.tuner.v1.logical_workload_sizing import (
    size_logical_workload)


class TestSizeLogicalWorkload(unittest.TestCase):

    def test_user_recipe_throughput_8b(self):
        """The L kernel quick-look recipe (Llama 3.1 8B, throughput
        workload) MUST produce 6 valid combos, one per max_num_subseqs
        candidate at MNS=128. This is the regression test for the
        0-cases bug from c5ade1bb."""
        # MNS=128, MNB=10275 from prefill_heavy.workload defaults.
        max_num_seqs = 128
        max_num_tokens = 10275
        K = 256
        # max_num_subseqs candidates from the formula:
        # max_num_seqs * (M + 1) for M in {1, 2, 4, 8, 16, 32}.
        mns_lst = [256, 384, 640, 1152, 2176, 4224]

        # Expected sizing per combo, computed by hand and reproduced
        # in the trace at commit c5ade1bb.
        expected = {
            #  mns      (actual,  M, per_q, total_iters)
            256:        (40,       1,   256,  40),
            384:        (20,       2,   512,  40),
            640:        (10,       4,  1024,  40),
            1152:       (5,        8,  2048,  40),
            2176:       (2,       16,  4096,  32),
            4224:       (1,       32,  8192,  32),
        }

        valid_count = 0
        for mns in mns_lst:
            sizing = size_logical_workload(
                max_num_seqs=max_num_seqs,
                max_num_tokens=max_num_tokens, K=K, max_num_subseqs=mns)
            self.assertTrue(sizing.valid,
                            f"mns={mns} should produce a valid combo")
            self.assertEqual(
                (sizing.actual_num_seqs, sizing.m_per_phys,
                 sizing.per_phys_q, sizing.total_iters),
                expected[mns], f"mismatch at mns={mns}")
            # Workload fits the q buffer.
            self.assertLessEqual(
                sizing.actual_num_seqs * sizing.per_phys_q,
                max_num_tokens, f"mns={mns} exceeds q buffer")
            valid_count += 1
        self.assertEqual(valid_count, 6,
                         "All 6 candidate mns values should be valid "
                         "for the throughput L kernel quick-look recipe")

    def test_user_recipe_latency_8b(self):
        """Same recipe against the latency workload (MNS=1) — also 6
        valid combos at smaller mns candidates."""
        max_num_seqs = 1
        max_num_tokens = 8192     # representative MNB for latency sweep
        K = 256
        mns_lst = [2, 3, 5, 9, 17, 33]   # MNS=1 * (M+1) for M in {1,2,4,8,16,32}

        for mns in mns_lst:
            sizing = size_logical_workload(
                max_num_seqs=max_num_seqs,
                max_num_tokens=max_num_tokens, K=K, max_num_subseqs=mns)
            self.assertTrue(sizing.valid,
                            f"latency-recipe mns={mns} should be valid")
            self.assertEqual(sizing.actual_num_seqs, 1,
                             f"MNS=1 means actual_num_seqs=1 (got "
                             f"{sizing.actual_num_seqs} at mns={mns})")
            self.assertEqual(sizing.m_per_phys, mns - 1,
                             f"M_per_phys formula at mns={mns}")

    def test_invalid_mns_below_max_num_seqs(self):
        # max_num_subseqs must be >= max_num_seqs; otherwise the prefetch
        # arrays cant hold one slot per phys req.
        sizing = size_logical_workload(
            max_num_seqs=128, max_num_tokens=10275,
            K=256, max_num_subseqs=64)
        self.assertFalse(sizing.valid)
        self.assertEqual(sizing.actual_num_seqs, 0)

    def test_invalid_per_phys_q_overflows_buffer(self):
        # If even one phys reqs full chunked q (= K * M) overflows
        # max_num_tokens, the workload is impossible to construct.
        # K=8192, M=2 -> per_phys_q = 16384 > MNB=8192.
        sizing = size_logical_workload(
            max_num_seqs=4, max_num_tokens=8192,
            K=8192, max_num_subseqs=12)   # M = 12/4 - 1 = 2
        self.assertFalse(sizing.valid)
        self.assertEqual(sizing.m_per_phys, 2)
        self.assertEqual(sizing.per_phys_q, 16384)
        self.assertEqual(sizing.actual_num_seqs, 0)

    def test_degenerate_mns_equals_max_num_seqs(self):
        # mns == max_num_seqs -> M = max(1, 1 - 1) = max(1, 0) = 1.
        # Each phys gets one chunk; identity-like behaviour.
        sizing = size_logical_workload(
            max_num_seqs=8, max_num_tokens=2048,
            K=128, max_num_subseqs=8)
        self.assertTrue(sizing.valid)
        self.assertEqual(sizing.m_per_phys, 1)
        self.assertEqual(sizing.per_phys_q, 128)
        # actual_num_seqs = min(2048/128, 8) = min(16, 8) = 8.
        self.assertEqual(sizing.actual_num_seqs, 8)
        self.assertEqual(sizing.total_iters, 8)

    def test_actual_num_seqs_clamped_at_one(self):
        # When per_phys_q is large enough to fit only ONE phys req,
        # actual_num_seqs = 1 (not 0).
        sizing = size_logical_workload(
            max_num_seqs=128, max_num_tokens=10275,
            K=256, max_num_subseqs=4224)   # M = 32, per_phys_q = 8192
        self.assertTrue(sizing.valid)
        self.assertEqual(sizing.per_phys_q, 8192)
        # 10275 // 8192 = 1; min(1, 128) = 1.
        self.assertEqual(sizing.actual_num_seqs, 1)
        self.assertEqual(sizing.total_iters, 32)

    def test_total_iters_consistent(self):
        # Across all valid combos for a given (max_num_seqs, MNB, K),
        # total_iters should be relatively stable — the workload
        # spreads the budget across (actual × M) iters either way.
        # This sanity-checks that the scaling produces comparable
        # per-iter measurements regardless of which mns the tuner picks.
        max_num_seqs = 128
        max_num_tokens = 10275
        K = 256
        all_iters = []
        for mns in [256, 384, 640, 1152, 2176, 4224]:
            sizing = size_logical_workload(
                max_num_seqs=max_num_seqs,
                max_num_tokens=max_num_tokens, K=K, max_num_subseqs=mns)
            self.assertTrue(sizing.valid)
            all_iters.append(sizing.total_iters)
        # Ratio between max and min iter count across combos should
        # be bounded — otherwise the tune is comparing apples to
        # oranges. For this recipe: range is [32, 40].
        self.assertLessEqual(max(all_iters) / min(all_iters), 2.0,
                             f"iter count varies too much across "
                             f"combos: {all_iters}")


if __name__ == "__main__":
    unittest.main()
