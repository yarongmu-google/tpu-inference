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
"""Unit tests for compute_per_rank_distribution() in tpu_runner.

Mirrors the bucket-population coverage of
test_persistent_batch_manager.TestReorderBatch (no reqs, decode only,
prefill only, mix only, decode+prefill, decode+mix, prefill+mix, all
three) but operates on per-rank slices and verifies the multi-rank
flattening shape used by AttentionMetadata.
"""

import unittest

from tpu_inference.runner.tpu_runner import compute_per_rank_distribution


K = 128


def _call(req_ids_per_rank, num_scheduled_tokens, chunk_prefill_size=K):
    """Helper: convenient call with num_req_per_rank derived from
    req_ids_per_rank lengths."""
    num_req_per_rank = [len(r) for r in req_ids_per_rank]
    return compute_per_rank_distribution(req_ids_per_rank, num_req_per_rank,
                                         num_scheduled_tokens,
                                         chunk_prefill_size)


class TestComputePerRankDistribution(unittest.TestCase):

    # ---- Single-rank cases (covers the same 8 bucket combos as the
    # _reorder_batch tests, since the per-rank slice is the same shape).

    def test_no_requests(self):
        self.assertEqual(_call([[]], {}), [[0, 0, 0]])

    def test_decode_only(self):
        nst = {"r0": 1, "r1": 1, "r2": 1}
        self.assertEqual(_call([["r0", "r1", "r2"]], nst), [[3, 3, 3]])

    def test_prefill_only(self):
        nst = {"r0": K, "r1": K, "r2": K}
        self.assertEqual(_call([["r0", "r1", "r2"]], nst), [[0, 3, 3]])

    def test_mix_only(self):
        nst = {"r0": 50, "r1": 200, "r2": 75}
        self.assertEqual(_call([["r0", "r1", "r2"]], nst), [[0, 0, 3]])

    def test_decode_plus_prefill(self):
        nst = {"r0": 1, "r1": K, "r2": 1, "r3": K}
        self.assertEqual(_call([["r0", "r1", "r2", "r3"]], nst), [[2, 4, 4]])

    def test_decode_plus_mix(self):
        nst = {"r0": 1, "r1": 200, "r2": 1, "r3": 50}
        self.assertEqual(_call([["r0", "r1", "r2", "r3"]], nst), [[2, 2, 4]])

    def test_prefill_plus_mix(self):
        nst = {"r0": K, "r1": 50, "r2": K, "r3": 200}
        self.assertEqual(_call([["r0", "r1", "r2", "r3"]], nst), [[0, 2, 4]])

    def test_all_three_buckets(self):
        nst = {
            "r0": 1, "r1": 1,        # decode
            "r2": K, "r3": K,        # uniform-K
            "r4": 50, "r5": 200,     # mixed
        }
        self.assertEqual(
            _call([["r0", "r1", "r2", "r3", "r4", "r5"]], nst),
            [[2, 4, 6]])

    # ---- Multi-rank cases.

    def test_two_ranks_independent_distributions(self):
        """Each rank has its own distribution; confirm cross-rank
        independence."""
        nst = {
            "r0": 1, "r1": K,      # rank 0: 1 decode, 1 uniform-K
            "r2": 50, "r3": 200,   # rank 1: 2 mixed
        }
        result = _call([["r0", "r1"], ["r2", "r3"]], nst)
        self.assertEqual(result, [[1, 2, 2], [0, 0, 2]])

    def test_two_ranks_empty_rank(self):
        """Empty rank produces [0, 0, 0]."""
        nst = {"r0": 1, "r1": K, "r2": 50}
        result = _call([["r0", "r1", "r2"], []], nst)
        self.assertEqual(result, [[1, 2, 3], [0, 0, 0]])

    # ---- chunk_prefill_size=None (regression to today's [D, D, T]).

    def test_disabled_falls_back_to_old_behavior(self):
        """K=None: P bucket stays empty even when reqs match what would be K."""
        nst = {"r0": 1, "r1": K, "r2": 50}
        result = _call([["r0", "r1", "r2"]], nst, chunk_prefill_size=None)
        self.assertEqual(result, [[1, 1, 3]])

    def test_disabled_multi_rank(self):
        nst = {
            "r0": 1, "r1": K,
            "r2": 50, "r3": 200,
        }
        result = _call([["r0", "r1"], ["r2", "r3"]], nst,
                       chunk_prefill_size=None)
        self.assertEqual(result, [[1, 1, 2], [0, 0, 2]])


if __name__ == "__main__":
    unittest.main()
