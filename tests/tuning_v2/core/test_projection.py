# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for tools.tuning.v2.core.projection."""

import unittest

from tools.tuning.v2.core.projection import project_winners


class TestProjectWinners(unittest.TestCase):

    # ----- happy paths -----

    def test_empty_input_returns_empty_list(self):
        self.assertEqual(
            project_winners([], group_fn=lambda r: r["g"],
                            objective_fn=lambda r: r["s"]),
            [],
        )

    def test_single_row_returns_that_row(self):
        rows = [{"g": "x", "s": 10}]
        self.assertEqual(
            project_winners(rows, lambda r: r["g"], lambda r: r["s"]),
            [{"g": "x", "s": 10}],
        )

    def test_ascending_picks_smallest(self):
        rows = [
            {"g": "x", "s": 30},
            {"g": "x", "s": 10},
            {"g": "x", "s": 20},
        ]
        result = project_winners(rows, lambda r: r["g"], lambda r: r["s"])
        self.assertEqual(result, [{"g": "x", "s": 10}])

    def test_descending_picks_largest(self):
        rows = [
            {"g": "x", "s": 10},
            {"g": "x", "s": 30},
            {"g": "x", "s": 20},
        ]
        result = project_winners(
            rows, lambda r: r["g"], lambda r: r["s"], descending=True,
        )
        self.assertEqual(result, [{"g": "x", "s": 30}])

    def test_multiple_groups_each_picks_winner(self):
        rows = [
            {"g": "a", "s": 5},
            {"g": "b", "s": 7},
            {"g": "a", "s": 3},
            {"g": "b", "s": 9},
            {"g": "c", "s": 1},
        ]
        result = project_winners(rows, lambda r: r["g"], lambda r: r["s"])
        # Sorted by canonical_json of group key (= string ordering here).
        self.assertEqual(result, [
            {"g": "a", "s": 3},
            {"g": "b", "s": 7},
            {"g": "c", "s": 1},
        ])

    # ----- determinism -----

    def test_ties_preserve_first_seen(self):
        """When two rows in the same group have equal objective, the
        FIRST one in input order is kept. This makes the projection
        deterministic against repeated calls and easier to reason about."""
        rows = [
            {"g": "x", "s": 5, "tag": "first"},
            {"g": "x", "s": 5, "tag": "second"},
            {"g": "x", "s": 5, "tag": "third"},
        ]
        result = project_winners(rows, lambda r: r["g"], lambda r: r["s"])
        self.assertEqual(result, [{"g": "x", "s": 5, "tag": "first"}])

    def test_idempotent_across_calls(self):
        rows = [
            {"g": "a", "s": 5},
            {"g": "b", "s": 7},
            {"g": "a", "s": 3},
        ]
        first = project_winners(rows, lambda r: r["g"], lambda r: r["s"])
        second = project_winners(rows, lambda r: r["g"], lambda r: r["s"])
        self.assertEqual(first, second)

    # ----- skip_if filter -----

    def test_skip_if_filters_before_grouping(self):
        rows = [
            {"g": "x", "s": 5, "status": "SUCCESS"},
            {"g": "x", "s": 3, "status": "UNKNOWN_ERROR"},
            {"g": "x", "s": 7, "status": "SUCCESS"},
        ]
        result = project_winners(
            rows, lambda r: r["g"], lambda r: r["s"],
            skip_if=lambda r: r["status"] != "SUCCESS",
        )
        # The s=3 UNKNOWN_ERROR row would have won; filter drops it.
        self.assertEqual(result, [
            {"g": "x", "s": 5, "status": "SUCCESS"},
        ])

    def test_skip_if_filters_all_rows_returns_empty(self):
        rows = [
            {"g": "x", "s": 1, "status": "FAILED_OOM"},
            {"g": "x", "s": 2, "status": "FAILED_OOM"},
        ]
        result = project_winners(
            rows, lambda r: r["g"], lambda r: r["s"],
            skip_if=lambda r: r["status"] == "FAILED_OOM",
        )
        self.assertEqual(result, [])

    def test_no_skip_if_keeps_all_rows(self):
        rows = [{"g": "x", "s": 1, "status": "SUCCESS"}]
        self.assertEqual(
            project_winners(rows, lambda r: r["g"], lambda r: r["s"]),
            [{"g": "x", "s": 1, "status": "SUCCESS"}],
        )

    # ----- group key types -----

    def test_group_key_can_be_tuple(self):
        """The kernel-tune case uses a tuple of (canonical_json(tuning_key),
        canonical_json(tunable_params)) as the group key for the skip-set.
        Same shape works for projection."""
        rows = [
            {"tk": "A", "tp": "X", "s": 5},
            {"tk": "A", "tp": "Y", "s": 3},
            {"tk": "B", "tp": "X", "s": 7},
        ]
        # Group only by tuning_key (the canonical projection target for
        # .kernel — one winner per tuning_key, across tunable_params).
        result = project_winners(
            rows, lambda r: r["tk"], lambda r: r["s"],
        )
        self.assertEqual(result, [
            {"tk": "A", "tp": "Y", "s": 3},
            {"tk": "B", "tp": "X", "s": 7},
        ])

    def test_group_key_can_be_dict_via_canonical_json(self):
        """Group keys are stringified for ordering via canonical_json,
        so dict-shaped keys work even though dicts aren't hashable
        directly — the group_fn just has to return SOMETHING hashable.
        Test with a string of a sorted dict to verify."""
        import json
        rows = [
            {"key": {"a": 1, "b": 2}, "s": 5},
            {"key": {"b": 2, "a": 1}, "s": 3},   # same dict, different insertion order
            {"key": {"a": 2, "b": 1}, "s": 7},
        ]
        # Use json.dumps with sort_keys as the group fn for hashability.
        result = project_winners(
            rows,
            group_fn=lambda r: json.dumps(r["key"], sort_keys=True),
            objective_fn=lambda r: r["s"],
        )
        # The two rows with same content (different insertion order) are
        # in the same group; row with s=3 wins.
        self.assertEqual(len(result), 2)
        # Find the winner of group {a:1,b:2}.
        winners_by_s = {r["s"] for r in result}
        self.assertEqual(winners_by_s, {3, 7})

    # ----- error propagation -----

    def test_objective_fn_error_propagates_when_not_filtered(self):
        """A row missing the objective field, NOT filtered by skip_if,
        raises. This is the caller's contract violation; we don't try
        to silently skip."""
        rows = [{"g": "x"}]   # no 's' field
        with self.assertRaises(KeyError):
            project_winners(rows, lambda r: r["g"], lambda r: r["s"])

    def test_objective_fn_error_silenced_by_skip_if(self):
        """The caller's pattern for handling missing fields: filter via
        skip_if BEFORE objective_fn runs."""
        rows = [
            {"g": "x"},   # no 's'
            {"g": "x", "s": 5},
        ]
        result = project_winners(
            rows, lambda r: r["g"], lambda r: r["s"],
            skip_if=lambda r: "s" not in r,
        )
        self.assertEqual(result, [{"g": "x", "s": 5}])

    # ----- realistic shapes -----

    def test_kernel_tune_winner_per_tuning_key(self):
        """Realistic kernel-tune shape: one winner (lowest latency)
        per (page_size, K, code_revision)."""
        rows = [
            {"tk": (128, 256, "abc"), "tp": (256, 2048, 256, 512),
             "latency_us": 2391},
            {"tk": (128, 256, "abc"), "tp": (128, 1024, 128, 512),
             "latency_us": 2150},
            {"tk": (128, 256, "abc"), "tp": (512, 2048, 256, 512),
             "latency_us": 2400},
            {"tk": (128, 512, "abc"), "tp": (256, 2048, 256, 512),
             "latency_us": 3100},
        ]
        winners = project_winners(
            rows,
            group_fn=lambda r: r["tk"],
            objective_fn=lambda r: r["latency_us"],
        )
        self.assertEqual(len(winners), 2)
        # K=256 winner is tp=(128,1024,128,512) @ 2150.
        # K=512 winner is the only one (3100).
        winning_tps = {r["tp"] for r in winners}
        self.assertEqual(
            winning_tps,
            {(128, 1024, 128, 512), (256, 2048, 256, 512)},
        )

    def test_service_sweep_winner_per_objective(self):
        """Realistic service-sweep shape: winner per objective name,
        each across the full combo space (single group)."""
        rows = [
            {"combo": {"MNB": 8192, "MNS": 128},
             "metrics": {"req_per_sec": 4.79, "ttft_ms": 107800}},
            {"combo": {"MNB": 131072, "MNS": 1000},
             "metrics": {"req_per_sec": 4.90, "ttft_ms": 104100}},
            {"combo": {"MNB": 8192, "MNS": 1000},
             "metrics": {"req_per_sec": 1.80, "ttft_ms": 280000}},
        ]
        # All in one workload group (use a constant key).
        throughput_winners = project_winners(
            rows, group_fn=lambda r: "workload",
            objective_fn=lambda r: r["metrics"]["req_per_sec"],
            descending=True,
        )
        ttft_winners = project_winners(
            rows, group_fn=lambda r: "workload",
            objective_fn=lambda r: r["metrics"]["ttft_ms"],
            descending=False,
        )
        self.assertEqual(len(throughput_winners), 1)
        self.assertEqual(throughput_winners[0]["metrics"]["req_per_sec"], 4.90)
        self.assertEqual(len(ttft_winners), 1)
        self.assertEqual(ttft_winners[0]["metrics"]["ttft_ms"], 104100)


if __name__ == "__main__":
    unittest.main()
