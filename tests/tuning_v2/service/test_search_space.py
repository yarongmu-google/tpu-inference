# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for tools.tuning.v2.service.search_space."""

import json
import tempfile
import unittest
from pathlib import Path

from tools.tuning.v2.service.search_space import (
    DEFAULT_MAX_NUM_BATCHED_TOKENS,
    DEFAULT_MAX_NUM_SEQS,
    service_search_space,
)


class TestServiceSearchSpace(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_returns_default_axes_when_no_overlay(self):
        space = service_search_space(self.dir, "any")
        self.assertEqual(space["MAX_NUM_BATCHED_TOKENS"],
                         DEFAULT_MAX_NUM_BATCHED_TOKENS)
        self.assertEqual(space["MAX_NUM_SEQS"], DEFAULT_MAX_NUM_SEQS)

    def test_returned_lists_are_copies(self):
        s1 = service_search_space(self.dir, "x")
        s1["MAX_NUM_SEQS"].append(9999)
        s2 = service_search_space(self.dir, "x")
        self.assertNotIn(9999, s2["MAX_NUM_SEQS"])

    def test_overlay_overrides_specific_axis(self):
        (self.dir / "x.service_axes.json").write_text(
            json.dumps({"MAX_NUM_SEQS": [128, 1000]}),
        )
        space = service_search_space(self.dir, "x")
        self.assertEqual(space["MAX_NUM_SEQS"], [128, 1000])
        # MAX_NUM_BATCHED_TOKENS still default.
        self.assertEqual(space["MAX_NUM_BATCHED_TOKENS"],
                         DEFAULT_MAX_NUM_BATCHED_TOKENS)

    def test_overlay_overrides_multiple_axes(self):
        (self.dir / "x.service_axes.json").write_text(json.dumps({
            "MAX_NUM_BATCHED_TOKENS": [8192, 131072],
            "MAX_NUM_SEQS":           [128, 1000],
        }))
        space = service_search_space(self.dir, "x")
        self.assertEqual(space["MAX_NUM_BATCHED_TOKENS"], [8192, 131072])
        self.assertEqual(space["MAX_NUM_SEQS"], [128, 1000])

    def test_overlay_can_add_new_axis(self):
        """Permissive: unknown axes are accepted (orchestrator decides
        whether to use them)."""
        (self.dir / "x.service_axes.json").write_text(
            json.dumps({"DATASET": ["sonnet", "shakespeare"]}),
        )
        space = service_search_space(self.dir, "x")
        self.assertEqual(space["DATASET"], ["sonnet", "shakespeare"])
        self.assertEqual(space["MAX_NUM_BATCHED_TOKENS"],
                         DEFAULT_MAX_NUM_BATCHED_TOKENS)

    def test_default_max_num_batched_tokens_extends_to_high_end(self):
        """Default MUST include the high-end MNB values that rpa3_2
        Phase 5 surfaced as wins. Otherwise we regress."""
        self.assertIn(131072, DEFAULT_MAX_NUM_BATCHED_TOKENS)
        self.assertIn(1081344, DEFAULT_MAX_NUM_BATCHED_TOKENS)

    def test_default_max_num_seqs_includes_1000(self):
        """MNS=1000 was the rpa3_2 Phase 5 throughput winner."""
        self.assertIn(1000, DEFAULT_MAX_NUM_SEQS)

    def test_invalid_overlay_json_raises(self):
        (self.dir / "x.service_axes.json").write_text("{not json")
        with self.assertRaises(json.JSONDecodeError):
            service_search_space(self.dir, "x")

    def test_malformed_overlay_schema_raises(self):
        """fix #7: schema-bad overlay (non-list axis value) is rejected
        loudly via OverlayValidationError."""
        from tools.tuning.v2.core.overlay import OverlayValidationError
        (self.dir / "x.service_axes.json").write_text(
            json.dumps({"MAX_NUM_SEQS": "128"}),    # str, not list
        )
        with self.assertRaises(OverlayValidationError):
            service_search_space(self.dir, "x")

    def test_mixed_type_list_in_overlay_raises(self):
        from tools.tuning.v2.core.overlay import OverlayValidationError
        (self.dir / "x.service_axes.json").write_text(
            json.dumps({"MAX_NUM_SEQS": [128, "1000"]}),
        )
        with self.assertRaises(OverlayValidationError):
            service_search_space(self.dir, "x")

    def test_smoke_test_env_does_NOT_truncate_search_space(self):
        """SMOKE_TEST behavior moved from the search-space layer to
        the runner (service/sweep.run_service_sweep stops at the
        first SUCCESS row). The search space itself is identical
        with or without SMOKE_TEST=1."""
        import os as _os
        from unittest import mock as _m
        with _m.patch.dict(_os.environ, {"SMOKE_TEST": "1"}):
            space_smoke = service_search_space(self.dir, "x")
        _os.environ.pop("SMOKE_TEST", None)
        space_full = service_search_space(self.dir, "x")
        self.assertEqual(space_smoke, space_full)

    def test_workload_name_used_in_overlay_filename(self):
        (self.dir / "alpha.service_axes.json").write_text(
            json.dumps({"MAX_NUM_SEQS": [128]}),
        )
        (self.dir / "bravo.service_axes.json").write_text(
            json.dumps({"MAX_NUM_SEQS": [1000]}),
        )
        alpha = service_search_space(self.dir, "alpha")
        bravo = service_search_space(self.dir, "bravo")
        self.assertEqual(alpha["MAX_NUM_SEQS"], [128])
        self.assertEqual(bravo["MAX_NUM_SEQS"], [1000])


if __name__ == "__main__":
    unittest.main()
