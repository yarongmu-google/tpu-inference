# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for tools.tuning.v2.kernel.search_space."""

import json
import tempfile
import unittest
from pathlib import Path

from tools.tuning.v2.kernel.search_space import (
    DEFAULT_BKV_CSZ,
    DEFAULT_BKV_SZ,
    DEFAULT_BQ_CSZ,
    DEFAULT_BQ_SZ,
    DEFAULT_KERNEL_K,
    DEFAULT_M_VALUES,
    DEFAULT_PAGE_SIZE,
    derive_mnss_candidates,
    kernel_search_space,
)


class TestDeriveMnssCandidates(unittest.TestCase):

    def test_default_m_values_at_mns_128(self):
        """Matches v1's mnss list at MNS=128: M*(M+1) for M ∈ {1,2,4,8,16,32}."""
        result = derive_mnss_candidates(128)
        self.assertEqual(result, [256, 384, 640, 1152, 2176, 4224])

    def test_default_m_values_at_mns_1(self):
        """At MNS=1 the natural mnss candidates are {2, 3, 5, 9, 17, 33}."""
        result = derive_mnss_candidates(1)
        self.assertEqual(result, [2, 3, 5, 9, 17, 33])

    def test_custom_m_values_override(self):
        result = derive_mnss_candidates(128, m_values=[64, 128])
        self.assertEqual(result, [128 * 65, 128 * 129])
        self.assertEqual(result, [8320, 16512])

    def test_duplicates_removed(self):
        """If two M values produce the same mnss (e.g. mns=0 makes all 0),
        the result is deduplicated."""
        result = derive_mnss_candidates(0)
        self.assertEqual(result, [0])

    def test_result_is_sorted(self):
        result = derive_mnss_candidates(10, m_values=[32, 1, 16, 4])
        self.assertEqual(result, sorted(result))

    def test_empty_m_values_returns_empty_list(self):
        self.assertEqual(derive_mnss_candidates(128, m_values=[]), [])


class TestKernelSearchSpace(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_returns_all_axes(self):
        space = kernel_search_space(self.dir, "any", max_num_seqs=128)
        self.assertEqual(
            set(space.keys()),
            {"bq_sz", "bkv_sz", "bq_csz", "bkv_csz",
             "kernel_K", "page_size", "mnss"},
        )

    def test_no_overlay_returns_defaults(self):
        space = kernel_search_space(self.dir, "x", max_num_seqs=128)
        self.assertEqual(space["bq_sz"], DEFAULT_BQ_SZ)
        self.assertEqual(space["bkv_sz"], DEFAULT_BKV_SZ)
        self.assertEqual(space["bq_csz"], DEFAULT_BQ_CSZ)
        self.assertEqual(space["bkv_csz"], DEFAULT_BKV_CSZ)
        self.assertEqual(space["kernel_K"], DEFAULT_KERNEL_K)
        self.assertEqual(space["page_size"], DEFAULT_PAGE_SIZE)
        self.assertEqual(space["mnss"],
                         derive_mnss_candidates(128, DEFAULT_M_VALUES))

    def test_returned_lists_are_copies_not_shared(self):
        """Mutating the returned lists must not affect subsequent calls."""
        s1 = kernel_search_space(self.dir, "x", max_num_seqs=128)
        s1["bq_sz"].append(99999)
        s2 = kernel_search_space(self.dir, "x", max_num_seqs=128)
        self.assertNotIn(99999, s2["bq_sz"])

    def test_overlay_overrides_specific_axis(self):
        overlay = self.dir / "x.kernel_axes.json"
        overlay.write_text(json.dumps({"bq_sz": [256]}))
        space = kernel_search_space(self.dir, "x", max_num_seqs=128)
        self.assertEqual(space["bq_sz"], [256])
        # Other axes still default.
        self.assertEqual(space["bkv_sz"], DEFAULT_BKV_SZ)

    def test_overlay_overrides_multiple_axes(self):
        overlay = self.dir / "x.kernel_axes.json"
        overlay.write_text(json.dumps({
            "bq_sz":    [256],
            "bkv_sz":   [2048],
            "kernel_K": [256],
        }))
        space = kernel_search_space(self.dir, "x", max_num_seqs=128)
        self.assertEqual(space["bq_sz"], [256])
        self.assertEqual(space["bkv_sz"], [2048])
        self.assertEqual(space["kernel_K"], [256])
        # Untouched axes still default.
        self.assertEqual(space["bq_csz"], DEFAULT_BQ_CSZ)
        self.assertEqual(space["page_size"], DEFAULT_PAGE_SIZE)

    def test_overlay_overrides_mnss_bypasses_derivation(self):
        """Explicit mnss in the overlay short-circuits the
        mns × (M+1) formula — useful for narrow-sweep recipes
        (Phase 4 mnss=33, Phase 5 mnss=4224)."""
        overlay = self.dir / "x.kernel_axes.json"
        overlay.write_text(json.dumps({"mnss": [33]}))
        space = kernel_search_space(self.dir, "x", max_num_seqs=1)
        # NOT the formula-derived {2, 3, 5, 9, 17, 33}; just [33].
        self.assertEqual(space["mnss"], [33])

    def test_overlay_with_unknown_axis_added(self):
        """Overlay with an unrecognised axis is allowed (added to dict).
        The tuner's case-generator decides whether to honor unknown
        axes; the search_space loader is permissive."""
        overlay = self.dir / "x.kernel_axes.json"
        overlay.write_text(json.dumps({"future_axis": [1, 2, 3]}))
        space = kernel_search_space(self.dir, "x", max_num_seqs=128)
        self.assertEqual(space["future_axis"], [1, 2, 3])
        # Standard axes still present.
        self.assertEqual(space["bq_sz"], DEFAULT_BQ_SZ)

    def test_invalid_overlay_json_raises(self):
        overlay = self.dir / "x.kernel_axes.json"
        overlay.write_text("{not json")
        with self.assertRaises(json.JSONDecodeError):
            kernel_search_space(self.dir, "x", max_num_seqs=128)

    def test_malformed_overlay_schema_raises(self):
        """fix #7: schema-bad overlay (non-list axis value) is rejected
        loudly via OverlayValidationError, not silently absorbed."""
        from tools.tuning.v2.core.overlay import OverlayValidationError
        overlay = self.dir / "x.kernel_axes.json"
        overlay.write_text(json.dumps({"bq_sz": 256}))   # int, not list
        with self.assertRaises(OverlayValidationError):
            kernel_search_space(self.dir, "x", max_num_seqs=128)

    def test_negative_int_in_overlay_raises(self):
        from tools.tuning.v2.core.overlay import OverlayValidationError
        overlay = self.dir / "x.kernel_axes.json"
        overlay.write_text(json.dumps({"bq_sz": [-256]}))
        with self.assertRaises(OverlayValidationError):
            kernel_search_space(self.dir, "x", max_num_seqs=128)

    def test_mnss_derived_from_max_num_seqs(self):
        space = kernel_search_space(self.dir, "x", max_num_seqs=1)
        self.assertEqual(space["mnss"], [2, 3, 5, 9, 17, 33])

    def test_smoke_test_env_truncates_every_axis_to_one(self):
        """fix #11: SMOKE_TEST=1 -> one value per axis -> one combo
        total. Same env-var contract as v1's rpa_v3_kernel_tuner."""
        import os as _os
        from unittest import mock as _m
        with _m.patch.dict(_os.environ, {"SMOKE_TEST": "1"}):
            space = kernel_search_space(self.dir, "x", max_num_seqs=128)
        # Every axis is length-1.
        for axis, values in space.items():
            with self.subTest(axis=axis):
                self.assertEqual(
                    len(values), 1,
                    f"axis {axis} has {len(values)} values under SMOKE_TEST",
                )
        # Picks each axis's first value (consistent with v1).
        self.assertEqual(space["bq_sz"], [DEFAULT_BQ_SZ[0]])

    def test_smoke_test_env_unset_uses_full_space(self):
        """Defensive: only the literal "1" enables smoke; absent or
        any other value -> full space."""
        import os as _os
        from unittest import mock as _m
        with _m.patch.dict(_os.environ, {}, clear=False):
            _os.environ.pop("SMOKE_TEST", None)
            space = kernel_search_space(self.dir, "x", max_num_seqs=128)
        self.assertEqual(space["bq_sz"], DEFAULT_BQ_SZ)

    def test_smoke_test_env_non_one_value_does_not_truncate(self):
        """SMOKE_TEST=true / =yes / =0 / etc. must NOT enable smoke —
        only the literal "1" matches the v1 contract."""
        import os as _os
        from unittest import mock as _m
        for val in ("0", "true", "yes", "TRUE", ""):
            with self.subTest(val=val):
                with _m.patch.dict(_os.environ, {"SMOKE_TEST": val}):
                    space = kernel_search_space(
                        self.dir, "x", max_num_seqs=128,
                    )
                self.assertEqual(space["bq_sz"], DEFAULT_BQ_SZ)

    def test_smoke_test_env_respects_overlay_choice_of_first_value(self):
        """Overlay applies BEFORE smoke truncation, so an operator can
        pin which single value smoke picks by writing a 1-element
        overlay for that axis."""
        import os as _os
        from unittest import mock as _m
        overlay = self.dir / "x.kernel_axes.json"
        # Pin bq_sz to a non-default first value via overlay.
        overlay.write_text(json.dumps({"bq_sz": [8192, 4096]}))
        with _m.patch.dict(_os.environ, {"SMOKE_TEST": "1"}):
            space = kernel_search_space(self.dir, "x", max_num_seqs=128)
        # Overlay's first value wins (not the global default's first).
        self.assertEqual(space["bq_sz"], [8192])

    def test_workload_name_used_in_overlay_filename(self):
        """Different workload names load different overlay files."""
        (self.dir / "alpha.kernel_axes.json").write_text(
            json.dumps({"bq_sz": [256]}),
        )
        (self.dir / "bravo.kernel_axes.json").write_text(
            json.dumps({"bq_sz": [128]}),
        )
        alpha = kernel_search_space(self.dir, "alpha", max_num_seqs=128)
        bravo = kernel_search_space(self.dir, "bravo", max_num_seqs=128)
        self.assertEqual(alpha["bq_sz"], [256])
        self.assertEqual(bravo["bq_sz"], [128])


if __name__ == "__main__":
    unittest.main()
