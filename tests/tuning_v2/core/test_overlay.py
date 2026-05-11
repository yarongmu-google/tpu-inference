# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for tools.tuning.v2.core.overlay schema validation (fix #7)."""

import unittest
from pathlib import Path

from tools.tuning.v2.core.overlay import (
    OverlayValidationError,
    validate_overlay_schema,
)


# Fixed sentinel path — only used in error messages, doesn't need to
# exist on disk.
PATH = Path("/tmp/x.kernel_axes.json")


class TestValidateOverlaySchema(unittest.TestCase):

    def test_empty_dict_is_valid(self):
        validate_overlay_schema({}, PATH)

    def test_valid_int_axis(self):
        validate_overlay_schema({"bq_sz": [256, 512]}, PATH)

    def test_valid_str_axis(self):
        """fix #7: str axes (e.g. DATASET) are accepted as long as the
        list is homogeneous strings."""
        validate_overlay_schema(
            {"DATASET": ["sonnet", "random"]}, PATH,
        )

    def test_single_element_int_list(self):
        validate_overlay_schema({"page_size": [128]}, PATH)

    def test_top_level_must_be_dict(self):
        for bad in ([1, 2, 3], "not a dict", 42, None):
            with self.subTest(bad=bad):
                with self.assertRaises(OverlayValidationError):
                    validate_overlay_schema(bad, PATH)

    def test_axis_value_must_be_list(self):
        with self.assertRaises(OverlayValidationError) as cm:
            validate_overlay_schema({"bq_sz": 256}, PATH)
        self.assertIn("bq_sz", str(cm.exception))
        self.assertIn("must map to a list", str(cm.exception))

    def test_axis_value_dict_rejected(self):
        with self.assertRaises(OverlayValidationError):
            validate_overlay_schema({"bq_sz": {"v": 256}}, PATH)

    def test_empty_list_rejected(self):
        """Empty list is ambiguous — does the operator want "no
        candidates" (so the axis is unsatisfiable) or "inherit
        default"? Reject; force them to omit the key."""
        with self.assertRaises(OverlayValidationError) as cm:
            validate_overlay_schema({"bq_sz": []}, PATH)
        self.assertIn("omit the key", str(cm.exception))

    def test_non_int_non_str_element_rejected(self):
        with self.assertRaises(OverlayValidationError) as cm:
            validate_overlay_schema({"bq_sz": [1.5]}, PATH)
        self.assertIn("bq_sz", str(cm.exception))

    def test_bool_rejected_even_though_subclass_of_int(self):
        """Python footgun: `True is 1` is False but `isinstance(True, int)`
        is True. Overlay schema must reject bools explicitly."""
        with self.assertRaises(OverlayValidationError):
            validate_overlay_schema({"bq_sz": [True, False]}, PATH)

    def test_zero_rejected(self):
        with self.assertRaises(OverlayValidationError) as cm:
            validate_overlay_schema({"bq_sz": [0]}, PATH)
        self.assertIn("non-positive", str(cm.exception))

    def test_negative_rejected(self):
        with self.assertRaises(OverlayValidationError):
            validate_overlay_schema({"bq_sz": [-1]}, PATH)

    def test_mixed_types_rejected(self):
        """A list of [128, "256"] mixes int + str. Reject — the
        downstream enumerator can't safely sweep heterogeneous
        types."""
        with self.assertRaises(OverlayValidationError) as cm:
            validate_overlay_schema({"bq_sz": [128, "256"]}, PATH)
        self.assertIn("homogeneous", str(cm.exception))

    def test_empty_string_rejected(self):
        with self.assertRaises(OverlayValidationError) as cm:
            validate_overlay_schema({"DATASET": [""]}, PATH)
        self.assertIn("non-empty", str(cm.exception))

    def test_error_message_includes_source_path(self):
        path = Path("/tmp/specific.json")
        with self.assertRaises(OverlayValidationError) as cm:
            validate_overlay_schema({"bq_sz": [-1]}, path)
        self.assertIn(str(path), str(cm.exception))


if __name__ == "__main__":
    unittest.main()
