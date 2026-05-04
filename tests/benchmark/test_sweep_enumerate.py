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
"""Unit tests for tools.benchmark.sweep — load + enumerate + naming.

Coverage target: 100% lines + branches of the functions defined in
tools/benchmark/sweep.py at this stage (the subprocess driver is added
in a later commit and gets its own test file).
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

from tools.benchmark import sweep


def _write_json(d) -> str:
    f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    if isinstance(d, str):
        f.write(d)
    else:
        json.dump(d, f)
    f.close()
    return f.name


class TestLoadSpec(unittest.TestCase):

    def test_minimal_valid(self):
        path = _write_json({"case_file": "x", "sweep_name": "y"})
        try:
            spec = sweep.load_spec(path)
            self.assertEqual(spec["case_file"], "x")
            self.assertEqual(spec["sweep_name"], "y")
        finally:
            os.unlink(path)

    def test_full_valid(self):
        path = _write_json({
            "case_file": "case.env",
            "sweep_name": "s",
            "sweep_axes": {"A": [1, 2]},
            "coupled_axes": [{"X": 1}, {"X": 2}],
            "fixed": {"F": "v"},
        })
        try:
            spec = sweep.load_spec(path)
            self.assertEqual(spec["sweep_axes"], {"A": [1, 2]})
        finally:
            os.unlink(path)

    def test_top_level_not_object(self):
        path = _write_json("[1,2,3]")
        try:
            with self.assertRaises(sweep.SpecError):
                sweep.load_spec(path)
        finally:
            os.unlink(path)

    def test_missing_required(self):
        path = _write_json({"case_file": "x"})
        try:
            with self.assertRaisesRegex(sweep.SpecError,
                                        "missing required keys"):
                sweep.load_spec(path)
        finally:
            os.unlink(path)

    def test_bad_json(self):
        path = _write_json("not json at all")
        try:
            with self.assertRaises(json.JSONDecodeError):
                sweep.load_spec(path)
        finally:
            os.unlink(path)


class TestValidateSpec(unittest.TestCase):

    def _v(self, **overrides):
        spec = {"case_file": "c", "sweep_name": "s"}
        spec.update(overrides)
        sweep._validate_spec(spec)

    def test_sweep_axes_must_be_dict(self):
        with self.assertRaisesRegex(sweep.SpecError, "sweep_axes must"):
            self._v(sweep_axes=[1, 2])

    def test_coupled_axes_must_be_list(self):
        with self.assertRaisesRegex(sweep.SpecError, "coupled_axes must"):
            self._v(coupled_axes={"a": 1})

    def test_fixed_must_be_dict(self):
        with self.assertRaisesRegex(sweep.SpecError, "fixed must"):
            self._v(fixed=[1])

    def test_sweep_axis_value_must_be_nonempty_list(self):
        with self.assertRaisesRegex(sweep.SpecError, "non-empty list"):
            self._v(sweep_axes={"A": []})
        with self.assertRaisesRegex(sweep.SpecError, "non-empty list"):
            self._v(sweep_axes={"A": "abc"})

    def test_coupled_entry_must_be_dict(self):
        with self.assertRaisesRegex(sweep.SpecError,
                                    r"coupled_axes\[1\] must be"):
            self._v(coupled_axes=[{"a": 1}, [1, 2]])

    def test_coupled_entries_must_share_keyset(self):
        with self.assertRaisesRegex(sweep.SpecError,
                                    r"coupled_axes\[1\] keys"):
            self._v(coupled_axes=[{"X": 1}, {"X": 2, "Y": 3}])

    def test_overlap_sweep_coupled(self):
        with self.assertRaisesRegex(sweep.SpecError,
                                    "sweep_axes/coupled_axes share"):
            self._v(sweep_axes={"A": [1]}, coupled_axes=[{"A": 1}])

    def test_overlap_sweep_fixed(self):
        with self.assertRaisesRegex(sweep.SpecError, "sweep_axes/fixed"):
            self._v(sweep_axes={"A": [1]}, fixed={"A": "v"})

    def test_overlap_coupled_fixed(self):
        with self.assertRaisesRegex(sweep.SpecError, "coupled_axes/fixed"):
            self._v(coupled_axes=[{"X": 1}], fixed={"X": "v"})

    def test_empty_optional_sections_ok(self):
        # All three optional sections empty -> still valid.
        self._v(sweep_axes={}, coupled_axes=[], fixed={})
        # None for optionals -> get coerced to {} / [] via `or`.
        self._v(sweep_axes=None, coupled_axes=None, fixed=None)


class TestEnumerateCombos(unittest.TestCase):

    def test_only_fixed(self):
        spec = {"case_file": "c", "sweep_name": "s", "fixed": {"X": "v"}}
        self.assertEqual(sweep.enumerate_combos(spec), [{"X": "v"}])

    def test_no_fixed_no_coupled_no_sweep(self):
        spec = {"case_file": "c", "sweep_name": "s"}
        self.assertEqual(sweep.enumerate_combos(spec), [{}])

    def test_only_sweep_axes_cartesian(self):
        spec = {
            "case_file": "c", "sweep_name": "s",
            "sweep_axes": {"A": [1, 2], "B": [10, 20]},
        }
        combos = sweep.enumerate_combos(spec)
        self.assertEqual(len(combos), 4)
        # Check both deterministic and value-correct
        self.assertEqual(combos[0], {"A": "1", "B": "10"})
        self.assertEqual(combos[1], {"A": "1", "B": "20"})
        self.assertEqual(combos[2], {"A": "2", "B": "10"})
        self.assertEqual(combos[3], {"A": "2", "B": "20"})

    def test_only_coupled(self):
        spec = {
            "case_file": "c", "sweep_name": "s",
            "coupled_axes": [{"X": 1, "Y": 10}, {"X": 2, "Y": 20}],
        }
        self.assertEqual(sweep.enumerate_combos(spec), [
            {"X": "1", "Y": "10"},
            {"X": "2", "Y": "20"},
        ])

    def test_cartesian_times_coupled(self):
        spec = {
            "case_file": "c", "sweep_name": "s",
            "sweep_axes":   {"A": [1, 2]},
            "coupled_axes": [{"X": "a"}, {"X": "b"}],
        }
        combos = sweep.enumerate_combos(spec)
        # 2 cartesian × 2 coupled = 4
        self.assertEqual(len(combos), 4)
        self.assertEqual(combos, [
            {"A": "1", "X": "a"},
            {"A": "1", "X": "b"},
            {"A": "2", "X": "a"},
            {"A": "2", "X": "b"},
        ])

    def test_fixed_merged_into_every_combo(self):
        spec = {
            "case_file": "c", "sweep_name": "s",
            "sweep_axes": {"A": [1, 2]},
            "fixed":      {"F": "fv"},
        }
        for combo in sweep.enumerate_combos(spec):
            self.assertEqual(combo["F"], "fv")

    def test_full_spec_combo_count(self):
        spec = {
            "case_file": "c", "sweep_name": "s",
            "sweep_axes":   {"A": [1, 2, 3], "B": [10, 20]},
            "coupled_axes": [{"X": 1}, {"X": 2}, {"X": 3}],
            "fixed":        {"F": "fv"},
        }
        combos = sweep.enumerate_combos(spec)
        self.assertEqual(len(combos), 3 * 2 * 3)
        # Spot-check fixed always present, A/B/X all present.
        for combo in combos:
            self.assertEqual(set(combo.keys()), {"A", "B", "X", "F"})
            self.assertEqual(combo["F"], "fv")

    def test_values_stringified(self):
        spec = {
            "case_file": "c", "sweep_name": "s",
            "sweep_axes":   {"A": [1, 2]},
            "coupled_axes": [{"X": 3.14, "Y": "literal"}],
            "fixed":        {"B": True},
        }
        combos = sweep.enumerate_combos(spec)
        self.assertTrue(all(isinstance(v, str) for c in combos for v in c.values()))
        self.assertEqual(combos[0]["A"], "1")
        self.assertEqual(combos[0]["X"], "3.14")
        self.assertEqual(combos[0]["B"], "True")


class TestComboId(unittest.TestCase):

    def test_deterministic(self):
        env = {"A": "1", "B": "2"}
        self.assertEqual(sweep.combo_id(env), sweep.combo_id(env))

    def test_key_order_irrelevant(self):
        # Same content, different insertion order -> same ID.
        a = sweep.combo_id({"A": "1", "B": "2"})
        b = sweep.combo_id({"B": "2", "A": "1"})
        self.assertEqual(a, b)

    def test_different_contents_different_ids(self):
        self.assertNotEqual(
            sweep.combo_id({"A": "1"}),
            sweep.combo_id({"A": "2"}),
        )

    def test_id_shape(self):
        cid = sweep.combo_id({"A": "1"})
        self.assertEqual(len(cid), 12)
        self.assertTrue(all(c in "0123456789abcdef" for c in cid))


class TestPaths(unittest.TestCase):

    def test_case_name_strips_dir_and_ext(self):
        self.assertEqual(
            sweep.case_name_from_path("tools/benchmark/cases/foo.env"),
            "foo")

    def test_case_name_no_ext(self):
        self.assertEqual(sweep.case_name_from_path("foo"), "foo")

    def test_case_name_other_ext(self):
        self.assertEqual(sweep.case_name_from_path("foo.json"), "foo.json")

    def test_result_dir_shape(self):
        p = sweep.result_dir("/tmp/x", "casename", "sweepname", "abc123")
        self.assertEqual(p, Path("/tmp/x/bench_casename_sweepname/abc123"))

    def test_sweep_dir_shape(self):
        p = sweep.sweep_dir("tmp", "casename", "sweepname")
        self.assertEqual(p, Path("tmp/bench_casename_sweepname"))

    def test_is_completed_false_for_missing(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(sweep.is_completed(d))

    def test_is_completed_true_when_metrics_present(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "metrics.txt").write_text("foo=bar\n")
            self.assertTrue(sweep.is_completed(d))

    def test_is_completed_false_for_dir_named_metrics(self):
        # If 'metrics.txt' exists as a directory, not a file, it's NOT done.
        with tempfile.TemporaryDirectory() as d:
            os.mkdir(Path(d) / "metrics.txt")
            self.assertFalse(sweep.is_completed(d))


if __name__ == "__main__":
    unittest.main()
