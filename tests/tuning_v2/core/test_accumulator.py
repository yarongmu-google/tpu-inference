# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for tools.tuning.v2.core.accumulator."""

import json
import tempfile
import unittest
from pathlib import Path

from tools.tuning.v2.core.accumulator import (
    build_production,
    discover_workload_files,
    union_per_workload,
)


class TestDiscoverWorkloadFiles(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_returns_empty_when_dir_missing(self):
        self.assertEqual(
            discover_workload_files(self.dir / "missing", ".kernel"),
            [],
        )

    def test_returns_empty_when_dir_is_file(self):
        f = self.dir / "file.txt"
        f.write_text("hi")
        self.assertEqual(discover_workload_files(f, ".kernel"), [])

    def test_returns_empty_when_dir_empty(self):
        self.assertEqual(discover_workload_files(self.dir, ".kernel"), [])

    def test_finds_kernel_files_sorted(self):
        (self.dir / "zebra.kernel").write_text("{}")
        (self.dir / "alpha.kernel").write_text("{}")
        (self.dir / "mango.kernel").write_text("{}")
        result = discover_workload_files(self.dir, ".kernel")
        self.assertEqual([p.name for p in result],
                         ["alpha.kernel", "mango.kernel", "zebra.kernel"])

    def test_excludes_production_kernel(self):
        (self.dir / "production.kernel").write_text("{}")
        (self.dir / "alpha.kernel").write_text("{}")
        result = discover_workload_files(self.dir, ".kernel")
        self.assertEqual([p.name for p in result], ["alpha.kernel"])

    def test_excludes_other_suffixes(self):
        (self.dir / "alpha.kernel").write_text("{}")
        (self.dir / "alpha.service").write_text("{}")
        (self.dir / "alpha.workload").write_text("MODEL=foo")
        result = discover_workload_files(self.dir, ".kernel")
        self.assertEqual([p.name for p in result], ["alpha.kernel"])

    def test_excludes_kernel_raw_directories(self):
        """`<workload>.kernel.raw/` is a DIR, not a `.kernel` file —
        should not be picked up."""
        (self.dir / "alpha.kernel").write_text("{}")
        raw_dir = self.dir / "alpha.kernel.raw"
        raw_dir.mkdir()
        (raw_dir / "abc.jsonl").write_text('{"row": 1}\n')
        result = discover_workload_files(self.dir, ".kernel")
        self.assertEqual([p.name for p in result], ["alpha.kernel"])

    def test_works_for_service_suffix(self):
        (self.dir / "production.service").write_text("{}")
        (self.dir / "alpha.service").write_text("{}")
        (self.dir / "beta.service").write_text("{}")
        result = discover_workload_files(self.dir, ".service")
        self.assertEqual([p.name for p in result],
                         ["alpha.service", "beta.service"])


class TestUnionPerWorkload(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_empty_list_returns_empty_dict(self):
        self.assertEqual(union_per_workload([]), {})

    def test_single_file_unions_one_entry(self):
        f = self.dir / "alpha.kernel"
        f.write_text('{"winners": [{"k": "v"}]}')
        result = union_per_workload([f], suffix=".kernel")
        self.assertEqual(result, {"alpha": {"winners": [{"k": "v"}]}})

    def test_multi_files_keyed_by_workload_name(self):
        a = self.dir / "alpha.kernel"
        b = self.dir / "bravo.kernel"
        a.write_text('{"x": 1}')
        b.write_text('{"y": 2}')
        result = union_per_workload([a, b], suffix=".kernel")
        self.assertEqual(result, {"alpha": {"x": 1}, "bravo": {"y": 2}})

    def test_empty_suffix_falls_back_to_stem(self):
        f = self.dir / "foo.bar.kernel"
        f.write_text('{"x": 1}')
        result = union_per_workload([f])
        # Stem strips only the LAST extension; default suffix is
        # empty so we fall back to .stem.
        self.assertIn("foo.bar", result)

    def test_explicit_suffix_strips_full_extension(self):
        """When suffix is given, it's used verbatim to strip from name."""
        f = self.dir / "foo.bar.kernel"
        f.write_text('{"x": 1}')
        result = union_per_workload([f], suffix=".kernel")
        self.assertEqual(result, {"foo.bar": {"x": 1}})

    def test_custom_parse_function(self):
        f = self.dir / "raw.txt"
        f.write_text("first\nsecond\nthird")
        result = union_per_workload(
            [f], suffix=".txt",
            parse=lambda s: s.strip().split("\n"),
        )
        self.assertEqual(result, {"raw": ["first", "second", "third"]})

    def test_missing_file_raises(self):
        """File-not-found bubbles up — caller decides how to handle."""
        with self.assertRaises(FileNotFoundError):
            union_per_workload([self.dir / "nope.kernel"])

    def test_invalid_json_raises(self):
        f = self.dir / "bad.kernel"
        f.write_text("{not valid json")
        with self.assertRaises(json.JSONDecodeError):
            union_per_workload([f], suffix=".kernel")


class TestBuildProduction(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        # Build a realistic-looking cases/<topo>/<model>/ path.
        self.cases = Path(self.tmp.name) / "cases"
        self.model_dir = self.cases / "v7x" / "llama3_8b"
        self.model_dir.mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_envelope_has_expected_keys(self):
        (self.model_dir / "alpha.kernel").write_text('{"w": []}')
        result = build_production(self.model_dir, ".kernel")
        self.assertEqual(set(result.keys()),
                         {"schema_version", "topo", "model", "by_workload"})
        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["topo"], "v7x")
        self.assertEqual(result["model"], "llama3_8b")
        self.assertEqual(result["by_workload"], {"alpha": {"w": []}})

    def test_empty_dir_produces_empty_by_workload(self):
        result = build_production(self.model_dir, ".kernel")
        self.assertEqual(result["by_workload"], {})

    def test_excludes_production_kernel_from_input(self):
        (self.model_dir / "production.kernel").write_text('{"old": true}')
        (self.model_dir / "alpha.kernel").write_text('{"w": [1]}')
        result = build_production(self.model_dir, ".kernel")
        self.assertEqual(result["by_workload"], {"alpha": {"w": [1]}})

    def test_explicit_topo_and_model_override_inference(self):
        (self.model_dir / "alpha.kernel").write_text('{"w": []}')
        result = build_production(
            self.model_dir, ".kernel",
            topo="custom_topo", model="custom_model",
        )
        self.assertEqual(result["topo"], "custom_topo")
        self.assertEqual(result["model"], "custom_model")

    def test_explicit_topo_only_model_inferred(self):
        """Partial override: topo given, model inferred."""
        (self.model_dir / "alpha.kernel").write_text('{"w": []}')
        result = build_production(
            self.model_dir, ".kernel", topo="custom_topo",
        )
        self.assertEqual(result["topo"], "custom_topo")
        self.assertEqual(result["model"], "llama3_8b")    # inferred

    def test_explicit_model_only_topo_inferred(self):
        """Partial override: model given, topo inferred."""
        (self.model_dir / "alpha.kernel").write_text('{"w": []}')
        result = build_production(
            self.model_dir, ".kernel", model="custom_model",
        )
        self.assertEqual(result["topo"], "v7x")    # inferred
        self.assertEqual(result["model"], "custom_model")

    def test_infers_unknown_when_path_has_no_cases_segment(self):
        plain = Path(self.tmp.name) / "plain_dir"
        plain.mkdir()
        (plain / "alpha.kernel").write_text('{"w": []}')
        result = build_production(plain, ".kernel")
        self.assertEqual(result["topo"], "unknown")
        self.assertEqual(result["model"], "unknown")

    def test_infers_unknown_when_cases_segment_has_no_topo_model(self):
        """Path like `.../cases/` with no further segments."""
        bare_cases = Path(self.tmp.name) / "cases"
        # cases already exists from setUp; just verify behavior.
        result = build_production(bare_cases, ".kernel")
        # At least one of (topo, model) should be unknown since the
        # path doesn't have <topo>/<model> after "cases".
        # cases/<v7x> exists from setUp, so partial inference: topo=v7x,
        # but model can't be inferred from a bare "cases" path.
        # Actually `cases/` parts = [..., "cases"]; idx + 2 >= len(parts).
        self.assertEqual(result["topo"], "unknown")
        self.assertEqual(result["model"], "unknown")

    def test_works_for_service_suffix(self):
        (self.model_dir / "alpha.service").write_text(
            '{"winners": {"throughput_max": {"req_per_sec": 4.9}}}',
        )
        result = build_production(self.model_dir, ".service")
        self.assertEqual(
            result["by_workload"]["alpha"],
            {"winners": {"throughput_max": {"req_per_sec": 4.9}}},
        )

    def test_multiple_workloads_keyed_by_name(self):
        (self.model_dir / "throughput.kernel").write_text(
            '{"winners": [{"K": 256, "lat": 2391}]}',
        )
        (self.model_dir / "latency.kernel").write_text(
            '{"winners": [{"K": 256, "lat": 388}]}',
        )
        result = build_production(self.model_dir, ".kernel")
        self.assertEqual(
            set(result["by_workload"].keys()),
            {"throughput", "latency"},
        )


if __name__ == "__main__":
    unittest.main()
