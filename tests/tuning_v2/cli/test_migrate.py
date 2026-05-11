# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for tools.tuning.v2.cli.migrate."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.tuning.v2.cli.migrate import (
    MigrationRefusedError,
    _convert_v1_entry,
    _is_v2_format,
    _model_shape_matches,
    main as migrate_main,
    migrate_model_dir,
    split_v1_production_kernel,
)


# Realistic v1 entry shape.
SAMPLE_V1_ENTRY = {
    "tuning_key": {
        "page_size": 128,
        "q_dtype": "bfloat16",
        "kv_dtype": "bfloat16",
        "num_q_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "max_model_len": 8192,
        "sliding_window": None,
        "case": "logical",
        "chunk_prefill_size": 256,
        "code_revision": "abc12345",
    },
    "tunable_params": {
        "bq_sz": 256, "bkv_sz": 2048,
        "bq_csz": 256, "bkv_csz": 512,
        "max_num_subseqs": 4224,
    },
    "Latency": 2203,
    "WarmupTime": 1058372,
    "CaseId": 5,
}

VALID_WORKLOAD_8B = """\
: "${MODEL:=meta-llama/Meta-Llama-3-8B-Instruct}"
: "${TENSOR_PARALLEL_SIZE:=1}"
: "${NUM_Q_HEADS:=32}"
: "${NUM_KV_HEADS:=8}"
: "${HEAD_DIM:=128}"
: "${MAX_MODEL_LEN:=8192}"
: "${MAX_NUM_SEQS:=128}"
"""

# A different-model workload (8 heads vs 32).
DIFFERENT_MODEL_WORKLOAD = """\
: "${MODEL:=some-other-model}"
: "${TENSOR_PARALLEL_SIZE:=1}"
: "${NUM_Q_HEADS:=8}"
: "${NUM_KV_HEADS:=2}"
: "${HEAD_DIM:=64}"
: "${MAX_MODEL_LEN:=4096}"
: "${MAX_NUM_SEQS:=128}"
"""


class TestConvertV1Entry(unittest.TestCase):

    def test_renames_Latency_to_latency_us_as_float(self):
        out = _convert_v1_entry(SAMPLE_V1_ENTRY)
        self.assertEqual(out["latency_us"], 2203.0)
        self.assertIsInstance(out["latency_us"], float)
        self.assertNotIn("Latency", out)

    def test_adds_status_success(self):
        out = _convert_v1_entry(SAMPLE_V1_ENTRY)
        self.assertEqual(out["status"], "SUCCESS")

    def test_drops_warmup_and_case_id(self):
        out = _convert_v1_entry(SAMPLE_V1_ENTRY)
        self.assertNotIn("WarmupTime", out)
        self.assertNotIn("CaseId", out)

    def test_keeps_tuning_key_verbatim(self):
        out = _convert_v1_entry(SAMPLE_V1_ENTRY)
        self.assertEqual(out["tuning_key"], SAMPLE_V1_ENTRY["tuning_key"])

    def test_keeps_tunable_params_verbatim(self):
        out = _convert_v1_entry(SAMPLE_V1_ENTRY)
        self.assertEqual(
            out["tunable_params"], SAMPLE_V1_ENTRY["tunable_params"],
        )

    def test_handles_missing_latency_field(self):
        entry = {
            "tuning_key": {"case": "decode"},
            "tunable_params": {"bq_sz": 1},
        }
        out = _convert_v1_entry(entry)
        self.assertEqual(out["status"], "SUCCESS")
        self.assertNotIn("latency_us", out)


class TestModelShapeMatches(unittest.TestCase):

    WORKLOAD_8B = {
        "NUM_Q_HEADS": "32",
        "NUM_KV_HEADS": "8",
        "HEAD_DIM": "128",
        "MAX_MODEL_LEN": "8192",
        "Q_DTYPE": "bfloat16",
        "KV_DTYPE": "bfloat16",
    }

    def test_full_match(self):
        tk = {
            "num_q_heads": 32, "num_kv_heads": 8,
            "head_dim": 128, "max_model_len": 8192,
            "q_dtype": "bfloat16", "kv_dtype": "bfloat16",
        }
        self.assertTrue(_model_shape_matches(tk, self.WORKLOAD_8B))

    def test_mismatch_on_head_count(self):
        tk = {"num_q_heads": 16, "num_kv_heads": 8}
        self.assertFalse(_model_shape_matches(tk, self.WORKLOAD_8B))

    def test_mismatch_on_max_model_len(self):
        tk = {"num_q_heads": 32, "max_model_len": 16384}
        self.assertFalse(_model_shape_matches(tk, self.WORKLOAD_8B))

    def test_missing_key_in_tuning_key_treated_as_match(self):
        """The tuning_key may not declare every model-shape field.
        Don't filter on absence."""
        tk = {"num_q_heads": 32}   # only one field
        self.assertTrue(_model_shape_matches(tk, self.WORKLOAD_8B))

    def test_missing_key_in_workload_treated_as_match(self):
        """The workload may not declare every model-shape field
        (e.g. q_dtype defaulted). Don't filter on absence."""
        tk = {"num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128}
        # Workload doesn't declare Q_DTYPE / KV_DTYPE.
        wl = {"NUM_Q_HEADS": "32", "NUM_KV_HEADS": "8", "HEAD_DIM": "128"}
        self.assertTrue(_model_shape_matches(tk, wl))

    def test_string_vs_int_comparison(self):
        """Workload env values are strings; tuning_key may have ints.
        Comparison casts to str — match still holds."""
        tk = {"num_q_heads": 32}    # int
        wl = {"NUM_Q_HEADS": "32"}  # str
        self.assertTrue(_model_shape_matches(tk, wl))

    def test_sliding_window_none_in_workload_skipped(self):
        """SLIDING_WINDOW with empty/unset value (None) does not
        constrain the match."""
        tk = {"sliding_window": 4096}
        wl = {}  # no SLIDING_WINDOW
        self.assertTrue(_model_shape_matches(tk, wl))


class TestSplitV1ProductionKernel(unittest.TestCase):

    WORKLOAD_8B = {
        "NUM_Q_HEADS": "32",
        "NUM_KV_HEADS": "8",
        "HEAD_DIM": "128",
        "MAX_MODEL_LEN": "8192",
    }

    def _v1_doc(self, *case_entries: tuple[str, dict]) -> dict:
        results: dict[str, list[dict]] = {}
        for case_name, entry in case_entries:
            results.setdefault(case_name, []).append(entry)
        return {"metadata": {}, "results": results}

    def test_envelope_shape(self):
        v1 = self._v1_doc(("logical", SAMPLE_V1_ENTRY))
        out = split_v1_production_kernel(v1, self.WORKLOAD_8B)
        self.assertEqual(out["schema_version"], 1)
        self.assertEqual(out["code_revision"], "v1_migration")
        self.assertEqual(out["raw_source"], "v1_migration")
        self.assertEqual(out["n_winners"], 1)
        self.assertIn("workload", out)
        self.assertIn("winners", out)

    def test_single_entry_migrates(self):
        v1 = self._v1_doc(("logical", SAMPLE_V1_ENTRY))
        out = split_v1_production_kernel(v1, self.WORKLOAD_8B)
        self.assertEqual(len(out["winners"]), 1)
        w = out["winners"][0]
        self.assertEqual(w["status"], "SUCCESS")
        self.assertEqual(w["latency_us"], 2203.0)
        self.assertEqual(w["tuning_key"]["case"], "logical")

    def test_filters_out_mismatched_model_shape(self):
        """Entry with different num_q_heads is excluded."""
        bigger = dict(SAMPLE_V1_ENTRY)
        bigger["tuning_key"] = dict(SAMPLE_V1_ENTRY["tuning_key"])
        bigger["tuning_key"]["num_q_heads"] = 64    # mismatch
        v1 = self._v1_doc(
            ("logical", SAMPLE_V1_ENTRY),
            ("logical", bigger),
        )
        out = split_v1_production_kernel(v1, self.WORKLOAD_8B)
        # Only the matching entry survives.
        self.assertEqual(out["n_winners"], 1)
        self.assertEqual(out["winners"][0]["tuning_key"]["num_q_heads"], 32)

    def test_includes_all_cases(self):
        """Decode, mixed, prefill, logical entries are all migrated."""
        v1 = self._v1_doc(
            ("decode", SAMPLE_V1_ENTRY),
            ("mixed", SAMPLE_V1_ENTRY),
            ("prefill", SAMPLE_V1_ENTRY),
            ("logical", SAMPLE_V1_ENTRY),
        )
        out = split_v1_production_kernel(v1, self.WORKLOAD_8B)
        self.assertEqual(out["n_winners"], 4)

    def test_derives_case_when_missing_in_tuning_key(self):
        """If a v1 entry omits `case` in tuning_key, the outer
        results key fills it in."""
        no_case = {
            "tuning_key": {
                "num_q_heads": 32, "num_kv_heads": 8,
                "head_dim": 128, "max_model_len": 8192,
            },
            "tunable_params": {"bq_sz": 1},
            "Latency": 1593,
        }
        v1 = {"metadata": {}, "results": {"decode": [no_case]}}
        out = split_v1_production_kernel(v1, self.WORKLOAD_8B)
        self.assertEqual(out["winners"][0]["tuning_key"]["case"], "decode")

    def test_explicit_code_revision_stamp(self):
        v1 = self._v1_doc(("logical", SAMPLE_V1_ENTRY))
        out = split_v1_production_kernel(
            v1, self.WORKLOAD_8B, code_revision="custom",
        )
        self.assertEqual(out["code_revision"], "custom")

    def test_empty_results_yields_empty_winners(self):
        v1 = {"metadata": {}, "results": {}}
        out = split_v1_production_kernel(v1, self.WORKLOAD_8B)
        self.assertEqual(out["n_winners"], 0)
        self.assertEqual(out["winners"], [])

    def test_no_results_key_yields_empty(self):
        v1 = {}
        out = split_v1_production_kernel(v1, self.WORKLOAD_8B)
        self.assertEqual(out["n_winners"], 0)


class TestMigrateModelDir(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cases = Path(self.tmp.name) / "cases" / "v7x" / "llama3_8b"
        self.cases.mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def _write_v1_production_kernel(self) -> None:
        (self.cases / "production.kernel").write_text(json.dumps({
            "metadata": {},
            "results": {
                "logical": [SAMPLE_V1_ENTRY],
                "decode":  [SAMPLE_V1_ENTRY],
            },
        }))

    def test_no_production_kernel_returns_zero(self):
        n, paths = migrate_model_dir(self.cases)
        self.assertEqual(n, 0)
        self.assertEqual(paths, [])

    def test_no_workload_files_returns_zero(self):
        self._write_v1_production_kernel()
        n, paths = migrate_model_dir(self.cases)
        self.assertEqual(n, 0)

    def test_one_workload_one_output(self):
        self._write_v1_production_kernel()
        (self.cases / "prefill_heavy.workload").write_text(
            VALID_WORKLOAD_8B,
        )
        n, paths = migrate_model_dir(self.cases)
        self.assertEqual(n, 1)
        self.assertEqual(paths, [self.cases / "prefill_heavy.kernel"])
        doc = json.loads((self.cases / "prefill_heavy.kernel").read_text())
        self.assertEqual(doc["workload"], "prefill_heavy")
        self.assertEqual(doc["n_winners"], 2)

    def test_two_workloads_same_model_each_gets_copy(self):
        """Both workloads share the 8B model shape; both get all entries."""
        self._write_v1_production_kernel()
        (self.cases / "prefill_heavy.workload").write_text(VALID_WORKLOAD_8B)
        (self.cases / "prefill_heavy_latency.workload").write_text(
            VALID_WORKLOAD_8B,
        )
        n, paths = migrate_model_dir(self.cases)
        self.assertEqual(n, 2)
        names = {p.name for p in paths}
        self.assertEqual(
            names,
            {"prefill_heavy.kernel", "prefill_heavy_latency.kernel"},
        )
        # Both files have all 2 entries (matching model shape).
        for p in paths:
            doc = json.loads(p.read_text())
            self.assertEqual(doc["n_winners"], 2)

    def test_different_model_workload_gets_filtered_entries(self):
        self._write_v1_production_kernel()
        (self.cases / "prefill_heavy.workload").write_text(VALID_WORKLOAD_8B)
        (self.cases / "different.workload").write_text(
            DIFFERENT_MODEL_WORKLOAD,
        )
        n, paths = migrate_model_dir(self.cases)
        # different.workload has NUM_Q_HEADS=8 vs entry's 32 -> 0 winners.
        diff_doc = json.loads(
            (self.cases / "different.kernel").read_text(),
        )
        self.assertEqual(diff_doc["n_winners"], 0)
        heavy_doc = json.loads(
            (self.cases / "prefill_heavy.kernel").read_text(),
        )
        self.assertEqual(heavy_doc["n_winners"], 2)

    def test_does_not_modify_v1_production_kernel(self):
        self._write_v1_production_kernel()
        before = (self.cases / "production.kernel").read_text()
        (self.cases / "x.workload").write_text(VALID_WORKLOAD_8B)
        migrate_model_dir(self.cases)
        after = (self.cases / "production.kernel").read_text()
        self.assertEqual(before, after)


class TestIsV2Format(unittest.TestCase):
    """fix #3: schema check distinguishes v1 (`results`) from v2
    (`by_workload`) format."""

    def test_v1_doc_returns_false(self):
        self.assertFalse(_is_v2_format({"results": {}}))

    def test_v2_doc_returns_true(self):
        self.assertTrue(_is_v2_format({"by_workload": {}}))

    def test_doc_with_both_keys_returns_false(self):
        """Defensive: if for some reason both keys exist, treat as v1
        (don't refuse — let the migration try)."""
        self.assertFalse(
            _is_v2_format({"results": {}, "by_workload": {}}),
        )

    def test_doc_with_neither_returns_false(self):
        """Empty / unknown shape — treat as v1 so migrate doesn't
        block; downstream code handles the empty case."""
        self.assertFalse(_is_v2_format({}))


class TestModelShapeMatchesWidened(unittest.TestCase):
    """fix #8: matching now includes MNS / INPUT / OUTPUT / TP."""

    WORKLOAD_8B_MNS_128 = {
        "NUM_Q_HEADS": "32", "NUM_KV_HEADS": "8",
        "HEAD_DIM": "128", "MAX_MODEL_LEN": "8192",
        "MAX_NUM_SEQS": "128",
        "INPUT_LEN": "8191", "OUTPUT_LEN": "1",
        "TENSOR_PARALLEL_SIZE": "1",
    }

    def test_matching_mns_passes(self):
        tk = {
            "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128,
            "max_model_len": 8192, "max_num_seqs": 128,
        }
        self.assertTrue(_model_shape_matches(tk, self.WORKLOAD_8B_MNS_128))

    def test_mismatched_mns_rejects(self):
        """Two workloads with same model shape but different MNS no
        longer share entries. Previously they would (regression
        captured by fix #8)."""
        tk = {
            "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128,
            "max_model_len": 8192, "max_num_seqs": 1,  # mismatch
        }
        self.assertFalse(_model_shape_matches(tk, self.WORKLOAD_8B_MNS_128))

    def test_mismatched_input_len_rejects(self):
        tk = {
            "num_q_heads": 32, "num_kv_heads": 8,
            "head_dim": 128, "max_model_len": 8192,
            "input_len": 1024,    # mismatch
        }
        self.assertFalse(_model_shape_matches(tk, self.WORKLOAD_8B_MNS_128))

    def test_mismatched_tp_rejects(self):
        tk = {
            "num_q_heads": 32, "tensor_parallel_size": 4,  # mismatch
        }
        self.assertFalse(_model_shape_matches(tk, self.WORKLOAD_8B_MNS_128))

    def test_mns_absent_from_tk_treated_as_match(self):
        """Legacy v1 entries may not carry MAX_NUM_SEQS in tuning_key.
        Don't filter on absence — they're still candidates."""
        tk = {"num_q_heads": 32}    # no max_num_seqs
        self.assertTrue(
            _model_shape_matches(tk, self.WORKLOAD_8B_MNS_128),
        )


class TestMigrateRefusal(unittest.TestCase):
    """fix #3: schema-input check + --force gate."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cases = Path(self.tmp.name) / "cases" / "v7x" / "llama3_8b"
        self.cases.mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def _write_v1(self) -> None:
        (self.cases / "production.kernel").write_text(json.dumps({
            "metadata": {},
            "results": {"logical": [SAMPLE_V1_ENTRY]},
        }))

    def _write_v2(self) -> None:
        (self.cases / "production.kernel").write_text(json.dumps({
            "schema_version": 1,
            "topo": "v7x", "model": "llama3_8b",
            "by_workload": {"alpha": {}},
        }))

    def test_v2_format_refused(self):
        self._write_v2()
        (self.cases / "alpha.workload").write_text(VALID_WORKLOAD_8B)
        with self.assertRaises(MigrationRefusedError):
            migrate_model_dir(self.cases)

    def test_existing_kernel_refused_without_force(self):
        self._write_v1()
        (self.cases / "alpha.workload").write_text(VALID_WORKLOAD_8B)
        (self.cases / "alpha.kernel").write_text('{"existing": true}')
        with self.assertRaises(MigrationRefusedError):
            migrate_model_dir(self.cases)
        # Existing file preserved.
        self.assertEqual(
            json.loads((self.cases / "alpha.kernel").read_text()),
            {"existing": True},
        )

    def test_existing_kernel_overwritten_with_force(self):
        self._write_v1()
        (self.cases / "alpha.workload").write_text(VALID_WORKLOAD_8B)
        (self.cases / "alpha.kernel").write_text('{"existing": true}')
        n, paths = migrate_model_dir(self.cases, force=True)
        self.assertEqual(n, 1)
        doc = json.loads((self.cases / "alpha.kernel").read_text())
        self.assertNotIn("existing", doc)
        self.assertEqual(doc["schema_version"], 1)


class TestCliMain(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_dir_returns_1(self):
        with mock.patch.object(sys, "stderr", new=open(os.devnull, "w")):
            rc = migrate_main([str(self.dir / "no_such")])
        self.assertEqual(rc, 1)

    def test_not_a_directory_returns_1(self):
        f = self.dir / "f.txt"
        f.write_text("hi")
        with mock.patch.object(sys, "stderr", new=open(os.devnull, "w")):
            rc = migrate_main([str(f)])
        self.assertEqual(rc, 1)

    def test_nothing_to_migrate_returns_1(self):
        with mock.patch.object(sys, "stderr", new=open(os.devnull, "w")):
            rc = migrate_main([str(self.dir)])
        self.assertEqual(rc, 1)

    def test_no_workloads_returns_1_with_distinct_message(self):
        """fix #18: v1 file exists but no .workload files. Exit
        message distinguishes this from 'no v1 file at all'."""
        (self.dir / "production.kernel").write_text(json.dumps({
            "metadata": {}, "results": {},
        }))
        import io
        buf = io.StringIO()
        with mock.patch.object(sys, "stderr", new=buf):
            rc = migrate_main([str(self.dir)])
        self.assertEqual(rc, 1)
        self.assertIn("no .workload files", buf.getvalue())

    def test_refusal_returns_1_with_reason(self):
        """v2-format input -> refused -> exit 1 with explanation."""
        (self.dir / "production.kernel").write_text(json.dumps({
            "by_workload": {},
        }))
        (self.dir / "x.workload").write_text(VALID_WORKLOAD_8B)
        import io
        buf = io.StringIO()
        with mock.patch.object(sys, "stderr", new=buf):
            rc = migrate_main([str(self.dir)])
        self.assertEqual(rc, 1)
        self.assertIn("v2 format", buf.getvalue())

    def test_force_flag_forwards_to_migrate_model_dir(self):
        (self.dir / "production.kernel").write_text(json.dumps({
            "metadata": {}, "results": {"logical": [SAMPLE_V1_ENTRY]},
        }))
        (self.dir / "x.workload").write_text(VALID_WORKLOAD_8B)
        (self.dir / "x.kernel").write_text('{"existing": true}')
        with mock.patch.object(sys, "stdout", new=open(os.devnull, "w")):
            with mock.patch.object(sys, "stderr",
                                   new=open(os.devnull, "w")):
                rc = migrate_main([str(self.dir), "--force"])
        self.assertEqual(rc, 0)
        # Overwritten.
        doc = json.loads((self.dir / "x.kernel").read_text())
        self.assertNotIn("existing", doc)

    def test_successful_migration_returns_0(self):
        model_dir = self.dir / "cases" / "v7x" / "llama3_8b"
        model_dir.mkdir(parents=True)
        (model_dir / "production.kernel").write_text(json.dumps({
            "metadata": {},
            "results": {"logical": [SAMPLE_V1_ENTRY]},
        }))
        (model_dir / "x.workload").write_text(VALID_WORKLOAD_8B)
        with mock.patch.object(sys, "stdout", new=open(os.devnull, "w")):
            with mock.patch.object(sys, "stderr", new=open(os.devnull, "w")):
                rc = migrate_main([str(model_dir)])
        self.assertEqual(rc, 0)
        self.assertTrue((model_dir / "x.kernel").exists())


if __name__ == "__main__":
    unittest.main()
