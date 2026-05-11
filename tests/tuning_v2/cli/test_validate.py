# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for tools.tuning.v2.cli.validate."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.tuning.v2.cli.validate import (
    main as validate_main,
    parse_workload_env,
    validate,
)


VALID_WORKLOAD = """\
: "${MODEL:=meta-llama/Meta-Llama-3-8B-Instruct}"
: "${TENSOR_PARALLEL_SIZE:=1}"
: "${NUM_Q_HEADS:=32}"
: "${NUM_KV_HEADS:=8}"
: "${HEAD_DIM:=128}"
: "${MAX_MODEL_LEN:=8192}"
: "${MAX_NUM_SEQS:=128}"
: "${INPUT_LEN:=8191}"
: "${OUTPUT_LEN:=1}"
"""


class TestParseWorkloadEnv(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_picks_up_set_vars(self):
        w = self.dir / "x.workload"
        w.write_text(VALID_WORKLOAD)
        env = parse_workload_env(w)
        self.assertEqual(env["MODEL"],
                         "meta-llama/Meta-Llama-3-8B-Instruct")
        self.assertEqual(env["MAX_NUM_SEQS"], "128")

    def test_handles_lines_without_equals(self):
        """If `env` ever output a non-K=V line (banner / warning), we
        should skip it gracefully in both halves of the sentinel-split
        output."""
        with mock.patch("subprocess.run") as run:
            run.return_value = mock.Mock(
                stdout=(
                    "A=1\nbanner without equals\nB=2\n"
                    "@@@WORKLOAD_DIFF_SENTINEL@@@\n"
                    "A=1\nB=2\nC=3\nanother banner\n"
                ),
                returncode=0,
            )
            w = self.dir / "x.workload"
            w.write_text("placeholder")
            env = parse_workload_env(w)
        # Only newly-set vars surface; banner lines are dropped.
        self.assertEqual(env, {"C": "3"})

    def test_filters_out_inherited_shell_env(self):
        """Env vars from the calling shell (e.g. PATH, HOME) must NOT
        appear in the result; only vars actually set by the file. The
        clean `env -i` subshell has no PATH/HOME to begin with, but
        this guards against regression if anyone removes `env -i`."""
        w = self.dir / "x.workload"
        w.write_text(VALID_WORKLOAD)
        env = parse_workload_env(w)
        self.assertNotIn("PATH", env)
        self.assertNotIn("HOME", env)

    def test_workload_var_surfaces_even_when_parent_shell_has_same_value(
        self,
    ):
        """fix #4: pre-set parent-shell vars must NOT mask workload
        vars that happen to carry the same value. The old
        snapshot-against-parent diff dropped them silently; the
        clean-subshell same-shell diff fixes this."""
        w = self.dir / "x.workload"
        w.write_text('MAX_MODEL_LEN="8192"\nNUM_Q_HEADS="32"\n')
        with mock.patch.dict(
            os.environ,
            {"MAX_MODEL_LEN": "8192", "NUM_Q_HEADS": "32"},
        ):
            env = parse_workload_env(w)
        self.assertEqual(env.get("MAX_MODEL_LEN"), "8192")
        self.assertEqual(env.get("NUM_Q_HEADS"), "32")

    def test_no_sentinel_in_output_treats_all_as_post_source(self):
        """Defensive: if the sentinel is somehow swallowed (unlikely),
        fall back to treating all output as workload-set rather than
        crashing or returning empty."""
        with mock.patch("subprocess.run") as run:
            run.return_value = mock.Mock(
                stdout="X=1\nY=2\n",   # no sentinel
                returncode=0,
            )
            w = self.dir / "x.workload"
            w.write_text("placeholder")
            env = parse_workload_env(w)
        self.assertEqual(env, {"X": "1", "Y": "2"})


class TestValidate(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_valid_workload_returns_empty(self):
        w = self.dir / "x.workload"
        w.write_text(VALID_WORKLOAD)
        self.assertEqual(validate(w), [])

    def test_missing_file_returns_error(self):
        issues = validate(self.dir / "absent.workload")
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0][0], "error")
        self.assertIn("not found", issues[0][1])

    def test_missing_required_var_reports_error(self):
        w = self.dir / "x.workload"
        # Omit NUM_KV_HEADS.
        w.write_text("""\
: "${MODEL:=foo}"
: "${TENSOR_PARALLEL_SIZE:=1}"
: "${NUM_Q_HEADS:=32}"
: "${HEAD_DIM:=128}"
: "${MAX_MODEL_LEN:=8192}"
: "${MAX_NUM_SEQS:=128}"
: "${INPUT_LEN:=8191}"
: "${OUTPUT_LEN:=1}"
""")
        issues = validate(w)
        errs = [m for sev, m in issues if sev == "error"]
        self.assertTrue(any("NUM_KV_HEADS" in m for m in errs))

    def test_missing_input_len_reports_error(self):
        """fix #4: INPUT_LEN is now required."""
        w = self.dir / "x.workload"
        w.write_text("""\
: "${MODEL:=foo}"
: "${TENSOR_PARALLEL_SIZE:=1}"
: "${NUM_Q_HEADS:=32}"
: "${NUM_KV_HEADS:=8}"
: "${HEAD_DIM:=128}"
: "${MAX_MODEL_LEN:=8192}"
: "${MAX_NUM_SEQS:=128}"
: "${OUTPUT_LEN:=1}"
""")
        issues = validate(w)
        errs = [m for sev, m in issues if sev == "error"]
        self.assertTrue(any("INPUT_LEN" in m for m in errs))

    def test_missing_output_len_reports_error(self):
        """fix #4: OUTPUT_LEN is now required."""
        w = self.dir / "x.workload"
        w.write_text("""\
: "${MODEL:=foo}"
: "${TENSOR_PARALLEL_SIZE:=1}"
: "${NUM_Q_HEADS:=32}"
: "${NUM_KV_HEADS:=8}"
: "${HEAD_DIM:=128}"
: "${MAX_MODEL_LEN:=8192}"
: "${MAX_NUM_SEQS:=128}"
: "${INPUT_LEN:=8191}"
""")
        issues = validate(w)
        errs = [m for sev, m in issues if sev == "error"]
        self.assertTrue(any("OUTPUT_LEN" in m for m in errs))

    def test_max_model_len_below_input_plus_output_reports_error(self):
        """fix #4 invariant: MAX_MODEL_LEN >= INPUT_LEN + OUTPUT_LEN."""
        w = self.dir / "x.workload"
        w.write_text("""\
: "${MODEL:=foo}"
: "${TENSOR_PARALLEL_SIZE:=1}"
: "${NUM_Q_HEADS:=32}"
: "${NUM_KV_HEADS:=8}"
: "${HEAD_DIM:=128}"
: "${MAX_MODEL_LEN:=4096}"
: "${MAX_NUM_SEQS:=128}"
: "${INPUT_LEN:=4000}"
: "${OUTPUT_LEN:=100}"
""")
        issues = validate(w)
        errs = [m for sev, m in issues if sev == "error"]
        self.assertTrue(any("MAX_MODEL_LEN" in m and "INPUT_LEN+OUTPUT_LEN"
                            in m for m in errs))

    def test_invariant_not_checked_when_inputs_missing(self):
        """If INPUT_LEN is missing, the invariant check must NOT run
        (it would crash on the missing key, or worse, report a
        spurious second error)."""
        w = self.dir / "x.workload"
        w.write_text("""\
: "${MODEL:=foo}"
: "${TENSOR_PARALLEL_SIZE:=1}"
: "${NUM_Q_HEADS:=32}"
: "${NUM_KV_HEADS:=8}"
: "${HEAD_DIM:=128}"
: "${MAX_MODEL_LEN:=8192}"
: "${MAX_NUM_SEQS:=128}"
: "${OUTPUT_LEN:=1}"
""")   # INPUT_LEN missing
        issues = validate(w)
        errs = [m for sev, m in issues if sev == "error"]
        # Only the missing-INPUT_LEN error; no invariant cascade.
        self.assertEqual(
            sum(1 for m in errs if "INPUT_LEN+OUTPUT_LEN" in m),
            0,
        )

    def test_invariant_not_checked_when_non_integer(self):
        """If MAX_MODEL_LEN isn't an integer, the int-check error fires
        but the invariant must NOT run (and must NOT crash)."""
        w = self.dir / "x.workload"
        w.write_text(VALID_WORKLOAD + 'MAX_MODEL_LEN="abc"\n')
        issues = validate(w)
        errs = [m for sev, m in issues if sev == "error"]
        # No invariant error.
        self.assertEqual(
            sum(1 for m in errs if "INPUT_LEN+OUTPUT_LEN" in m),
            0,
        )

    def test_empty_value_reports_error(self):
        w = self.dir / "x.workload"
        w.write_text(VALID_WORKLOAD + 'MAX_NUM_SEQS=""\n')
        issues = validate(w)
        errs = [m for sev, m in issues if sev == "error"]
        self.assertTrue(any("MAX_NUM_SEQS" in m and "empty" in m
                            for m in errs))

    def test_non_integer_int_var_reports_error(self):
        w = self.dir / "x.workload"
        w.write_text(VALID_WORKLOAD + "MAX_NUM_SEQS=not_a_number\n")
        issues = validate(w)
        errs = [m for sev, m in issues if sev == "error"]
        self.assertTrue(any("MAX_NUM_SEQS" in m and "integer" in m
                            for m in errs))

    def test_below_min_value_reports_error(self):
        w = self.dir / "x.workload"
        w.write_text(VALID_WORKLOAD + "MAX_NUM_SEQS=0\n")
        issues = validate(w)
        errs = [m for sev, m in issues if sev == "error"]
        self.assertTrue(any("MAX_NUM_SEQS" in m and ">= 1" in m
                            for m in errs))

    def test_max_num_batched_tokens_warns(self):
        """MNB is a service-sweep var; flagging it nudges operators
        toward the correct location (.service_axes.json)."""
        w = self.dir / "x.workload"
        w.write_text(VALID_WORKLOAD + 'MAX_NUM_BATCHED_TOKENS="8192"\n')
        issues = validate(w)
        warns = [m for sev, m in issues if sev == "warning"]
        self.assertTrue(any("MAX_NUM_BATCHED_TOKENS" in m
                            for m in warns))
        # No errors though.
        errs = [m for sev, m in issues if sev == "error"]
        self.assertEqual(errs, [])

    def test_kernel_derived_env_var_in_workload_warns(self):
        """RPA_KERNEL_K in .workload is a category mistake — it's
        kernel-derived, comes from the kernel registry."""
        w = self.dir / "x.workload"
        w.write_text(VALID_WORKLOAD + 'RPA_KERNEL_K="256"\n')
        issues = validate(w)
        warns = [m for sev, m in issues if sev == "warning"]
        self.assertTrue(any("RPA_KERNEL_K" in m for m in warns))

    def test_multiple_warnings_accumulate(self):
        w = self.dir / "x.workload"
        w.write_text(VALID_WORKLOAD
                     + 'MAX_NUM_BATCHED_TOKENS="8192"\n'
                     + 'BLOCK_SIZE="128"\n'
                     + 'RPA_KERNEL_K="256"\n')
        issues = validate(w)
        warns = [m for sev, m in issues if sev == "warning"]
        self.assertTrue(any("MAX_NUM_BATCHED_TOKENS" in m for m in warns))
        self.assertTrue(any("BLOCK_SIZE" in m for m in warns))
        self.assertTrue(any("RPA_KERNEL_K" in m for m in warns))


class TestCliMain(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_valid_workload_returns_0(self):
        w = self.dir / "x.workload"
        w.write_text(VALID_WORKLOAD)
        with mock.patch.object(sys, "stderr", new=open(os.devnull, "w")):
            rc = validate_main([str(w)])
        self.assertEqual(rc, 0)

    def test_missing_workload_returns_1(self):
        with mock.patch.object(sys, "stderr", new=open(os.devnull, "w")):
            rc = validate_main([str(self.dir / "absent.workload")])
        self.assertEqual(rc, 1)

    def test_invalid_workload_returns_1(self):
        w = self.dir / "x.workload"
        w.write_text('MODEL=""\n')   # missing nearly everything
        with mock.patch.object(sys, "stderr", new=open(os.devnull, "w")):
            rc = validate_main([str(w)])
        self.assertEqual(rc, 1)

    def test_warnings_only_returns_0(self):
        w = self.dir / "x.workload"
        w.write_text(VALID_WORKLOAD + 'MAX_NUM_BATCHED_TOKENS="8192"\n')
        with mock.patch.object(sys, "stderr", new=open(os.devnull, "w")):
            rc = validate_main([str(w)])
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
