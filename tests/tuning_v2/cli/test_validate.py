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
        should skip it gracefully in both the snapshot and the sourced
        pass."""
        with mock.patch("subprocess.run") as run:
            run.side_effect = [
                # snapshot env
                mock.Mock(stdout="A=1\nbanner without equals\nB=2\n",
                          returncode=0),
                # sourced env (workload set C=3, plus banner)
                mock.Mock(stdout="A=1\nB=2\nC=3\nanother banner\n",
                          returncode=0),
            ]
            w = self.dir / "x.workload"
            w.write_text("placeholder")
            env = parse_workload_env(w)
        # Only newly-set vars surface; banner lines are dropped.
        self.assertEqual(env, {"C": "3"})

    def test_filters_out_inherited_shell_env(self):
        """Env vars from the calling shell (e.g. PATH, HOME) must NOT
        appear in the result; only vars actually set by the file."""
        w = self.dir / "x.workload"
        w.write_text(VALID_WORKLOAD)
        env = parse_workload_env(w)
        self.assertNotIn("PATH", env)
        self.assertNotIn("HOME", env)


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
""")
        issues = validate(w)
        errs = [m for sev, m in issues if sev == "error"]
        self.assertTrue(any("NUM_KV_HEADS" in m for m in errs))

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
