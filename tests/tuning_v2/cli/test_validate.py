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
: "${DATASET:=sonnet}"
: "${NUM_PROMPTS:=1000}"
: "${REQUEST_RATE:=inf}"
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

    def test_missing_dataset_reports_error(self):
        """Category-2 measurement vars (arch doc §2 line 46) are
        REQUIRED — a typo like DATSET=sonnet should error at
        validation, not let bench fall back to defaults."""
        w = self.dir / "x.workload"
        wl = VALID_WORKLOAD.replace(': "${DATASET:=sonnet}"\n', "")
        w.write_text(wl)
        errs = [m for sev, m in validate(w) if sev == "error"]
        self.assertTrue(any("DATASET" in m for m in errs))

    def test_missing_num_prompts_reports_error(self):
        w = self.dir / "x.workload"
        wl = VALID_WORKLOAD.replace(': "${NUM_PROMPTS:=1000}"\n', "")
        w.write_text(wl)
        errs = [m for sev, m in validate(w) if sev == "error"]
        self.assertTrue(any("NUM_PROMPTS" in m for m in errs))

    def test_missing_request_rate_reports_error(self):
        w = self.dir / "x.workload"
        wl = VALID_WORKLOAD.replace(': "${REQUEST_RATE:=inf}"\n', "")
        w.write_text(wl)
        errs = [m for sev, m in validate(w) if sev == "error"]
        self.assertTrue(any("REQUEST_RATE" in m for m in errs))

    def test_request_rate_accepts_inf(self):
        """REQUEST_RATE accepts the string 'inf' (send-as-fast)
        as well as numerics. Validate only checks non-empty; the
        bench tool parses the actual value."""
        w = self.dir / "x.workload"
        w.write_text(VALID_WORKLOAD)   # default is inf
        self.assertEqual(validate(w), [])

    def test_request_rate_accepts_numeric(self):
        w = self.dir / "x.workload"
        w.write_text(VALID_WORKLOAD.replace(
            ': "${REQUEST_RATE:=inf}"\n',
            ': "${REQUEST_RATE:=10.5}"\n',
        ))
        self.assertEqual(validate(w), [])

    def test_num_prompts_zero_errors(self):
        w = self.dir / "x.workload"
        w.write_text(VALID_WORKLOAD + "NUM_PROMPTS=0\n")
        errs = [m for sev, m in validate(w) if sev == "error"]
        self.assertTrue(any("NUM_PROMPTS" in m and ">= 1" in m
                            for m in errs))

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
        w.write_text(VALID_WORKLOAD + 'MAX_MODEL_LEN=""\n')
        issues = validate(w)
        errs = [m for sev, m in issues if sev == "error"]
        self.assertTrue(any("MAX_MODEL_LEN" in m and "empty" in m
                            for m in errs))

    def test_non_integer_int_var_reports_error(self):
        w = self.dir / "x.workload"
        w.write_text(VALID_WORKLOAD + "MAX_MODEL_LEN=not_a_number\n")
        issues = validate(w)
        errs = [m for sev, m in issues if sev == "error"]
        self.assertTrue(any("MAX_MODEL_LEN" in m and "integer" in m
                            for m in errs))

    def test_below_min_value_reports_error(self):
        w = self.dir / "x.workload"
        w.write_text(VALID_WORKLOAD + "MAX_MODEL_LEN=0\n")
        issues = validate(w)
        errs = [m for sev, m in issues if sev == "error"]
        self.assertTrue(any("MAX_MODEL_LEN" in m and ">= 1" in m
                            for m in errs))

    def test_optional_mns_absent_is_ok(self):
        """MAX_NUM_SEQS is OPTIONAL per architecture-doc §2 line 73 —
        throughput scenarios omit it and let the service sweep find
        the best value. Validate should NOT error on absence."""
        w = self.dir / "x.workload"
        wl_without_mns = VALID_WORKLOAD.replace(
            ': "${MAX_NUM_SEQS:=128}"\n', "",
        )
        # MAX_NUM_SEQS isn't in VALID_WORKLOAD anymore by default,
        # but be defensive in case it gets added back.
        w.write_text(wl_without_mns)
        self.assertEqual(validate(w), [])

    def test_optional_mns_pinned_for_latency_validates_int(self):
        """MNS=1 in .workload is the latency scenario. Should
        validate as a positive int."""
        w = self.dir / "x.workload"
        w.write_text(VALID_WORKLOAD + 'MAX_NUM_SEQS="1"\n')
        self.assertEqual(validate(w), [])

    def test_optional_mns_non_integer_still_errors(self):
        """If the operator sets MNS to a bogus value, surface it."""
        w = self.dir / "x.workload"
        w.write_text(VALID_WORKLOAD + "MAX_NUM_SEQS=not_a_number\n")
        errs = [m for sev, m in validate(w) if sev == "error"]
        self.assertTrue(any("MAX_NUM_SEQS" in m for m in errs))

    def test_optional_mns_zero_errors(self):
        w = self.dir / "x.workload"
        w.write_text(VALID_WORKLOAD + "MAX_NUM_SEQS=0\n")
        errs = [m for sev, m in validate(w) if sev == "error"]
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


class TestFeasibilityPrecheck(unittest.TestCase):
    """prev-3: validate-step catches zero-feasible-combos workloads
    BEFORE any TPU time is spent. The May-11 prefill_heavy workload
    at MNS=1000 + default mnss multipliers yielded zero combos
    feasible against v7x's 13 GB bottom-of-memory cap; this check
    would have failed validate-step in milliseconds instead of
    burning 3.5 hours of tune time."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_silent_skip_when_kernel_module_unavailable(self):
        """On dev hosts without vllm/TPU deps, the feasibility check
        falls open silently — no spurious warnings on every laptop
        invocation. Verified by VALID_WORKLOAD returning []."""
        w = self.dir / "x.workload"
        w.write_text(VALID_WORKLOAD)
        # On this laptop, kernel module isn't importable. Precheck
        # must NOT emit a warning. Result must remain empty.
        self.assertEqual(validate(w), [])

    def test_error_when_zero_feasible_combos(self):
        """The May-11 failure mode. We mock the enumerator (and the
        flag) to act as though we're on TPU but yield no combos."""
        from unittest import mock
        w = self.dir / "x.workload"
        w.write_text(VALID_WORKLOAD)

        with mock.patch(
            "tools.tuning.v2.kernel.enumerate_logical."
            "_STATIC_PRUNE_AVAILABLE", True,
        ), mock.patch(
            "tools.tuning.v2.kernel.enumerate_logical."
            "enumerate_logical_combos", return_value=iter([]),
        ):
            issues = validate(w)

        errs = [m for sev, m in issues if sev == "error"]
        self.assertEqual(len(errs), 1)
        self.assertIn("feasibility precheck", errs[0])
        self.assertIn("ZERO combos", errs[0])

    def test_ok_when_at_least_one_feasible_combo(self):
        """Precheck early-outs after the first feasible combo —
        enumerator yields one, no error."""
        from unittest import mock
        w = self.dir / "x.workload"
        w.write_text(VALID_WORKLOAD)

        fake_combo = ({"case": "logical"}, {"mnss": 1000})
        with mock.patch(
            "tools.tuning.v2.kernel.enumerate_logical."
            "_STATIC_PRUNE_AVAILABLE", True,
        ), mock.patch(
            "tools.tuning.v2.kernel.enumerate_logical."
            "enumerate_logical_combos", return_value=iter([fake_combo]),
        ):
            issues = validate(w)

        errs = [m for sev, m in issues if sev == "error"]
        self.assertEqual(errs, [])

    def test_throughput_workload_resolves_mns_from_service_sweep(self):
        """Regression for the 2026-05-12 14:31 fail: a throughput
        workload (no MAX_NUM_SEQS pin) was getting MNS=1 / MNB=1
        defaulted inside _feasibility_issues, so every combo failed
        the `per_phys_q > MNB` constraint and validate reported
        'ZERO combos'.

        Now: when MAX_NUM_SEQS isn't pinned in the workload env, we
        resolve via service_search_space (mirroring tune.py). MNB
        always from service-sweep max.
        """
        from unittest import mock
        w = self.dir / "x.workload"
        # Strip MAX_NUM_SEQS from the workload (throughput scenario).
        # The real prefill_heavy.workload omits MAX_NUM_SEQS so that
        # the service sweep picks the value; this test mirrors that.
        wl = VALID_WORKLOAD.replace(': "${MAX_NUM_SEQS:=128}"\n', "")
        w.write_text(wl)

        captured_kwargs: dict = {}

        def _capture(*, max_num_seqs, max_num_batched_tokens, **_kw):
            captured_kwargs["mns"] = max_num_seqs
            captured_kwargs["mnb"] = max_num_batched_tokens
            return iter([({"case": "logical"}, {"mnss": 1000})])

        # The service_search_space module returns lists; max() gives
        # us the upper bound. Patch the function call from validate.
        with mock.patch(
            "tools.tuning.v2.kernel.enumerate_logical."
            "_STATIC_PRUNE_AVAILABLE", True,
        ), mock.patch(
            "tools.tuning.v2.service.search_space."
            "service_search_space",
            return_value={
                "MAX_NUM_SEQS":           [128, 256, 1000],
                "MAX_NUM_BATCHED_TOKENS": [8192, 16384, 131072],
            },
        ), mock.patch(
            "tools.tuning.v2.kernel.enumerate_logical."
            "enumerate_logical_combos", side_effect=_capture,
        ):
            issues = validate(w)

        # No errors — feasibility passed.
        errs = [m for sev, m in issues if sev == "error"]
        self.assertEqual(errs, [])
        # Confirms we passed the SERVICE-SWEEP MAX, not the
        # nonsensical default-1 from before this fix.
        self.assertEqual(captured_kwargs.get("mns"), 1000)
        self.assertEqual(captured_kwargs.get("mnb"), 131072)

    def test_skipped_when_prior_errors_present(self):
        """Don't cascade feasibility complaints on top of schema
        errors — they're confusing and feasibility may be malformed
        too. We force a schema error (missing required MODEL) and
        ALSO mock the enumerator to yield zero combos; the feasibility
        error must NOT appear because the prior schema error short-
        circuits the precheck."""
        from unittest import mock
        w = self.dir / "x.workload"
        # Strip the MODEL var → schema error.
        wl = VALID_WORKLOAD.replace(
            ': "${MODEL:=meta-llama/Meta-Llama-3-8B-Instruct}"\n', ""
        )
        w.write_text(wl)
        with mock.patch(
            "tools.tuning.v2.kernel.enumerate_logical."
            "_STATIC_PRUNE_AVAILABLE", True,
        ), mock.patch(
            "tools.tuning.v2.kernel.enumerate_logical."
            "enumerate_logical_combos", return_value=iter([]),
        ):
            issues = validate(w)
        errs = [m for sev, m in issues if sev == "error"]
        # MODEL error present, feasibility error NOT present.
        self.assertTrue(any("MODEL" in m for m in errs))
        self.assertFalse(any("feasibility" in m for m in errs))


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
