# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for tools/tuning/v2/scripts/*.sh.

Lightweight: verify each wrapper exists, is executable, has the right
shebang, prints a usage banner when invoked with no args (exit 1),
and that run_pipeline.sh calls each step in order.
"""

import os
import re
import stat
import subprocess
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "tools" / "tuning" / "v2" / "scripts"

ALL_SCRIPTS = (
    "tune_kernel.sh",
    "project_kernel.sh",
    "sweep_service.sh",
    "project_service.sh",
    "aggregate.sh",
    "lookup.sh",
    "validate.sh",
    "run_pipeline.sh",
)


class TestScriptFiles(unittest.TestCase):

    def test_all_scripts_exist(self):
        for name in ALL_SCRIPTS:
            with self.subTest(name=name):
                self.assertTrue(
                    (SCRIPTS_DIR / name).exists(),
                    f"missing script: {name}",
                )

    def test_all_scripts_are_executable(self):
        for name in ALL_SCRIPTS:
            with self.subTest(name=name):
                mode = (SCRIPTS_DIR / name).stat().st_mode
                self.assertTrue(
                    mode & stat.S_IXUSR,
                    f"script not executable: {name}",
                )

    def test_all_scripts_have_bash_shebang(self):
        for name in ALL_SCRIPTS:
            with self.subTest(name=name):
                first = (SCRIPTS_DIR / name).read_text().splitlines()[0]
                self.assertEqual(first, "#!/bin/bash")

    def test_all_scripts_have_set_e(self):
        """Per repo convention; ensures errors abort the script."""
        for name in ALL_SCRIPTS:
            with self.subTest(name=name):
                content = (SCRIPTS_DIR / name).read_text()
                self.assertIn("set -euo pipefail", content)


class TestUsageBanners(unittest.TestCase):
    """Each script (except run_pipeline) accepts a workload-ish arg
    and exits 1 with a usage banner when called with no args. We
    don't exec the underlying python module — too much setup."""

    def _invoke_no_args(self, name: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(SCRIPTS_DIR / name)],
            capture_output=True, text=True,
        )

    def test_tune_kernel_no_args_exits_1(self):
        result = self._invoke_no_args("tune_kernel.sh")
        self.assertEqual(result.returncode, 1)
        self.assertIn("Usage:", result.stderr)

    def test_project_kernel_no_args_exits_1(self):
        result = self._invoke_no_args("project_kernel.sh")
        self.assertEqual(result.returncode, 1)
        self.assertIn("Usage:", result.stderr)

    def test_sweep_service_no_args_exits_1(self):
        result = self._invoke_no_args("sweep_service.sh")
        self.assertEqual(result.returncode, 1)
        self.assertIn("Usage:", result.stderr)

    def test_project_service_no_args_exits_1(self):
        result = self._invoke_no_args("project_service.sh")
        self.assertEqual(result.returncode, 1)
        self.assertIn("Usage:", result.stderr)

    def test_aggregate_no_args_exits_1(self):
        result = self._invoke_no_args("aggregate.sh")
        self.assertEqual(result.returncode, 1)
        self.assertIn("Usage:", result.stderr)

    def test_lookup_no_args_exits_1(self):
        result = self._invoke_no_args("lookup.sh")
        self.assertEqual(result.returncode, 1)
        self.assertIn("Usage:", result.stderr)

    def test_validate_no_args_exits_1(self):
        result = self._invoke_no_args("validate.sh")
        self.assertEqual(result.returncode, 1)
        self.assertIn("Usage:", result.stderr)

    def test_run_pipeline_no_args_exits_1(self):
        result = self._invoke_no_args("run_pipeline.sh")
        self.assertEqual(result.returncode, 1)
        self.assertIn("Usage:", result.stderr)


class TestRunPipelineComposition(unittest.TestCase):
    """run_pipeline.sh must invoke each primitive in the documented
    order. We inspect the shell source rather than running it."""

    def test_invokes_all_six_steps_in_order(self):
        content = (SCRIPTS_DIR / "run_pipeline.sh").read_text()
        expected_order = [
            "validate.sh",
            "tune_kernel.sh",
            "project_kernel.sh",
            "sweep_service.sh",
            "project_service.sh",
            "aggregate.sh",
        ]
        positions = []
        for name in expected_order:
            idx = content.find(name)
            self.assertNotEqual(
                idx, -1,
                f"run_pipeline.sh does not reference {name}",
            )
            positions.append((name, idx))
        # Positions must be monotonically increasing.
        for i in range(1, len(positions)):
            prev_name, prev_idx = positions[i - 1]
            cur_name,  cur_idx  = positions[i]
            self.assertLess(
                prev_idx, cur_idx,
                f"run_pipeline.sh has {cur_name} before {prev_name}",
            )

    def test_supports_from_flag(self):
        """fix #16: `--from <step>` flag is documented and parsed."""
        content = (SCRIPTS_DIR / "run_pipeline.sh").read_text()
        self.assertIn("--from", content)
        # All step names must be valid `--from` targets.
        for step in ("validate", "tune_kernel", "project_kernel",
                     "sweep_service", "project_service", "aggregate"):
            with self.subTest(step=step):
                self.assertIn(step, content)

    def test_supports_extra_flags_passthrough(self):
        """Smoke runs need to thread --iters/--warmup etc. into the
        inner steps. The orchestrator reads EXTRA_*_FLAGS env vars
        and forwards them to each step. Asserts the contract is
        documented and the var names are stable."""
        content = (SCRIPTS_DIR / "run_pipeline.sh").read_text()
        for var in ("EXTRA_VALIDATE_FLAGS", "EXTRA_TUNE_FLAGS",
                    "EXTRA_PROJECT_KERNEL_FLAGS",
                    "EXTRA_SWEEP_FLAGS",
                    "EXTRA_PROJECT_SERVICE_FLAGS",
                    "EXTRA_AGGREGATE_FLAGS"):
            with self.subTest(var=var):
                self.assertIn(var, content)

    def test_extra_tune_flags_actually_threaded_at_runtime(self):
        """run_pipeline.sh + EXTRA_TUNE_FLAGS='--iters 1 --warmup 0'
        should reach tune_kernel.sh. Replace the wrapper scripts with
        echo-only stubs so we can observe what flags they receive."""
        import tempfile
        import subprocess
        with tempfile.TemporaryDirectory() as t:
            scripts_dir = Path(t) / "scripts"
            scripts_dir.mkdir()
            # Stub each wrapper to echo its args to a log file.
            for name in ("validate.sh", "tune_kernel.sh",
                         "project_kernel.sh", "sweep_service.sh",
                         "project_service.sh", "aggregate.sh"):
                stub = scripts_dir / name
                stub.write_text(
                    "#!/bin/bash\n"
                    f"echo {name} \"$@\" >> \"{t}/run.log\"\n"
                )
                stub.chmod(0o755)
            # Copy the real orchestrator into the stub dir.
            real = (SCRIPTS_DIR / "run_pipeline.sh").read_text()
            (scripts_dir / "run_pipeline.sh").write_text(real)
            (scripts_dir / "run_pipeline.sh").chmod(0o755)

            env = {
                **os.environ,
                "EXTRA_TUNE_FLAGS":  "--iters 1 --warmup 0",
                "EXTRA_SWEEP_FLAGS": "--timeout 60",
                # prev-5: the orchestrator's post-tune health check
                # reads the workload's kernel.raw to verify the tune
                # step actually produced rows. Stub wrappers don't
                # produce a raw store, so the check would fail this
                # composition test. KERNEL_TUNE_SKIP_HEALTH=1 bypasses
                # the check — appropriate for e2e composition tests
                # where we're verifying arg threading, not tune output.
                "KERNEL_TUNE_SKIP_HEALTH": "1",
            }
            subprocess.run(
                ["bash", str(scripts_dir / "run_pipeline.sh"),
                 "/tmp/fake.workload"],
                check=True, env=env,
            )
            log = (Path(t) / "run.log").read_text()
        self.assertIn("tune_kernel.sh /tmp/fake.workload "
                      "--iters 1 --warmup 0", log)
        self.assertIn("sweep_service.sh /tmp/fake.workload "
                      "--timeout 60", log)

    def test_no_dead_repo_root_variable(self):
        """fix #15: REPO_ROOT was assigned but never used — removed."""
        content = (SCRIPTS_DIR / "run_pipeline.sh").read_text()
        self.assertNotIn("REPO_ROOT=", content)

    def test_passes_workload_arg_to_sub_steps(self):
        """The orchestrator must forward the workload arg to each
        per-workload step (and the workload's parent dir to aggregate).
        Regex allows either direct invocation (`tune_kernel.sh "$X"`)
        or via a helper function (`run_step ... tune_kernel.sh "$X"`)."""
        content = (SCRIPTS_DIR / "run_pipeline.sh").read_text()
        for script in (
            "tune_kernel", "project_kernel",
            "sweep_service", "project_service",
        ):
            with self.subTest(script=script):
                # `tune_kernel.sh` followed (eventually on same line)
                # by "$WORKLOAD".
                self.assertTrue(re.search(
                    rf"{script}\.sh\b[^\n]*\"\$WORKLOAD\"",
                    content,
                ))
        # aggregate gets the workload's PARENT dir.
        self.assertTrue(re.search(
            r'aggregate\.sh\b[^\n]*"\$WORKLOAD_DIR"',
            content,
        ))


class TestRealExecValidate(unittest.TestCase):
    """fix #10 / #22: actually run validate.sh through bash with a
    real (synthetic) workload. Exercises the shell wrapper + Python
    module + clean-subshell env-diff (fix #4) end-to-end."""

    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, *args):
        return subprocess.run(
            ["bash", str(SCRIPTS_DIR / "validate.sh"), *args],
            capture_output=True, text=True,
            cwd=str(SCRIPTS_DIR.parents[3]),  # repo root for PYTHONPATH
        )

    def test_validates_a_real_workload_file(self):
        wl = self.dir / "x.workload"
        wl.write_text(
            ': "${MODEL:=foo}"\n'
            ': "${TENSOR_PARALLEL_SIZE:=1}"\n'
            ': "${NUM_Q_HEADS:=32}"\n'
            ': "${NUM_KV_HEADS:=8}"\n'
            ': "${HEAD_DIM:=128}"\n'
            ': "${MAX_MODEL_LEN:=8192}"\n'
            ': "${MAX_NUM_SEQS:=128}"\n'
            ': "${INPUT_LEN:=8191}"\n'
            ': "${OUTPUT_LEN:=1}"\n'
            ': "${DATASET:=sonnet}"\n'
            ': "${NUM_PROMPTS:=1000}"\n'
            ': "${REQUEST_RATE:=inf}"\n'
        )
        result = self._run(str(wl))
        self.assertEqual(
            result.returncode, 0,
            f"validate.sh failed: stdout={result.stdout!r} "
            f"stderr={result.stderr!r}",
        )

    def test_invalid_workload_exits_nonzero(self):
        wl = self.dir / "bad.workload"
        # MAX_MODEL_LEN < INPUT_LEN + OUTPUT_LEN -> invariant violation.
        wl.write_text(
            ': "${MODEL:=foo}"\n'
            ': "${TENSOR_PARALLEL_SIZE:=1}"\n'
            ': "${NUM_Q_HEADS:=32}"\n'
            ': "${NUM_KV_HEADS:=8}"\n'
            ': "${HEAD_DIM:=128}"\n'
            ': "${MAX_MODEL_LEN:=512}"\n'
            ': "${MAX_NUM_SEQS:=128}"\n'
            ': "${INPUT_LEN:=8000}"\n'
            ': "${OUTPUT_LEN:=1}"\n'
            ': "${DATASET:=sonnet}"\n'
            ': "${NUM_PROMPTS:=1000}"\n'
            ': "${REQUEST_RATE:=inf}"\n'
        )
        result = self._run(str(wl))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("INPUT_LEN+OUTPUT_LEN", result.stderr)


class TestWrapperDelegation(unittest.TestCase):
    """Each thin wrapper should delegate to a python -m invocation."""

    DELEGATIONS = {
        "tune_kernel.sh":     "tools.tuning.v2.kernel.tune",
        "project_kernel.sh":  "tools.tuning.v2.kernel.project",
        "sweep_service.sh":   "tools.tuning.v2.service.sweep",
        "project_service.sh": "tools.tuning.v2.service.project",
        "aggregate.sh":       "tools.tuning.v2.cli.aggregate",
        "lookup.sh":          "tools.tuning.v2.cli.lookup",
        "validate.sh":        "tools.tuning.v2.cli.validate",
    }

    def test_each_wrapper_invokes_expected_module(self):
        for script, module in self.DELEGATIONS.items():
            with self.subTest(script=script):
                content = (SCRIPTS_DIR / script).read_text()
                self.assertIn(f"python3 -m {module}", content)


if __name__ == "__main__":
    unittest.main()
