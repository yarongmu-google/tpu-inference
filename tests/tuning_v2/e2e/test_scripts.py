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

    def test_passes_workload_arg_to_sub_steps(self):
        """The orchestrator must forward the workload arg to each
        per-workload step (and the workload's parent dir to aggregate)."""
        content = (SCRIPTS_DIR / "run_pipeline.sh").read_text()
        # workload var name should appear in each step call.
        self.assertTrue(re.search(r'tune_kernel\.sh"\s+"\$WORKLOAD"', content))
        self.assertTrue(re.search(r'project_kernel\.sh"\s+"\$WORKLOAD"', content))
        self.assertTrue(re.search(r'sweep_service\.sh"\s+"\$WORKLOAD"', content))
        self.assertTrue(re.search(r'project_service\.sh"\s+"\$WORKLOAD"', content))
        # aggregate gets the workload's PARENT dir.
        self.assertTrue(re.search(r'aggregate\.sh"\s+"\$WORKLOAD_DIR"', content))


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
