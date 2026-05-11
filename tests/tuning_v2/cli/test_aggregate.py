# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for tools.tuning.v2.cli.aggregate."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.tuning.v2.cli.aggregate import aggregate, main as aggregate_main
from tools.tuning.v2.core.git_atomic import NO_PUSH_ENV


def _init_git_repo(d: Path) -> None:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Test", "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "Test", "GIT_COMMITTER_EMAIL": "t@example.com",
    }
    subprocess.run(["git", "init", "-q"], cwd=d, check=True, env=env)
    subprocess.run(["git", "config", "commit.gpgsign", "false"],
                   cwd=d, check=True, env=env)
    (d / "init.txt").write_text("hi")
    subprocess.run(["git", "add", "init.txt"], cwd=d, check=True, env=env)
    subprocess.run(["git", "commit", "-q", "-m", "init"],
                   cwd=d, check=True, env=env)


class TestAggregate(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cases = Path(self.tmp.name) / "cases" / "v7x" / "llama3_8b"
        self.cases.mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_writes_both_production_files_when_inputs_exist(self):
        (self.cases / "alpha.kernel").write_text('{"winners": []}')
        (self.cases / "bravo.kernel").write_text('{"winners": [1]}')
        (self.cases / "alpha.service").write_text('{"winners": {}}')
        kp, sp = aggregate(self.cases)
        self.assertEqual(kp, self.cases / "production.kernel")
        self.assertEqual(sp, self.cases / "production.service")
        kdoc = json.loads(kp.read_text())
        self.assertEqual(set(kdoc["by_workload"].keys()), {"alpha", "bravo"})
        self.assertEqual(kdoc["topo"], "v7x")
        self.assertEqual(kdoc["model"], "llama3_8b")

    def test_no_kernel_files_returns_none_for_kernel(self):
        (self.cases / "alpha.service").write_text('{"winners": {}}')
        kp, sp = aggregate(self.cases)
        self.assertIsNone(kp)
        self.assertEqual(sp, self.cases / "production.service")

    def test_no_service_files_returns_none_for_service(self):
        (self.cases / "alpha.kernel").write_text('{"winners": []}')
        kp, sp = aggregate(self.cases)
        self.assertEqual(kp, self.cases / "production.kernel")
        self.assertIsNone(sp)

    def test_no_files_at_all_returns_both_none(self):
        kp, sp = aggregate(self.cases)
        self.assertIsNone(kp)
        self.assertIsNone(sp)

    def test_excludes_existing_production_files(self):
        """A pre-existing production.kernel must not be folded back."""
        (self.cases / "production.kernel").write_text(
            '{"by_workload": {"stale": {}}}',
        )
        (self.cases / "alpha.kernel").write_text('{"winners": []}')
        kp, _sp = aggregate(self.cases)
        kdoc = json.loads(kp.read_text())
        self.assertNotIn("stale", kdoc["by_workload"])
        self.assertIn("alpha", kdoc["by_workload"])

    def test_explicit_topo_model_overrides_inferred(self):
        (self.cases / "alpha.kernel").write_text('{"winners": []}')
        kp, _sp = aggregate(self.cases, topo="custom_topo", model="custom_model")
        kdoc = json.loads(kp.read_text())
        self.assertEqual(kdoc["topo"], "custom_topo")
        self.assertEqual(kdoc["model"], "custom_model")


class TestCliMain(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        _init_git_repo(self.repo)
        self.cases = self.repo / "cases" / "v7x" / "llama3_8b"
        self.cases.mkdir(parents=True)
        self._saved_no_push = os.environ.pop(NO_PUSH_ENV, None)
        os.environ[NO_PUSH_ENV] = "1"

    def tearDown(self):
        if self._saved_no_push is None:
            os.environ.pop(NO_PUSH_ENV, None)
        else:
            os.environ[NO_PUSH_ENV] = self._saved_no_push
        self.tmp.cleanup()

    def test_main_missing_dir_returns_1(self):
        with mock.patch.object(sys, "stderr", new=open(os.devnull, "w")):
            rc = aggregate_main([str(self.repo / "no_such_dir")])
        self.assertEqual(rc, 1)

    def test_main_not_a_directory_returns_1(self):
        f = self.repo / "file.txt"
        f.write_text("hi")
        with mock.patch.object(sys, "stderr", new=open(os.devnull, "w")):
            rc = aggregate_main([str(f)])
        self.assertEqual(rc, 1)

    def test_main_empty_model_dir_returns_1(self):
        with mock.patch.object(sys, "stderr", new=open(os.devnull, "w")):
            rc = aggregate_main([str(self.cases)])
        self.assertEqual(rc, 1)

    def test_main_writes_files_returns_0(self):
        (self.cases / "alpha.kernel").write_text('{"winners": []}')
        (self.cases / "alpha.service").write_text('{"winners": {}}')
        with mock.patch.object(sys, "stdout", new=open(os.devnull, "w")):
            rc = aggregate_main([str(self.cases), "--no-commit"])
        self.assertEqual(rc, 0)
        self.assertTrue((self.cases / "production.kernel").exists())
        self.assertTrue((self.cases / "production.service").exists())

    def test_main_commits_by_default(self):
        (self.cases / "alpha.kernel").write_text('{"winners": []}')
        with mock.patch(
            "tools.tuning.v2.cli.aggregate.commit_and_push"
        ) as cap:
            with mock.patch.object(sys, "stdout", new=open(os.devnull, "w")):
                rc = aggregate_main([str(self.cases)])
        self.assertEqual(rc, 0)
        cap.assert_called_once()

    def test_main_explicit_topo_and_model(self):
        (self.cases / "alpha.kernel").write_text('{"winners": []}')
        with mock.patch.object(sys, "stdout", new=open(os.devnull, "w")):
            rc = aggregate_main([
                str(self.cases),
                "--topo", "MY_TOPO",
                "--model", "MY_MODEL",
                "--no-commit",
            ])
        self.assertEqual(rc, 0)
        doc = json.loads((self.cases / "production.kernel").read_text())
        self.assertEqual(doc["topo"], "MY_TOPO")
        self.assertEqual(doc["model"], "MY_MODEL")


if __name__ == "__main__":
    unittest.main()
