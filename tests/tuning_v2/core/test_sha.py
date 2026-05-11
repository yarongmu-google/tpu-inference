# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for tools.tuning.v2.core.sha.

We exercise the SHA helpers against ad-hoc git repos created in temp
directories — avoids relying on whatever state the running tpu_inference
checkout happens to be in. Covers: happy path, missing dir, missing
.git, missing git binary on PATH, non-zero exit, blank output, paired
service_sha encoding.
"""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.tuning.v2.core.sha import kernel_sha, service_sha


def _init_git_repo(path: Path) -> str:
    """Create a one-commit git repo at `path`. Return its full SHA."""
    path.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Test", "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test", "GIT_COMMITTER_EMAIL": "test@example.com",
    }
    subprocess.run(["git", "init", "-q"], cwd=path, check=True, env=env)
    subprocess.run(["git", "config", "commit.gpgsign", "false"],
                   cwd=path, check=True, env=env)
    (path / "f.txt").write_text("hello\n")
    subprocess.run(["git", "add", "f.txt"], cwd=path, check=True, env=env)
    subprocess.run(["git", "commit", "-q", "-m", "init"],
                   cwd=path, check=True, env=env)
    full = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path, check=True, capture_output=True, text=True, env=env,
    ).stdout.strip()
    return full


class TestKernelSha(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_kernel_sha_returns_8_char_sha_of_real_repo(self):
        repo = self.dir / "repo"
        full = _init_git_repo(repo)
        sha = kernel_sha(repo)
        self.assertEqual(len(sha), 8)
        self.assertEqual(sha, full[:8])

    def test_kernel_sha_unknown_when_dir_missing(self):
        self.assertEqual(kernel_sha(self.dir / "does_not_exist"), "unknown")

    def test_kernel_sha_unknown_when_not_a_git_repo(self):
        plain = self.dir / "not_a_repo"
        plain.mkdir()
        (plain / "f.txt").write_text("hi")
        self.assertEqual(kernel_sha(plain), "unknown")

    def test_kernel_sha_unknown_when_git_command_fails(self):
        """Mock subprocess to simulate non-zero exit code (corrupted repo)."""
        repo = self.dir / "repo"
        _init_git_repo(repo)
        with mock.patch("subprocess.run") as run:
            run.return_value = mock.Mock(returncode=128, stdout="")
            self.assertEqual(kernel_sha(repo), "unknown")

    def test_kernel_sha_unknown_when_git_binary_missing(self):
        """FileNotFoundError is what subprocess raises when `git` isn't on PATH."""
        repo = self.dir / "repo"
        _init_git_repo(repo)
        with mock.patch("subprocess.run", side_effect=FileNotFoundError):
            self.assertEqual(kernel_sha(repo), "unknown")

    def test_kernel_sha_unknown_when_git_times_out(self):
        repo = self.dir / "repo"
        _init_git_repo(repo)
        with mock.patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="git", timeout=5),
        ):
            self.assertEqual(kernel_sha(repo), "unknown")

    def test_kernel_sha_unknown_when_oserror(self):
        repo = self.dir / "repo"
        _init_git_repo(repo)
        with mock.patch("subprocess.run", side_effect=OSError("nope")):
            self.assertEqual(kernel_sha(repo), "unknown")

    def test_kernel_sha_unknown_when_blank_output(self):
        """Some edge cases (detached uninitialised HEAD) give blank stdout."""
        repo = self.dir / "repo"
        _init_git_repo(repo)
        with mock.patch("subprocess.run") as run:
            run.return_value = mock.Mock(returncode=0, stdout="   \n")
            self.assertEqual(kernel_sha(repo), "unknown")

    def test_kernel_sha_default_repo_root(self):
        """When called with no arg, infers repo root from this file's path.
        Just check it returns a string, since we can't predict the actual SHA
        of the developer's checkout."""
        sha = kernel_sha()
        self.assertIsInstance(sha, str)
        self.assertTrue(len(sha) > 0)


class TestServiceSha(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_service_sha_joins_both_repos(self):
        inf = self.dir / "tpu_inference"
        vllm = self.dir / "vllm"
        inf_full = _init_git_repo(inf)
        vllm_full = _init_git_repo(vllm)
        sha = service_sha(repo_root=inf, vllm_dir=vllm)
        self.assertEqual(sha, f"{inf_full[:8]}-{vllm_full[:8]}")

    def test_service_sha_vllm_unknown_when_missing(self):
        inf = self.dir / "tpu_inference"
        inf_full = _init_git_repo(inf)
        sha = service_sha(repo_root=inf, vllm_dir=self.dir / "no_vllm")
        self.assertEqual(sha, f"{inf_full[:8]}-unknown")

    def test_service_sha_inf_unknown_when_missing(self):
        vllm = self.dir / "vllm"
        vllm_full = _init_git_repo(vllm)
        sha = service_sha(repo_root=self.dir / "no_inf", vllm_dir=vllm)
        self.assertEqual(sha, f"unknown-{vllm_full[:8]}")

    def test_service_sha_both_unknown(self):
        sha = service_sha(
            repo_root=self.dir / "no_inf",
            vllm_dir=self.dir / "no_vllm",
        )
        self.assertEqual(sha, "unknown-unknown")

    def test_service_sha_defaults_to_sibling_vllm(self):
        """Default vllm_dir is `<repo_root>/../vllm`. Verify that path is
        consulted when not provided."""
        inf = self.dir / "tpu_inference"
        vllm = self.dir / "vllm"   # sibling of inf
        inf_full = _init_git_repo(inf)
        vllm_full = _init_git_repo(vllm)
        sha = service_sha(repo_root=inf)
        self.assertEqual(sha, f"{inf_full[:8]}-{vllm_full[:8]}")

    def test_service_sha_default_repo_root_returns_string(self):
        sha = service_sha()
        self.assertIsInstance(sha, str)
        self.assertIn("-", sha)


if __name__ == "__main__":
    unittest.main()
