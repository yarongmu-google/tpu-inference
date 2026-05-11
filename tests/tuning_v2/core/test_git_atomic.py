# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for tools.tuning.v2.core.git_atomic.

Uses real git repos in temp dirs (local file:// "remote") for happy-path
testing; mocks subprocess for failure-path coverage. Each test runs in
isolation — no reliance on the running tpu_inference checkout's git state.
"""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.tuning.v2.core.git_atomic import (
    NO_PUSH_ENV,
    commit_and_push,
    push_disabled,
)


def _run_git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Test", "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "Test", "GIT_COMMITTER_EMAIL": "t@example.com",
    }
    return subprocess.run(
        ["git"] + args, cwd=cwd, env=env,
        capture_output=True, text=True, check=True,
    )


def _init_repo_with_remote(d: Path) -> tuple[Path, Path]:
    """Create a bare 'remote' and a working 'repo' that pushes to it.
    Returns (repo_path, remote_path)."""
    remote = d / "remote.git"
    repo = d / "repo"
    remote.mkdir()
    _run_git(["init", "--bare", "-q"], cwd=remote)
    repo.mkdir()
    _run_git(["init", "-q"], cwd=repo)
    _run_git(["config", "commit.gpgsign", "false"], cwd=repo)
    _run_git(["remote", "add", "origin", str(remote)], cwd=repo)
    (repo / "init.txt").write_text("hello\n")
    _run_git(["add", "init.txt"], cwd=repo)
    _run_git(["commit", "-q", "-m", "init"], cwd=repo)
    _run_git(["branch", "-M", "main"], cwd=repo)
    _run_git(["push", "-u", "origin", "main"], cwd=repo)
    return repo, remote


class TestPushDisabled(unittest.TestCase):

    def test_disabled_when_env_set_to_1(self):
        with mock.patch.dict(os.environ, {NO_PUSH_ENV: "1"}):
            self.assertTrue(push_disabled())

    def test_enabled_when_env_unset(self):
        env = {k: v for k, v in os.environ.items() if k != NO_PUSH_ENV}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertFalse(push_disabled())

    def test_enabled_when_env_set_to_other_value(self):
        with mock.patch.dict(os.environ, {NO_PUSH_ENV: "0"}):
            self.assertFalse(push_disabled())
        with mock.patch.dict(os.environ, {NO_PUSH_ENV: "yes"}):
            self.assertFalse(push_disabled())


class TestCommitAndPush(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        # Push is disabled by default in tests so we don't accidentally
        # hit a network or shell out unexpectedly. Each test enables
        # explicitly via the temp remote.
        self._saved_env = os.environ.pop(NO_PUSH_ENV, None)
        os.environ[NO_PUSH_ENV] = "1"

    def tearDown(self):
        if self._saved_env is None:
            os.environ.pop(NO_PUSH_ENV, None)
        else:
            os.environ[NO_PUSH_ENV] = self._saved_env
        self.tmp.cleanup()

    def test_empty_paths_returns_false(self):
        self.assertFalse(commit_and_push([], "msg"))

    def test_no_repo_returns_false(self):
        f = self.dir / "lonely.txt"
        f.write_text("x")
        self.assertFalse(commit_and_push([f], "msg"))

    def test_commit_succeeds_push_disabled_returns_false(self):
        repo, _ = _init_repo_with_remote(self.dir)
        new_file = repo / "new.txt"
        new_file.write_text("hi")
        # NO_PUSH=1, so push skipped → False.
        self.assertFalse(commit_and_push([new_file], "Add new.txt"))
        # But the commit should have landed locally.
        log = _run_git(["log", "--format=%s", "-n", "1"], cwd=repo)
        self.assertEqual(log.stdout.strip(), "Add new.txt")

    def test_commit_and_push_returns_true(self):
        repo, _ = _init_repo_with_remote(self.dir)
        # Enable push via env override.
        os.environ.pop(NO_PUSH_ENV)
        new_file = repo / "added.txt"
        new_file.write_text("hello")
        self.assertTrue(commit_and_push([new_file], "Add added.txt"))
        # Local commit landed.
        log = _run_git(["log", "--format=%s", "-n", "1"], cwd=repo)
        self.assertEqual(log.stdout.strip(), "Add added.txt")

    def test_nothing_to_commit_returns_false(self):
        repo, _ = _init_repo_with_remote(self.dir)
        os.environ.pop(NO_PUSH_ENV)
        # init.txt already committed; commit_and_push of same file = no-op.
        self.assertFalse(
            commit_and_push([repo / "init.txt"], "no-op"),
        )

    def test_path_restricted_commit_does_not_pick_up_other_staged(self):
        """If the user has unrelated changes staged manually, our commit
        should not fold them in. Path-restricted commit guards this."""
        repo, _ = _init_repo_with_remote(self.dir)
        os.environ.pop(NO_PUSH_ENV)
        # User stages a different file outside our control.
        other = repo / "user_unrelated.txt"
        other.write_text("user's work")
        _run_git(["add", "user_unrelated.txt"], cwd=repo)
        # We commit OUR file.
        ours = repo / "ours.txt"
        ours.write_text("our work")
        self.assertTrue(commit_and_push([ours], "Add ours"))
        # Check the latest commit only carries ours.txt.
        show = _run_git(
            ["show", "--name-only", "--format=", "HEAD"], cwd=repo,
        )
        committed = [p for p in show.stdout.split() if p]
        self.assertEqual(committed, ["ours.txt"])

    def test_add_failure_returns_false(self):
        """If git add itself fails (e.g. invalid pathspec syntax), we
        return False rather than blowing up the caller."""
        repo, _ = _init_repo_with_remote(self.dir)
        with mock.patch(
            "tools.tuning.v2.core.git_atomic._git"
        ) as g:
            # rev-parse --is-inside-work-tree → ok
            # add → fail
            g.side_effect = [
                mock.Mock(returncode=0, stdout="true\n"),
                mock.Mock(returncode=128, stderr="bad path"),
            ]
            self.assertFalse(
                commit_and_push([repo / "x.txt"], "msg",
                                repo_root=repo),
            )

    def test_push_failure_returns_false(self):
        """If push fails (network, divergence, …), commit was made
        locally but the function returns False. Caller's tune is
        unaffected."""
        repo, _ = _init_repo_with_remote(self.dir)
        os.environ.pop(NO_PUSH_ENV)
        # Break the remote URL so push fails.
        _run_git(["remote", "set-url", "origin",
                  "/nonexistent/path.git"], cwd=repo)
        new_file = repo / "broken_push.txt"
        new_file.write_text("hi")
        self.assertFalse(
            commit_and_push([new_file], "Push will fail"),
        )
        # Commit landed locally even though push failed.
        log = _run_git(["log", "--format=%s", "-n", "1"], cwd=repo)
        self.assertEqual(log.stdout.strip(), "Push will fail")

    def test_detached_head_skips_push(self):
        repo, _ = _init_repo_with_remote(self.dir)
        os.environ.pop(NO_PUSH_ENV)
        # Detach HEAD.
        sha = _run_git(["rev-parse", "HEAD"], cwd=repo).stdout.strip()
        _run_git(["checkout", "-q", sha], cwd=repo)
        new_file = repo / "detached.txt"
        new_file.write_text("hi")
        # Commit lands; push skipped because we can't infer branch.
        self.assertFalse(commit_and_push([new_file], "Detached"))

    def test_explicit_repo_root_arg(self):
        """When repo_root is given, we don't walk up from the path."""
        repo, _ = _init_repo_with_remote(self.dir)
        os.environ.pop(NO_PUSH_ENV)
        new_file = repo / "explicit.txt"
        new_file.write_text("hi")
        self.assertTrue(
            commit_and_push([new_file], "Explicit root",
                            repo_root=repo),
        )

    def test_repo_root_arg_not_a_git_repo(self):
        """Explicit repo_root that isn't a repo returns False."""
        not_a_repo = self.dir / "plain"
        not_a_repo.mkdir()
        (not_a_repo / "f.txt").write_text("hi")
        self.assertFalse(
            commit_and_push(
                [not_a_repo / "f.txt"],
                "won't commit",
                repo_root=not_a_repo,
            ),
        )

    def test_walks_up_to_find_repo_root(self):
        repo, _ = _init_repo_with_remote(self.dir)
        os.environ.pop(NO_PUSH_ENV)
        # Path nested inside the repo.
        nested = repo / "a" / "b" / "c"
        nested.mkdir(parents=True)
        new_file = nested / "deep.txt"
        new_file.write_text("hi")
        self.assertTrue(commit_and_push([new_file], "Deep file"))

    def test_walk_up_returns_none_outside_any_repo(self):
        """Path under tmp (no parent .git) returns False."""
        plain = self.dir / "no_repo_anywhere" / "deep"
        plain.mkdir(parents=True)
        (plain / "x.txt").write_text("hi")
        self.assertFalse(commit_and_push([plain / "x.txt"], "msg"))

    def test_handles_git_binary_missing(self):
        """FileNotFoundError when git isn't installed."""
        repo, _ = _init_repo_with_remote(self.dir)
        with mock.patch(
            "subprocess.run", side_effect=FileNotFoundError,
        ):
            # _is_inside_git_repo returns False → early return.
            self.assertFalse(
                commit_and_push([repo / "init.txt"], "msg",
                                repo_root=repo),
            )

    def test_current_branch_handles_nonzero_returncode(self):
        """rev-parse failures inside the branch lookup return None;
        commit_and_push skips push but still returns False without
        crashing."""
        repo, _ = _init_repo_with_remote(self.dir)
        os.environ.pop(NO_PUSH_ENV)
        new_file = repo / "branch_fail.txt"
        new_file.write_text("hi")
        # First two _git calls (is_inside, add) succeed; third (commit)
        # succeeds; fourth (rev-parse for current branch) fails with
        # non-zero exit.
        real_git = __import__(
            "tools.tuning.v2.core.git_atomic", fromlist=["_git"]
        )._git

        call_idx = [0]
        def fake_git(*args, **kwargs):
            call_idx[0] += 1
            if call_idx[0] == 4:    # the current-branch lookup
                return mock.Mock(returncode=1, stdout="")
            return real_git(*args, **kwargs)

        with mock.patch(
            "tools.tuning.v2.core.git_atomic._git", side_effect=fake_git,
        ):
            self.assertFalse(commit_and_push([new_file], "msg",
                                             repo_root=repo))

    def test_current_branch_handles_oserror(self):
        repo, _ = _init_repo_with_remote(self.dir)
        os.environ.pop(NO_PUSH_ENV)
        new_file = repo / "oserror.txt"
        new_file.write_text("hi")
        real_git = __import__(
            "tools.tuning.v2.core.git_atomic", fromlist=["_git"]
        )._git

        call_idx = [0]
        def fake_git(*args, **kwargs):
            call_idx[0] += 1
            if call_idx[0] == 4:    # current-branch lookup
                raise OSError("no")
            return real_git(*args, **kwargs)

        with mock.patch(
            "tools.tuning.v2.core.git_atomic._git", side_effect=fake_git,
        ):
            self.assertFalse(commit_and_push([new_file], "msg",
                                             repo_root=repo))

    def test_path_argument_can_be_directory(self):
        """The walk-up helper accepts a directory path (start is not
        a file), not just a file."""
        repo, _ = _init_repo_with_remote(self.dir)
        os.environ.pop(NO_PUSH_ENV)
        # Add a new file then ask commit_and_push to stage its parent dir.
        new_dir = repo / "subdir"
        new_dir.mkdir()
        (new_dir / "f.txt").write_text("hi")
        # Pass the DIRECTORY as the path. git add -f <dir> stages
        # all files under it.
        self.assertTrue(commit_and_push([new_dir], "Add subdir"))


if __name__ == "__main__":
    unittest.main()
