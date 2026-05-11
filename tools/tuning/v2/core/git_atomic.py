# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Git commit-and-push helper for the tuning pipeline.

Both long-running tune/sweep and short-running project/aggregate need to
land their artifacts on the remote. v1 had this scattered across three
scripts with subtly different conventions; v2 factors it into one
helper that both use.

  commit_and_push(paths, message)
    -> stages the paths, commits if anything was staged, pushes unless
       disabled via env. Returns True iff a push was performed.

Conventions inherited from v1:

- Auto-push is **opt-out** (default on). Set `KERNEL_TUNER_NO_PUSH=1`
  in env to disable — matches the gate that tune_all_cases.sh and
  build_kernel_registry.sh already use.
- A "no-op commit" (nothing staged because the paths haven't changed)
  is not an error — function returns False without raising.
- A push failure (network, divergence, etc.) is not an error —
  function logs a warning to stderr and returns False. Caller chose
  to push; we don't crash their tune.

Caller patterns:

  # Long run (kernel tune, service sweep): periodic checkpoint.
  for batch in batches_of_n:
      run_batch(batch)
      commit_and_push([raw_path],
                      f"[Tune-v2] progress: {n_done} cases")

  # Short run (project, aggregate): once at the end.
  write_output(out_path, winners)
  commit_and_push([out_path],
                  f"[Tune-v2] Update {out_path.name}")
"""

import os
import subprocess
import sys
from pathlib import Path


# Env var that disables push. Honored verbatim from v1 so users with the
# muscle memory don't have to relearn.
NO_PUSH_ENV = "KERNEL_TUNER_NO_PUSH"


def _git(cmd: list[str], cwd: Path, check: bool = False) -> subprocess.CompletedProcess:
    """Run a git command in `cwd`. Returns the completed process.

    `check=False` (default): caller inspects returncode. `check=True`
    raises on non-zero. We don't use `check=True` here because git's
    exit codes carry meaning (e.g. `commit` exit 1 = nothing to commit,
    which is fine).
    """
    return subprocess.run(
        ["git"] + cmd, cwd=cwd, capture_output=True, text=True, check=check,
    )


def _is_inside_git_repo(path: Path) -> bool:
    """True if `path` is inside a git working tree."""
    try:
        result = _git(["rev-parse", "--is-inside-work-tree"], cwd=path)
        return result.returncode == 0 and result.stdout.strip() == "true"
    except (FileNotFoundError, OSError):
        return False


def _current_branch(path: Path) -> str | None:
    """Return the current branch name, or None on detached HEAD / error."""
    try:
        result = _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=path)
        if result.returncode != 0:
            return None
        branch = result.stdout.strip()
        return branch if branch and branch != "HEAD" else None
    except (FileNotFoundError, OSError):
        return None


def push_disabled() -> bool:
    """Check whether the NO_PUSH env var is set to '1'."""
    return os.environ.get(NO_PUSH_ENV) == "1"


def commit_and_push(
    paths: list[Path],
    message: str,
    *,
    repo_root: Path | None = None,
) -> bool:
    """Stage `paths`, commit if staged anything, push if not disabled.

    Args:
      paths: Files (or directories) to stage. Forwarded to `git add -f`.
      message: Commit message body.
      repo_root: Repo to operate in. Defaults to the parent of the
                 first path's resolve()d location, walking up until
                 finding a .git/.

    Returns:
      True iff a push actually happened (commit created AND push
      succeeded). False if nothing to commit, push disabled by env,
      or push failed.

    Errors are reported via stderr; the function does not raise on
    typical operational failures — the caller's main work
    (tune/sweep/project) should not be aborted by a git hiccup.
    """
    if not paths:
        return False

    if repo_root is None:
        repo_root = _find_repo_root(paths[0])
        if repo_root is None:
            print(
                f"git_atomic: no git repo found for {paths[0]}; "
                "skipping commit.",
                file=sys.stderr,
            )
            return False

    if not _is_inside_git_repo(repo_root):
        print(
            f"git_atomic: {repo_root} is not a git repo; skipping commit.",
            file=sys.stderr,
        )
        return False

    # Stage. `-f` lets us add gitignored files (we use it for runlogs
    # under tmp/ that we want to commit despite gitignore).
    add_args = ["add", "-f", "--"] + [str(p) for p in paths]
    add = _git(add_args, cwd=repo_root)
    if add.returncode != 0:
        print(
            f"git_atomic: git add failed: {add.stderr.strip()}",
            file=sys.stderr,
        )
        return False

    # Commit. Path-restricted so we don't pick up unrelated staged
    # changes from the user's worktree.
    commit_args = ["commit", "-m", message, "--"] + [str(p) for p in paths]
    commit = _git(commit_args, cwd=repo_root)
    if commit.returncode != 0:
        # Most common case: nothing to commit (paths already clean).
        # Not an error.
        return False

    if push_disabled():
        return False

    branch = _current_branch(repo_root)
    if branch is None:
        print(
            "git_atomic: detached HEAD or branch lookup failed; "
            "skipping push.",
            file=sys.stderr,
        )
        return False

    push = _git(["push", "origin", branch], cwd=repo_root)
    if push.returncode != 0:
        print(
            f"git_atomic: push failed for branch {branch!r}: "
            f"{push.stderr.strip()}",
            file=sys.stderr,
        )
        return False
    return True


def _find_repo_root(start: Path) -> Path | None:
    """Walk up from `start` until finding a `.git` directory. Returns
    the parent of that .git, or None if none found before filesystem root."""
    cur = start.resolve()
    if cur.is_file():
        cur = cur.parent
    while True:
        if (cur / ".git").exists():
            return cur
        parent = cur.parent
        if parent == cur:
            return None
        cur = parent
