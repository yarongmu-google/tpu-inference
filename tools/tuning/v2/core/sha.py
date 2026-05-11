# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Source-revision SHA helpers for raw-store partitioning.

`.kernel.raw/<kernel_sha>.jsonl` and `.service.raw/<service_sha>.jsonl`
partition raw measurements by the source revision that produced them.
This buys schema-evolution safety: when the kernel source or vLLM source
changes, new measurements go to a new SHA file and old measurements
remain valid for their own source revision.

Definitions (per docs/tuning_architecture.md Glossary):

- **kernel_sha**: 8-char git SHA of `tpu_inference` HEAD at tune time.
  Conservatively captures any change to the repo, not just to
  `kernels/ragged_paged_attention/v3/kernel.py` — the over-coverage is
  fine (just rotates `.raw` shards more often than strictly necessary).
- **service_sha**: pair `<tpu_inference_sha>-<vllm_sha>` joined with `-`.
  Either repo's change can affect vLLM scheduling or runner behavior;
  tracking both is the conservative choice. Today's `meta.txt` already
  captures both via `git rev-parse HEAD` in each repo's worktree.

Failure mode: if any git invocation fails (no repo, no git on PATH,
detached state with no SHA), the corresponding component returns the
literal string `"unknown"`. The raw store partitions just as well with
a `unknown` SHA — the rotation is just lossier when the operator can't
say WHICH version produced the row. Matches v1 behavior in
`tools/benchmark/run_benchmark.sh`.
"""

import subprocess
from pathlib import Path


def _git_rev_parse(repo_dir: Path, short: int = 8) -> str:
    """Return the short HEAD SHA of `repo_dir`, or 'unknown' on failure."""
    if not repo_dir.exists() or not (repo_dir / ".git").exists():
        return "unknown"
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse",
             f"--short={short}", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if result.returncode != 0:
            return "unknown"
        sha = result.stdout.strip()
        return sha if sha else "unknown"
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return "unknown"


def kernel_sha(repo_root: Path | None = None) -> str:
    """Return the 8-char git SHA of `tpu_inference` HEAD.

    Args:
      repo_root: Path to the tpu_inference repo. Defaults to the repo
                 root inferred from this file's location (three levels
                 up: this file lives at tools/tuning/v2/core/sha.py).

    Returns:
      8-char SHA, or 'unknown' if anything fails.
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[4]
    return _git_rev_parse(repo_root)


def service_sha(
    repo_root: Path | None = None,
    vllm_dir: Path | None = None,
) -> str:
    """Return '<tpu_inference_sha>-<vllm_sha>'.

    Args:
      repo_root: tpu_inference repo root. Defaults to inferred.
      vllm_dir: path to a vLLM source checkout. Defaults to
                ../vllm (relative to repo_root), matching
                tools/benchmark/run_benchmark.sh's VLLM_DIR default.

    Returns:
      String '<8>-<8>'. Either side may be 'unknown'.
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[4]
    if vllm_dir is None:
        vllm_dir = repo_root.parent / "vllm"
    inf = _git_rev_parse(repo_root)
    vllm = _git_rev_parse(vllm_dir)
    return f"{inf}-{vllm}"
