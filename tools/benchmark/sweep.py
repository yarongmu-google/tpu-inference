# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Sweep-spec loading + combo enumeration for vLLM-bench sweeps.

This module is the *pure* part of the sweep driver: parse a JSON spec
into a list of (env-var dict) combos, plus deterministic ID / path
helpers. Subprocess execution and CLI live in a follow-up commit so this
file stays unit-testable end-to-end without touching disk or invoking
vLLM.

Spec shape:
    {
      "case_file":   "<path to .env>",
      "sweep_name":  "<short label>",
      "sweep_axes":  {"VAR_A": [a1, a2], "VAR_B": [b1]},      // cartesian
      "coupled_axes":[{"VAR_X": x1, "VAR_Y": y1}, ...],        // each entry = one combo
      "fixed":       {"VAR_C": "c"}                            // applied to every combo
    }

Keys must be **disjoint** across `sweep_axes`, `coupled_axes` (entry
keyset), and `fixed`. Coupled entries must all share the same key set.
"""

import argparse
import dataclasses
import enum
import hashlib
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable


class SpecError(ValueError):
    """Raised when a sweep spec is malformed."""


# ----- Loading & validation ------------------------------------------------


def load_spec(path: str | os.PathLike) -> dict[str, Any]:
    """Read + validate a sweep spec from JSON. Raises SpecError on bad shape."""
    with open(path) as f:
        spec = json.load(f)
    if not isinstance(spec, dict):
        raise SpecError("top-level spec must be a JSON object")
    _validate_spec(spec)
    return spec


def _validate_spec(spec: dict[str, Any]) -> None:
    required = {"case_file", "sweep_name"}
    missing = required - spec.keys()
    if missing:
        raise SpecError(f"missing required keys: {sorted(missing)}")

    sweep_axes = spec.get("sweep_axes") or {}
    coupled_axes = spec.get("coupled_axes") or []
    fixed = spec.get("fixed") or {}

    if not isinstance(sweep_axes, dict):
        raise SpecError("sweep_axes must be a JSON object")
    if not isinstance(coupled_axes, list):
        raise SpecError("coupled_axes must be a JSON array")
    if not isinstance(fixed, dict):
        raise SpecError("fixed must be a JSON object")

    for k, v in sweep_axes.items():
        if not isinstance(v, list) or not v:
            raise SpecError(
                f"sweep_axes['{k}'] must be a non-empty list (got {v!r})")

    for i, entry in enumerate(coupled_axes):
        if not isinstance(entry, dict):
            raise SpecError(
                f"coupled_axes[{i}] must be a JSON object (got {type(entry).__name__})")

    if coupled_axes:
        first_keys = set(coupled_axes[0].keys())
        for i, entry in enumerate(coupled_axes):
            if set(entry.keys()) != first_keys:
                raise SpecError(
                    f"coupled_axes[{i}] keys {sorted(entry.keys())} differ "
                    f"from coupled_axes[0] keys {sorted(first_keys)}")

    sweep_keys = set(sweep_axes.keys())
    coupled_keys = set(coupled_axes[0].keys()) if coupled_axes else set()
    fixed_keys = set(fixed.keys())

    overlap = {
        "sweep_axes/coupled_axes": sweep_keys & coupled_keys,
        "sweep_axes/fixed":         sweep_keys & fixed_keys,
        "coupled_axes/fixed":       coupled_keys & fixed_keys,
    }
    for label, ks in overlap.items():
        if ks:
            raise SpecError(
                f"keys must be disjoint; {label} share: {sorted(ks)}")


# ----- Enumeration ---------------------------------------------------------


def enumerate_combos(spec: dict[str, Any]) -> list[dict[str, str]]:
    """Yield the cartesian × coupled product of `spec`, merged with `fixed`.

    Order is deterministic:
      - sweep_axes iterated in insertion order (the dict's key order)
      - cartesian product in the standard itertools.product order
      - coupled_axes entries in their list order

    All values are stringified (vllm flags want strings).
    """
    sweep_axes = spec.get("sweep_axes") or {}
    coupled_axes = spec.get("coupled_axes") or []
    fixed = spec.get("fixed") or {}

    sweep_keys = list(sweep_axes.keys())
    sweep_values = [sweep_axes[k] for k in sweep_keys]
    cartesian: list[tuple] = (
        list(itertools.product(*sweep_values)) if sweep_values else [()])
    coupled: list[dict] = list(coupled_axes) if coupled_axes else [{}]

    combos: list[dict[str, str]] = []
    for cart_tuple in cartesian:
        cart_dict = dict(zip(sweep_keys, cart_tuple))
        for cpl in coupled:
            env: dict[str, Any] = {}
            env.update(cart_dict)
            env.update(cpl)
            env.update(fixed)
            combos.append({k: str(v) for k, v in env.items()})
    return combos


# ----- Naming & paths ------------------------------------------------------


def combo_id(env: dict[str, str]) -> str:
    """Deterministic 12-hex-char ID derived from canonical-JSON of `env`.

    Stable across spec reorderings: the same set of (key, value) pairs
    always hashes to the same ID. That gives us free resumability — if a
    combo's metrics.txt already exists, we skip it on re-run.
    """
    canonical = json.dumps(env, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]


def case_name_from_path(case_file: str | os.PathLike) -> str:
    """Strip dir + .env suffix. e.g. cases/foo.env -> foo."""
    base = os.path.basename(str(case_file))
    if base.endswith(".env"):
        return base[:-len(".env")]
    return base


def result_dir(
    base_dir: str | os.PathLike,
    case_name: str,
    sweep_name: str,
    cid: str,
) -> Path:
    """Path to a single combo's result dir."""
    return Path(base_dir) / f"bench_{case_name}_{sweep_name}" / cid


def sweep_dir(
    base_dir: str | os.PathLike,
    case_name: str,
    sweep_name: str,
) -> Path:
    """Parent directory holding all combos for one sweep."""
    return Path(base_dir) / f"bench_{case_name}_{sweep_name}"


def is_completed(rdir: str | os.PathLike) -> bool:
    """A combo is 'done' if metrics.txt exists in its result dir."""
    return Path(rdir, "metrics.txt").is_file()


# ----- Runner --------------------------------------------------------------


DEFAULT_SCRIPT = "tools/benchmark/run_benchmark.sh"


class RunStatus(enum.Enum):
    SKIPPED_RESUMED = "skipped_resumed"   # metrics.txt already existed
    COMPLETED_FRESH = "completed_fresh"   # ran successfully this time
    FAILED = "failed"                     # subprocess returned non-zero or exception


@dataclasses.dataclass
class RunResult:
    status: RunStatus
    combo_id: str
    result_dir: Path
    return_code: int | None = None
    duration_seconds: float | None = None
    error: str | None = None


def run_one(
    spec: dict[str, Any],
    combo: dict[str, str],
    *,
    base_dir: str | os.PathLike = "tmp",
    script_path: str | os.PathLike = DEFAULT_SCRIPT,
    run_subprocess: Callable = subprocess.run,
    timer: Callable[[], float] = time.monotonic,
    environ: dict[str, str] | None = None,
) -> RunResult:
    """Execute one combo by invoking run_benchmark.sh with combo env vars.

    Skip if its result dir already has a metrics.txt (resumability). Returns
    a RunResult describing what happened. Never raises on subprocess failure
    — failures are reported as `RunStatus.FAILED` so a sweep can continue.

    `run_subprocess`, `timer`, and `environ` are injected for testability.
    """
    case_file = spec["case_file"]
    case_name = case_name_from_path(case_file)
    sweep_name = spec["sweep_name"]
    cid = combo_id(combo)
    rdir = result_dir(base_dir, case_name, sweep_name, cid)

    if is_completed(rdir):
        return RunResult(
            status=RunStatus.SKIPPED_RESUMED,
            combo_id=cid,
            result_dir=rdir,
        )

    rdir.mkdir(parents=True, exist_ok=True)
    base_environ = dict(os.environ if environ is None else environ)
    env = {**base_environ, **combo, "RESULT_DIR": str(rdir)}

    cmd = [str(script_path), str(case_file)]
    started = timer()
    try:
        proc = run_subprocess(cmd, env=env, check=False)
    except Exception as err:  # subprocess setup failure (rare)
        return RunResult(
            status=RunStatus.FAILED,
            combo_id=cid,
            result_dir=rdir,
            duration_seconds=timer() - started,
            error=repr(err),
        )
    duration = timer() - started

    rc = getattr(proc, "returncode", 1)
    if rc != 0:
        return RunResult(
            status=RunStatus.FAILED,
            combo_id=cid,
            result_dir=rdir,
            return_code=rc,
            duration_seconds=duration,
        )

    # Success only counts if metrics.txt landed.
    if not is_completed(rdir):
        return RunResult(
            status=RunStatus.FAILED,
            combo_id=cid,
            result_dir=rdir,
            return_code=rc,
            duration_seconds=duration,
            error="run_benchmark.sh exited 0 but metrics.txt missing",
        )

    return RunResult(
        status=RunStatus.COMPLETED_FRESH,
        combo_id=cid,
        result_dir=rdir,
        return_code=rc,
        duration_seconds=duration,
    )


def run_sweep(
    spec_path: str | os.PathLike,
    *,
    base_dir: str | os.PathLike = "tmp",
    script_path: str | os.PathLike = DEFAULT_SCRIPT,
    run_one_fn: Callable[..., RunResult] = run_one,
    on_result: Callable[[RunResult, int, int], None] | None = None,
) -> list[RunResult]:
    """Iterate every combo in `spec_path`, invoking run_one for each.

    `on_result(result, idx, total)` is called after each combo (post-skip
    or post-run); use it to log progress or trigger an auto-commit. Failures
    do not abort the sweep — the next combo runs.
    """
    spec = load_spec(spec_path)
    combos = enumerate_combos(spec)
    results: list[RunResult] = []
    for i, combo in enumerate(combos):
        result = run_one_fn(spec, combo, base_dir=base_dir,
                            script_path=script_path)
        results.append(result)
        if on_result is not None:
            on_result(result, i, len(combos))
    return results


# ----- Git helpers ---------------------------------------------------------


def git_commit_paths(
    paths: list[str | os.PathLike],
    message: str,
    *,
    push: bool = True,
    remote: str = "origin",
    branch: str | None = None,
    run_subprocess: Callable = subprocess.run,
) -> bool:
    """Stage `paths`, commit with `message`, optionally push.

    Returns True on success. Returns False if any subprocess step fails or
    if `git diff --cached --quiet` shows nothing to commit (we treat
    "nothing to commit" as a non-fatal no-op).

    `run_subprocess` is injected for tests.
    """
    if not paths:
        return False
    try:
        run_subprocess(["git", "add", "--", *(str(p) for p in paths)],
                       check=True)
        # `--quiet` returns 1 when there ARE staged changes (i.e., something
        # to commit), 0 when there are NONE. So returncode==0 means nothing
        # staged: nothing to commit -> non-fatal no-op.
        diff = run_subprocess(["git", "diff", "--cached", "--quiet"],
                              check=False)
        if diff.returncode == 0:
            return False
        run_subprocess(["git", "commit", "-m", message], check=True)
        if push:
            target_branch = branch or _current_branch(run_subprocess)
            run_subprocess(["git", "push", remote, target_branch],
                           check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def _current_branch(run_subprocess: Callable) -> str:
    """Return the short name of the currently-checked-out git branch."""
    proc = run_subprocess(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        check=True, capture_output=True, text=True)
    return proc.stdout.strip()


# ----- CLI -----------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Drive a vLLM-bench sweep over a JSON spec.")
    p.add_argument("spec", type=str,
                   help="Path to a sweep spec JSON file.")
    p.add_argument("--base-dir", default="tmp",
                   help="Root for result dirs (default: tmp).")
    p.add_argument("--script", default=DEFAULT_SCRIPT,
                   help=f"run_benchmark.sh location (default: {DEFAULT_SCRIPT}).")
    p.add_argument("--auto-commit-every", type=int, default=0,
                   help=("Commit + push after every N completed combos. "
                         "0 = no auto-commit (default). Pushes to "
                         "`origin <current-branch>`."))
    p.add_argument("--no-push", action="store_true",
                   help="With --auto-commit-every > 0, commit but don't push.")
    return p


def _print_progress(result: RunResult, idx: int, total: int) -> None:
    n = idx + 1
    dur = (f"{result.duration_seconds:.1f}s"
           if result.duration_seconds is not None else "—")
    line = (f"[{n}/{total}] {result.combo_id} {result.status.value} "
            f"({dur})  {result.result_dir}")
    if result.error:
        line += f"  err={result.error}"
    print(line, flush=True)


def _make_auto_commit_callback(
    every: int,
    push: bool,
    git_fn: Callable = git_commit_paths,
) -> Callable[[RunResult, int, int], None]:
    """Build an on_result callback that commits every N combos and at end."""

    def cb(result: RunResult, idx: int, total: int) -> None:
        _print_progress(result, idx, total)
        n = idx + 1
        is_last = (n == total)
        if (n % every == 0) or is_last:
            git_fn([result.result_dir.parent], message=(
                f"[Bench] Auto-commit sweep progress {n}/{total}"
            ), push=push)

    return cb


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.auto_commit_every > 0:
        on_result = _make_auto_commit_callback(
            args.auto_commit_every, push=not args.no_push)
    else:
        on_result = _print_progress

    results = run_sweep(args.spec, base_dir=args.base_dir,
                        script_path=args.script, on_result=on_result)
    n_fail = sum(1 for r in results if r.status is RunStatus.FAILED)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
