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
      "case_file":       "<path to .workload>",
      "sweep_name":      "<short label>",
      "sweep_axes":      {"VAR_A": [a1, a2], "VAR_B": [b1]},     // cartesian
      "coupled_axes":    [{"VAR_X": x1, "VAR_Y": y1}, ...],       // each entry = one combo
      "fixed":           {"VAR_C": "c"},                          // applied to every combo
      "timeout_seconds": 3600,                                    // optional, per-combo cap
      "kernel_registry": "<path to .kernel JSON>"                 // optional; enables auto-link
    }

When `kernel_registry` is set, enumerate_combos auto-links
RPA_D/M/P_BLOCK_SIZES into each combo by looking up the registry on
(case, page_size, K). Note K=0 (chunk-prefill OFF) is treated as "no
PREFILL pass" and produces NO RPA_P_BLOCK_SIZES injection — the
registry is not expected to carry a K=0 entry for the PREFILL flavor.
DECODE and MIXED entries (K=0 by convention in the registry) are
still injected for every combo regardless of LONG_PREFILL_TOKEN_THRESHOLD.

Keys must be **disjoint** across `sweep_axes`, `coupled_axes` (entry
keyset), and `fixed`. Coupled entries must all share the same key set.

`case_file` may be absolute or relative; relative paths are resolved
against the spec file's directory (so a spec is self-contained as long
as the case file is reachable via the encoded relative path).

Convention: any top-level key beginning with an underscore (e.g.
"_comment") is reserved for human-readable annotations and ignored by
the validator. Use this for inline JSON 'comments' since JSON has no
syntactic comment form.
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

from tools.benchmark._schema import (
    META_FILENAME, METRICS_FILENAME, THROUGHPUT_METRIC,
)


class SpecError(ValueError):
    """Raised when a sweep spec is malformed."""


# ----- Loading & validation ------------------------------------------------


def load_spec(path: str | os.PathLike) -> dict[str, Any]:
    """Read + validate a sweep spec from JSON.

    Raises SpecError on any problem: file-read failure (missing path,
    permissions), JSON parse failure, or shape/type validation failure.
    Caller has a single exception type to handle.
    """
    try:
        with open(path) as f:
            spec = json.load(f)
    except OSError as e:
        raise SpecError(f"could not read spec file {path}: {e}") from e
    except json.JSONDecodeError as e:
        raise SpecError(f"invalid JSON in {path}: {e}") from e
    if not isinstance(spec, dict):
        raise SpecError("top-level spec must be a JSON object")
    _validate_spec(spec)
    # Resolve a relative case_file against the spec file's directory so
    # specs are self-contained and the runner doesn't depend on the
    # caller's CWD. Absolute paths pass through unchanged.
    cf = Path(spec["case_file"])
    if not cf.is_absolute():
        spec["case_file"] = str((Path(path).resolve().parent / cf).resolve())
    # Pre-flight existence check: fail at load, not 1740 times in a row at
    # runtime. Without this guard, a bad case_file path would be reported
    # by every combo as 'case file not found' from inside run_benchmark.sh,
    # wasting hours before the user sees the spec was wrong.
    if not Path(spec["case_file"]).is_file():
        raise SpecError(
            f"case_file does not exist: {spec['case_file']}")

    if "kernel_registry" in spec:
        kr = Path(spec["kernel_registry"])
        if not kr.is_absolute():
            spec["kernel_registry"] = str((Path(path).resolve().parent / kr).resolve())
        if not Path(spec["kernel_registry"]).is_file():
            raise SpecError(f"kernel_registry does not exist: {spec['kernel_registry']}")
        try:
            with open(spec["kernel_registry"]) as f:
                spec["_loaded_kernel_registry"] = json.load(f)
        except Exception as e:
            raise SpecError(f"failed to load kernel_registry: {e}") from e


    return spec


def _normalize_optional(spec: dict[str, Any], key: str, default: Any) -> Any:
    """Return spec[key] unless missing or JSON null, else `default`.

    Crucially: a falsy-but-wrong-type value (e.g. sweep_axes=[]) falls
    through unchanged so the downstream isinstance check catches it.
    Earlier `spec.get(key) or default` masked these — `[]` is falsy so
    `or {}` substituted a dict, and the type error went undetected.
    """
    val = spec.get(key, default)
    if val is None:
        return default
    return val


def _validate_spec(spec: dict[str, Any]) -> None:
    required = {"case_file", "sweep_name"}
    missing = required - spec.keys()
    if missing:
        raise SpecError(f"missing required keys: {sorted(missing)}")
    for k in required:
        v = spec[k]
        if not isinstance(v, str) or not v:
            raise SpecError(f"{k} must be a non-empty string (got {v!r})")

    if "kernel_registry" in spec:
        v = spec["kernel_registry"]
        if not isinstance(v, str) or not v:
            raise SpecError(f"kernel_registry must be a non-empty string (got {v!r})")

    sweep_axes = _normalize_optional(spec, "sweep_axes", {})
    coupled_axes = _normalize_optional(spec, "coupled_axes", [])
    fixed = _normalize_optional(spec, "fixed", {})

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

    if "timeout_seconds" in spec and spec["timeout_seconds"] is not None:
        ts = spec["timeout_seconds"]
        # bool is a subclass of int; reject it explicitly (True/False
        # would silently coerce to 1/0).
        if isinstance(ts, bool) or not isinstance(ts, int) or ts <= 0:
            raise SpecError(
                f"timeout_seconds must be a positive int (got {ts!r})")

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


def apply_smoke_truncation_in_place(spec: dict[str, Any]) -> None:
    """Truncate sweep_axes/coupled_axes to the first value when
    SMOKE_TEST=1 in the environment. Mutates the spec dict in place.

    Kept separate from load_spec so the loader stays pure: a pure
    loader that conditionally rewrites the spec on a global env var
    surprises callers. This helper is what run_sweep + main() invoke
    explicitly when smoke mode is desired.
    """
    if os.environ.get("SMOKE_TEST") != "1":
        return
    print("SMOKE_TEST=1 detected: truncating service sweep space to 1 combo.",
          file=sys.stderr)
    if "sweep_axes" in spec:
        for k in spec["sweep_axes"]:
            spec["sweep_axes"][k] = spec["sweep_axes"][k][:1]
    if spec.get("coupled_axes"):
        spec["coupled_axes"] = spec["coupled_axes"][:1]


def _registry_lookup(
    registry: dict[str, Any],
    case_name: str,
    target_page_size: int,
    target_k: int,
) -> dict[str, int] | None:
    """Look up tunable_params in a .kernel registry by (case, page_size, K).

    Registry shape: registry["results"][case_name] = [
        {"tuning_key": {"page_size": <int>, ..., "chunk_prefill_size": <int>}},
         "tunable_params": {"bq_sz": ..., "bkv_sz": ..., ...}, ...},
        ...
    ]

    Returns the FIRST entry whose tuning_key matches (page_size +
    chunk_prefill_size for prefill / page_size for decode-mixed). Today
    the registry is pre-pruned to one winner per key by
    build_kernel_registry.py's accumulation logic, so "first match" is
    "the winner". If a future change starts storing multiple entries
    per key, callers should rank explicitly.
    """
    for entry in registry.get("results", {}).get(case_name, []):
        tk = entry.get("tuning_key", {})
        if tk.get("page_size") != target_page_size:
            continue
        # PREFILL and LOGICAL winners are both K-keyed (LOGICAL stores
        # K_kernel in chunk_prefill_size, same field). DECODE / MIXED
        # store chunk_prefill_size = 0 by convention.
        if case_name in ("prefill", "logical"):
            if tk.get("chunk_prefill_size") != target_k:
                continue
        return entry.get("tunable_params")
    return None


def _apply_auto_link(
    combo: dict[str, str],
    registry: dict[str, Any],
) -> None:
    """Populate RPA_D/M/P_BLOCK_SIZES on `combo` from `registry`.

    Mutates `combo` in place. Honors user overrides (existing
    RPA_*_BLOCK_SIZES values are kept). Raises SpecError if the
    registry is missing an entry the combo would need (so the failure
    is caught at spec-load, not after a kernel-default OOM at runtime).
    """
    page_size = int(combo.get("BLOCK_SIZE", "128"))
    k = int(combo.get("LONG_PREFILL_TOKEN_THRESHOLD", "0"))
    # Decoupled-K is signaled by RPA_KERNEL_K in the combo. When set,
    # the kernels P/L pass runs the LOGICAL branch instead of PREFILL,
    # so we route RPA_P_BLOCK_SIZES from the LOGICAL winner and also
    # auto-link RPA_MAX_NUM_SUBSEQS from its tunable_params. RPA_KERNEL_K
    # value is the K_kernel; chunk_prefill_size in the LOGICAL tuning_key
    # equals K_kernel (the tuner stores it in the same field as PREFILL).
    rpa_kernel_k = int(combo.get("RPA_KERNEL_K", "0"))
    decoupled_k_active = rpa_kernel_k > 0

    def fmt(p: dict[str, int]) -> str:
        return f"{p['bq_sz']},{p['bkv_sz']},{p['bq_csz']},{p['bkv_csz']}"

    if "RPA_D_BLOCK_SIZES" not in combo:
        d_params = _registry_lookup(registry, "decode", page_size, 0)
        if d_params is None:
            raise SpecError(
                f"kernel_registry has no DECODE entry at "
                f"page_size={page_size}. Either populate the registry "
                f"by re-running kernel tuning, or set RPA_D_BLOCK_SIZES "
                f"manually in the spec.")
        combo["RPA_D_BLOCK_SIZES"] = fmt(d_params)

    if "RPA_M_BLOCK_SIZES" not in combo:
        m_params = _registry_lookup(registry, "mixed", page_size, 0)
        if m_params is None:
            raise SpecError(
                f"kernel_registry has no MIXED entry at "
                f"page_size={page_size}. Either populate the registry "
                f"by re-running kernel tuning, or set RPA_M_BLOCK_SIZES "
                f"manually in the spec.")
        combo["RPA_M_BLOCK_SIZES"] = fmt(m_params)

    # PREFILL / LOGICAL block sizes are K-dependent. K=0 means
    # chunk-prefill OFF (the request_distribution stays [D, D, T] and
    # the kernel skips its PREFILL/LOGICAL pass entirely), so no
    # block-size lookup is needed.
    if decoupled_k_active and "RPA_P_BLOCK_SIZES" not in combo:
        # Decoupled-K: pull from the LOGICAL winner, keyed by
        # K=K_kernel (the chunk size the kernel actually runs).
        l_params = _registry_lookup(registry, "logical", page_size,
                                    rpa_kernel_k)
        if l_params is None:
            raise SpecError(
                f"kernel_registry has no LOGICAL entry at "
                f"(page_size={page_size}, K_kernel={rpa_kernel_k}). "
                f"Either re-run kernel tuning with "
                f"RPA_V3_TUNER_CASES=logical to populate this K, or "
                f"set RPA_P_BLOCK_SIZES manually for this combo.")
        combo["RPA_P_BLOCK_SIZES"] = fmt(l_params)
        if "RPA_MAX_NUM_SUBSEQS" not in combo:
            mns = l_params.get("max_num_subseqs")
            if mns is None:
                raise SpecError(
                    f"kernel_registry LOGICAL entry at "
                    f"(page_size={page_size}, K_kernel={rpa_kernel_k}) "
                    f"is missing max_num_subseqs in tunable_params. "
                    f"Re-run with the post-A.1 tuner, or set "
                    f"RPA_MAX_NUM_SUBSEQS manually for this combo.")
            combo["RPA_MAX_NUM_SUBSEQS"] = str(mns)
        # LPTT is a derived deployment value, not a sweep dimension.
        # Server enforces LPTT == mnss * kernel_K at runner init
        # (tpu_runner.py). Set it from the resolved mnss when the user
        # left it unset; honor an explicit user pin and let the server
        # reject it if it violates the invariant.
        if "LONG_PREFILL_TOKEN_THRESHOLD" not in combo:
            combo["LONG_PREFILL_TOKEN_THRESHOLD"] = str(
                int(combo["RPA_MAX_NUM_SUBSEQS"]) * rpa_kernel_k)
    elif k > 0 and "RPA_P_BLOCK_SIZES" not in combo:
        p_params = _registry_lookup(registry, "prefill", page_size, k)
        if p_params is None:
            raise SpecError(
                f"kernel_registry has no PREFILL entry at "
                f"(page_size={page_size}, K={k}). Either re-run kernel "
                f"tuning to populate this K, or set RPA_P_BLOCK_SIZES "
                f"manually for this combo.")
        combo["RPA_P_BLOCK_SIZES"] = fmt(p_params)


def enumerate_combos(spec: dict[str, Any]) -> list[dict[str, str]]:
    """Yield the cartesian × coupled product of `spec`, merged with `fixed`.

    Order is deterministic:
      - sweep_axes iterated in insertion order (the dict's key order)
      - cartesian product in the standard itertools.product order
      - coupled_axes entries in their list order

    Per-combo merge precedence: cartesian < coupled < fixed (later wins).
    With the disjoint-keys rule that _validate_spec enforces, this
    precedence isn't observable in well-formed specs — but the order
    is documented so future-you knows the intended behavior if the
    validator ever loosens.

    All values are stringified (vllm flags want strings).
    """
    # By this point _validate_spec has run; values are well-typed.
    # Use _normalize_optional to handle missing/None uniformly.
    sweep_axes = _normalize_optional(spec, "sweep_axes", {})
    coupled_axes = _normalize_optional(spec, "coupled_axes", [])
    fixed = _normalize_optional(spec, "fixed", {})

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
            combo = {k: str(v) for k, v in env.items()}

            if "_loaded_kernel_registry" in spec:
                _apply_auto_link(combo, spec["_loaded_kernel_registry"])

            combos.append(combo)
    return combos


# ----- Naming & paths ------------------------------------------------------


def combo_id(env: dict[str, str]) -> str:
    """Deterministic 12-hex-char ID derived from canonical-JSON of `env`.

    Stable across spec reorderings: the same set of (key, value) pairs
    always hashes to the same ID. That gives us free resumability — if a
    combo's metrics.txt already exists, we skip it on re-run.

    12 hex = 48 bits. Birthday-collision probability stays < 1% up to
    roughly 2 million combos and < 50% up to ~17 million; safe for any
    realistic sweep size we'd run.
    """
    canonical = json.dumps(env, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]


def case_name_from_path(case_file: str | os.PathLike) -> str:
    """Strip dir + a single trailing .workload suffix.

    Conservative on multi-suffix names: ``foo.workload.bak`` -> ``foo.workload.bak``
    (no .workload suffix at the very end). ``foo.workload.workload`` -> ``foo.workload`` (only
    the last suffix is stripped). Plain non-.workload names pass through.
    """
    base = os.path.basename(str(case_file))
    if base.endswith(".workload"):
        return base[:-len(".workload")]
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


def _read_meta_value(rdir: Path, key: str) -> str | None:
    """Read a single key=value line out of meta.txt. Returns the value
    string (after `=`) or None if the file is missing or the key is
    absent. No type coercion."""
    p = Path(rdir, META_FILENAME)
    if not p.is_file():
        return None
    try:
        text = p.read_text()
    except OSError:
        return None
    prefix = f"{key}="
    for line in text.splitlines():
        if line.startswith(prefix):
            return line.split("=", 1)[1].strip()
    return None


def _current_stack_commits() -> tuple[str | None, str | None]:
    """Return (tpu_inference HEAD, vllm HEAD) full SHAs, or None each
    if unavailable. Used by is_completed to invalidate cached combos
    after a code change on either side of the served stack.

    Cached at module load — both repos are static across one sweep
    invocation.
    """
    return (_CURRENT_TPU_COMMIT, _CURRENT_VLLM_COMMIT)


def _git_head(repo_path: str) -> str | None:
    try:
        import subprocess
        r = subprocess.run(
            ["git", "-C", repo_path, "rev-parse", "HEAD"],
            capture_output=True, check=True, text=True, timeout=5,
        )
        return r.stdout.strip()
    except (subprocess.SubprocessError, OSError, FileNotFoundError):
        return None


_CURRENT_TPU_COMMIT = _git_head(".")
_CURRENT_VLLM_COMMIT = _git_head(os.environ.get("VLLM_DIR", "../vllm"))


def is_completed(rdir: str | os.PathLike) -> bool:
    """A combo is 'done' iff metrics.txt has a non-empty RequestThroughput
    AND meta.txt records the current (tpu_inference, vllm) HEAD commits.

    Two layers:

    1. File-existence alone is NOT enough: parse_bench_log.format_metrics
       always emits the full key list with blank values for missing data,
       so a bench that died after parse_bench_log started writing leaves
       a non-zero-byte metrics.txt with `RequestThroughput=` (empty).
       Without this guard, a re-run would skip-resume the dead combo
       and the sweep would silently produce blank rows that look 'done'.

    2. Even a complete metrics.txt is stale if the served stack
       (tpu_inference + vllm) has changed since the run. The kernel /
       runner / scheduler may have moved; the cached number no longer
       reflects what would happen NOW. Both meta.git_commit and
       meta.vllm_commit must match the current HEADs of the two repos
       — otherwise return False so the combo re-runs.

       Conservative: ANY commit mismatch invalidates. If you want softer
       invalidation (e.g., only when perf-affecting code changed), set
       SKIP_COMMIT_CACHE_CHECK=1 in the env to bypass this layer.
    """
    rdir = Path(rdir)
    p = rdir / METRICS_FILENAME
    if not p.is_file():
        return False
    try:
        text = p.read_text()
    except OSError:
        return False
    prefix = f"{THROUGHPUT_METRIC}="
    has_throughput = any(
        line.startswith(prefix) and line.split("=", 1)[1].strip()
        for line in text.splitlines())
    if not has_throughput:
        return False

    if os.environ.get("SKIP_COMMIT_CACHE_CHECK") == "1":
        return True

    cur_tpu, cur_vllm = _current_stack_commits()
    cached_tpu = _read_meta_value(rdir, "git_commit")
    cached_vllm = _read_meta_value(rdir, "vllm_commit")

    # If we cant determine the current commits (e.g., running outside
    # a checkout), fall back to the legacy "throughput exists =>
    # complete" check rather than refusing to skip anything.
    if cur_tpu is None and cur_vllm is None:
        return True

    if cur_tpu is not None and cached_tpu != cur_tpu:
        return False
    if cur_vllm is not None and cached_vllm != cur_vllm:
        return False
    return True


# ----- Runner --------------------------------------------------------------


DEFAULT_SCRIPT = "tools/benchmark/run_benchmark.sh"

# Per-combo wall-clock cap. Generous (one combo on Llama 3 8B takes a few
# minutes; the cap is meant to stop a hung process from blocking a sweep,
# not to enforce a perf goal). Override per-spec via "timeout_seconds".
DEFAULT_TIMEOUT_SECONDS = 3600


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

    Skip if its result dir already has a non-empty metrics.txt
    (resumability — see is_completed). Returns a RunResult describing
    what happened. Never raises on subprocess failure (only on bugs in
    *our* code) — known failure modes are reported as RunStatus.FAILED
    so a sweep can continue.

    Failure-mode reference (what populates the RunResult on FAILED):
      - subprocess timed out   → error='combo timed out after Ns',
                                 return_code=None
      - subprocess raised      → error=repr(err), return_code=None
        (caught: OSError, subprocess.SubprocessError; other exceptions
        propagate so test/code bugs surface)
      - subprocess exit != 0   → error='subprocess exited with code N',
                                 return_code=N
      - exit 0, metrics blank  → error='... missing or has no
                                 RequestThroughput', return_code=0

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
    # _normalize_optional (vs `or default`) so an explicit `0` would
    # surface — _validate_spec rejects 0 anyway, but the read here is
    # consistent with the rest of the spec-loading path.
    timeout = int(_normalize_optional(
        spec, "timeout_seconds", DEFAULT_TIMEOUT_SECONDS))
    started = timer()
    try:
        proc = run_subprocess(cmd, env=env, check=False, timeout=timeout)
    except subprocess.TimeoutExpired:
        return RunResult(
            status=RunStatus.FAILED,
            combo_id=cid,
            result_dir=rdir,
            duration_seconds=timer() - started,
            error=f"combo timed out after {timeout}s",
        )
    except (OSError, subprocess.SubprocessError) as err:
        # Narrow on purpose: catch only subprocess-launch failures (script
        # not executable, ENOENT, etc.). Bugs in our own code (KeyError,
        # AttributeError) should crash loudly so tests / sweeps surface
        # them instead of producing a misleading FAILED row.
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
            error=f"subprocess exited with code {rc}",
        )

    # Success only counts if metrics.txt has actual content. is_completed
    # rejects both "file missing" and "file present but RequestThroughput
    # is blank" (a partial-write artifact of a dying bench).
    if not is_completed(rdir):
        return RunResult(
            status=RunStatus.FAILED,
            combo_id=cid,
            result_dir=rdir,
            return_code=rc,
            duration_seconds=duration,
            error=(f"run_benchmark.sh exited 0 but {METRICS_FILENAME} is "
                   f"missing or has no {THROUGHPUT_METRIC}"),
        )

    return RunResult(
        status=RunStatus.COMPLETED_FRESH,
        combo_id=cid,
        result_dir=rdir,
        return_code=rc,
        duration_seconds=duration,
    )


# Statuses where re-running the combo wastes time. FAILED stays OUT
# so a transient bench failure retries on the next sweep — same
# permanence policy as the kernel-tune resume in kernel_tuner_base.
_SERVICE_SKIP_STATUSES = frozenset({"completed_fresh", "skipped_resumed"})


def _load_service_raw_skip_set(raw_path: Path) -> set[str]:
    """Build a combo_id skip-set from a prior sweep's service.raw.jsonl.

    Symmetric to `KernelTunerBase._load_raw_jsonl_skip_set` — same JSONL
    discipline, tolerates a truncated trailing line from a Ctrl-C mid-
    write, logs a one-line summary so the operator can sanity-check.

    Returns the empty set when the file is absent (fresh sweep) or
    when nothing in it has a skip-worthy status.

    FAILED rows are intentionally NOT in the skip-set: a sweep that
    failed N combos and is re-run after a fix should re-attempt those
    N. Operators who want to retain failures-as-skipped can manually
    flip the status in the JSONL.
    """
    if not raw_path.exists():
        return set()
    skip: set[str] = set()
    n_rows = n_skip = n_malformed = 0
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                n_malformed += 1
                continue
            if (row.get("status") in _SERVICE_SKIP_STATUSES
                    and row.get("combo_id")):
                skip.add(row["combo_id"])
                n_skip += 1
    print(
        f"[Sweep] Resume from {raw_path}: {n_rows} rows "
        f"({n_skip} skip, {n_rows - n_skip - n_malformed} retry, "
        f"{n_malformed} malformed)",
        file=sys.stderr, flush=True,
    )
    return skip


def _append_service_raw_jsonl(
    raw_path: Path,
    combo: dict[str, Any],
    result: RunResult,
) -> None:
    """Append one durable JSONL row per combo to <sweep_dir>/service.raw.jsonl.

    Symmetric counterpart to the kernel-tune `kernel.raw.jsonl` written
    by `kernel_tuner_base.measure_latency`. Each row is flushed + fsync'd
    immediately, so a Ctrl-C between combos loses nothing: every combo
    that COMPLETED its run_one call before the signal has its outcome
    on disk.

    The per-combo `metrics.txt` (under each combo's own result_dir)
    remains the bench's primary artifact — `is_completed()` and
    `build_service_registry` both consume it. The service.raw.jsonl
    is a SEPARATE aggregated log: one line per combo across the whole
    sweep, durable, easy to grep, the source of truth for the
    JSONL-driven sweep resume that lands in the next commit.

    Row schema mirrors kernel.raw.jsonl's discipline: flat
    dict-of-primitives, default=str on json.dumps so any non-JSON
    field type can't crash the write.
    """
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    # Pull the key metrics from the combo's metrics.txt if it landed
    # one — FAILED combos may not have it; SKIPPED_RESUMED combos
    # always do (that's what was checked to mark them resumed).
    metrics: dict[str, str] = {}
    metrics_path = Path(result.result_dir) / METRICS_FILENAME
    if metrics_path.exists():
        try:
            for line in metrics_path.read_text(encoding="utf-8").splitlines():
                if "=" in line:
                    k, _, v = line.partition("=")
                    metrics[k.strip()] = v.strip()
        except OSError:
            # Race / permissions / corrupt — tolerate; the row still
            # lands, just without metrics. Operator can re-inspect
            # the per-combo dir directly.
            pass
    row = {
        "combo":            combo,
        "combo_id":         result.combo_id,
        "status":           result.status.value,
        "return_code":      result.return_code,
        "duration_seconds": result.duration_seconds,
        "error":            result.error,
        "result_dir":       str(result.result_dir),
        "metrics":          metrics,
        "timestamp_sec":    int(time.time()),
    }
    with open(raw_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


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
    apply_smoke_truncation_in_place(spec)
    combos = enumerate_combos(spec)
    # Pre-compute the durable per-sweep JSONL log path. One file at
    # sweep_dir root, append-only, fsync'd per row. See
    # `_append_service_raw_jsonl` for the row schema.
    case_name = case_name_from_path(spec["case_file"])
    sweep_name = spec["sweep_name"]
    s_dir = sweep_dir(base_dir, case_name, sweep_name)
    raw_path = s_dir / "service.raw.jsonl"
    # Combo-id resume skip-set. Loaded once; checked per-combo before
    # the run_one dispatch. JSONL-based, authoritative — trusts the
    # prior log over a per-combo metrics.txt stat. Operators can
    # `rm` the JSONL (or its rows) to force a fresh sweep.
    skip_set = _load_service_raw_skip_set(raw_path)
    results: list[RunResult] = []
    for i, combo in enumerate(combos):
        cid = combo_id(combo)
        if cid in skip_set:
            # Synthesize a SKIPPED_RESUMED result without invoking
            # run_one — same shape as run_one returns when its
            # internal is_completed(rdir) trips, so downstream code
            # (build_service_registry, on_result callback, etc.) sees
            # an identical-looking outcome.
            result = RunResult(
                status=RunStatus.SKIPPED_RESUMED,
                combo_id=cid,
                result_dir=result_dir(
                    base_dir, case_name, sweep_name, cid,
                ),
            )
        else:
            result = run_one_fn(
                spec, combo, base_dir=base_dir,
                script_path=script_path,
            )
        results.append(result)
        _append_service_raw_jsonl(raw_path, combo, result)
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

    Returns True on success. Returns False if any subprocess step fails,
    if `git diff --cached --quiet` shows nothing to commit (treated as a
    non-fatal no-op), or if pushing was requested but the auto-detected
    target branch is the literal 'HEAD' (detached checkout — refuse to
    guess).

    Not reentrant against concurrent index manipulation: the gap between
    `git diff --cached --quiet` and `git commit` lets another process
    change what's staged. Safe in single-script sweeps; risky if the
    user is `git add`-ing in the same repo while a sweep runs.

    `run_subprocess` is injected for tests.
    """
    if not paths:
        # Defensive: empty list is a programming bug (caller should have
        # filtered). Print a specific reason so it doesn't blend into
        # the auto-commit callback's generic WARN.
        print("git_commit_paths: no paths to commit", flush=True)
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
            print(f"git_commit_paths: nothing staged for "
                  f"{[str(p) for p in paths]}", flush=True)
            return False
        run_subprocess(["git", "commit", "-m", message], check=True)
        if push:
            if branch is None:
                target_branch = _current_branch(run_subprocess)
                if target_branch == "HEAD":
                    # `git rev-parse --abbrev-ref HEAD` returns the literal
                    # "HEAD" on a detached checkout. Pushing to that has
                    # surprising / config-dependent semantics (push.default
                    # decides), so refuse to guess. Caller can pass an
                    # explicit branch= if they really mean it.
                    print("git_commit_paths: detached HEAD; commit landed "
                          "but refusing to push (pass branch=… to override)",
                          flush=True)
                    return False
            else:
                target_branch = branch
            run_subprocess(["git", "push", remote, target_branch],
                           check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"git_commit_paths: subprocess failed: {e}", flush=True)
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
    p.add_argument("--runlog", default=None,
                   help=("Path to a console runlog (typically the file you "
                         "are tee-ing this command's stdout/stderr into). "
                         "When set, every auto-commit also stages this file "
                         "so a remote reader can diagnose hangs/crashes "
                         "alongside the metrics."))
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
    runlog: str | os.PathLike | None = None,
) -> Callable[[RunResult, int, int], None]:
    """Build an on_result callback that commits every N combos and at end.

    Stages the per-combo metrics.txt + meta.txt files explicitly rather
    than the sweep parent directory. The parent-dir form would also work
    when the sweep lives under tmp/ (the .gitignore patterns for
    vllm.log, bench.log, sonnet_4x.txt, profile/ are scoped there), but
    breaks under any other --base-dir: `git add results/sweep_dir/`
    would also stage the multi-GB log files. Enumerating the safe
    files explicitly is robust regardless of where the sweep lands.

    Each fire re-stages all currently-completed combos, not just the
    one that just finished — so a previous combo that was somehow
    missed (e.g., a partial-failure window) still gets picked up.
    `git diff --cached --quiet` in git_commit_paths short-circuits to
    a no-op when nothing's changed since the last commit.

    If `runlog` is given, that path is staged alongside the metrics on
    each commit. The file is expected to be the tee'd console output
    of this run; its presence isn't required at every commit (early
    fires before tee creates the file are tolerated — the path is just
    skipped). This is what gets a remote reader (or a future you) the
    progress / error context that metrics.txt doesn't carry.
    """

    runlog_path = Path(runlog) if runlog is not None else None

    def cb(result: RunResult, idx: int, total: int) -> None:
        _print_progress(result, idx, total)
        n = idx + 1
        is_last = (n == total)
        if (n % every == 0) or is_last:
            # Local var renamed from `sweep_dir` to `combo_parent_dir`
            # because `sweep_dir` shadows the module-level function of
            # the same name, which is a footgun if a future caller in
            # this closure ever needs the function.
            combo_parent_dir = result.result_dir.parent
            paths: list = sorted(combo_parent_dir.glob(f"*/{METRICS_FILENAME}")) \
                  + sorted(combo_parent_dir.glob(f"*/{META_FILENAME}"))
            if runlog_path is not None and runlog_path.is_file():
                paths.append(runlog_path)
            ok = git_fn(paths, message=(
                f"[Bench] Auto-commit sweep progress {n}/{total}"
            ), push=push)
            if not ok:
                # git_commit_paths returns False on subprocess error,
                # nothing-staged, or detached HEAD. Surface so the user
                # doesn't return tomorrow expecting results in the remote
                # and find an unsynced local.
                print(f"WARN: auto-commit/push failed at {n}/{total}",
                      flush=True)

    return cb


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.auto_commit_every < 0:
        parser.error("--auto-commit-every must be >= 0")
    if args.no_push and args.auto_commit_every == 0:
        parser.error("--no-push requires --auto-commit-every > 0")

    if args.auto_commit_every > 0:
        on_result = _make_auto_commit_callback(
            args.auto_commit_every, push=not args.no_push,
            runlog=args.runlog)
    elif args.runlog is not None:
        # --runlog without --auto-commit-every is meaningless: the runlog
        # only gets staged on auto-commit fires, so without auto-commit
        # there's nothing to stage. Surface as a CLI error rather than
        # silently ignoring the flag.
        parser.error("--runlog requires --auto-commit-every > 0")
    else:
        on_result = _print_progress

    results = run_sweep(args.spec, base_dir=args.base_dir,
                        script_path=args.script, on_result=on_result)
    n_fail = sum(1 for r in results if r.status is RunStatus.FAILED)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
