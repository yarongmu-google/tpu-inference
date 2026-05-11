# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""`.workload` schema validator.

CLI entry: `python3 -m tools.tuning.v2.cli.validate <workload>`.

Checks a `.workload` file for the minimum-required vars and catches
common typos at spec-load time (item 6 of the §12 migration plan):

  - MODEL                       — required (string, non-empty)
  - TENSOR_PARALLEL_SIZE        — required int >= 1
  - NUM_Q_HEADS, NUM_KV_HEADS,
    HEAD_DIM                    — required int >= 1
  - MAX_MODEL_LEN               — required int >= 1
  - MAX_NUM_SEQS                — required int >= 1
  - INPUT_LEN                   — required int >= 1
  - OUTPUT_LEN                  — required int >= 1

Plus the invariant: `MAX_MODEL_LEN >= INPUT_LEN + OUTPUT_LEN`. Below
that, vllm bench will refuse the request at runtime; the validator
catches it at spec-load time so operators don't burn a sweep slot
on a guaranteed-fail config.

`MAX_NUM_BATCHED_TOKENS` is NOT required at the workload layer (it's
a service-sweep dimension per §2 of `docs/tuning_architecture.md`);
the validator flags it as a warning, not an error, since some legacy
workloads still hand-code it.

Returns a list of `(severity, message)` tuples. Severity is "error"
(blocks valid use) or "warning" (operator should look but won't
necessarily break things). The CLI exits 0 iff no errors.
"""

import sys
from pathlib import Path
from typing import Any


# Required vars + their type constraints. Map var -> (is_int, min_value).
REQUIRED: dict[str, tuple[bool, int]] = {
    "MODEL":                 (False, 0),
    "TENSOR_PARALLEL_SIZE":  (True, 1),
    "NUM_Q_HEADS":           (True, 1),
    "NUM_KV_HEADS":          (True, 1),
    "HEAD_DIM":              (True, 1),
    "MAX_MODEL_LEN":         (True, 1),
    # Category-2 measurement vars (architecture-doc §2 line 46-47):
    # "DATASET, INPUT_LEN, OUTPUT_LEN, NUM_PROMPTS, REQUEST_RATE
    # — load shape used by bench driver, bench duration / arrival
    # pattern." All five required so a typo (DATSET=sonnet) errors
    # at validation time instead of silently letting the bench fall
    # back to defaults and characterizing the wrong workload.
    "INPUT_LEN":             (True, 1),
    "OUTPUT_LEN":            (True, 1),
    "DATASET":               (False, 0),
    "NUM_PROMPTS":           (True, 1),
    # REQUEST_RATE accepts either a numeric (e.g. 10, 10.5) or the
    # string "inf" (send-as-fast-as-possible). Non-empty-string
    # check only; the bench tool does the numeric/string parsing.
    "REQUEST_RATE":          (False, 0),
}

# MAX_NUM_SEQS: OPTIONAL per architecture-doc §2 line 73 — it's
# category-5 (service-tuned). Two scenarios:
#   - Latency: pin MAX_NUM_SEQS=1 in .workload. Single-MNS
#     pipeline; kernel-tune + service-sweep both targeted at the
#     low-concurrency regime.
#   - Throughput: omit MAX_NUM_SEQS from .workload. The service
#     sweep enumerates MNS candidates and the projection picks the
#     best. Kernel-tune uses the max MNS candidate as its mnss-
#     derivation parameter (worst-case ceiling).
# If present, it's validated as int >= 1.
OPTIONAL_INTS: dict[str, int] = {
    "MAX_NUM_SEQS": 1,
}

# Vars that warn-on-presence: they straddle .workload (legacy) and
# .service (correct); the validator nudges operators toward .service.
WARN_PRESENT: tuple[str, ...] = (
    "MAX_NUM_BATCHED_TOKENS",      # service-sweep var (refactor item 7)
    "LONG_PREFILL_TOKEN_THRESHOLD", # kernel-derived pin
    "BLOCK_SIZE",                   # kernel-derived pin (= page_size)
    "RPA_KERNEL_K",                 # kernel-derived pin (= kernel_K)
    "RPA_MAX_NUM_SUBSEQS",          # kernel-derived pin (= mnss)
    "RPA_D_BLOCK_SIZES",            # kernel-derived
    "RPA_M_BLOCK_SIZES",            # kernel-derived
    "RPA_P_BLOCK_SIZES",            # kernel-derived
)


def parse_workload_env(workload_path: Path) -> dict[str, str]:
    """Source the `.workload` bash file in a clean sub-shell.

    Runs `env -i bash --noprofile --norc -c 'env; <SENTINEL>; source
    <file>; env'` and diffs the two env snapshots. Using `env -i` (no
    inherited environment) plus a same-shell diff fixes two prior bugs
    (fix #4):

      1. Pre-set parent-shell vars masking workload vars. Old code
         snapshot-diffed against the calling shell — if the parent
         had `MAX_MODEL_LEN=8192` exported and the workload also set
         `MAX_MODEL_LEN=8192`, the diff returned EMPTY for that var,
         silently dropping it from the validator and from migrate.
      2. Bash-internal seeds (PWD, SHLVL, BASH_*, etc.) leaking into
         the result. Diffing the same clean subshell pre/post-source
         cancels these out automatically — no hard-coded skip list to
         go stale across bash versions.
    """
    import subprocess
    sentinel = "@@@WORKLOAD_DIFF_SENTINEL@@@"
    script = (
        f"set -a; env; echo '{sentinel}'; "
        f"source '{workload_path}'; env"
    )
    proc = subprocess.run(
        ["env", "-i", "bash", "--noprofile", "--norc", "-c", script],
        capture_output=True, text=True, check=True,
    )
    parts = proc.stdout.split(sentinel + "\n", 1)
    if len(parts) != 2:
        # Sentinel split failed — treat the whole output as post-source
        # (no before-state to filter against, so every var surfaces).
        before_text, after_text = "", proc.stdout
    else:
        before_text, after_text = parts

    def _parse(text: str) -> dict[str, str]:
        out: dict[str, str] = {}
        for line in text.splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                out[k] = v
        return out

    before = _parse(before_text)
    after = _parse(after_text)
    return {
        k: v for k, v in after.items()
        if k not in before or before[k] != v
    }


def _validate_int(name: str, raw: str, min_value: int) -> str | None:
    """Return an error message or None."""
    try:
        v = int(raw)
    except ValueError:
        return f"{name}={raw!r} must be an integer"
    if v < min_value:
        return f"{name}={v} must be >= {min_value}"
    return None


def validate(workload_path: Path) -> list[tuple[str, str]]:
    """Validate a `.workload` file. Return (severity, message) list.

    Severity is `"error"` (blocks valid use) or `"warning"` (nudge
    only). Empty list => fully valid.
    """
    issues: list[tuple[str, str]] = []
    if not workload_path.exists():
        issues.append(("error", f"workload file not found: {workload_path}"))
        return issues

    env = parse_workload_env(workload_path)

    for var, (is_int, min_value) in REQUIRED.items():
        if var not in env:
            issues.append(("error", f"missing required variable: {var}"))
            continue
        if not env[var]:
            issues.append(("error", f"variable {var} is empty"))
            continue
        if is_int:
            err = _validate_int(var, env[var], min_value)
            if err is not None:
                issues.append(("error", err))

    # Optional ints: if present, must validate. If absent, no error
    # (the scenario is intentional — see OPTIONAL_INTS comment).
    for var, min_value in OPTIONAL_INTS.items():
        if var not in env or not env[var]:
            continue
        err = _validate_int(var, env[var], min_value)
        if err is not None:
            issues.append(("error", err))

    for var in WARN_PRESENT:
        if var in env:
            issues.append((
                "warning",
                f"{var} should not be set in .workload — it belongs in "
                f".service (sweep dim) or .kernel (derived pin). See "
                f"docs/tuning_architecture.md §2 for the variable "
                f"classification.",
            ))

    # fix #4: invariant — MAX_MODEL_LEN must accommodate prompt + decode.
    # Only checked if all three vars validated OK (no point cascading
    # invariants on top of "missing" or "non-integer" errors).
    if all(v in env and env[v] for v in
           ("MAX_MODEL_LEN", "INPUT_LEN", "OUTPUT_LEN")):
        try:
            mml = int(env["MAX_MODEL_LEN"])
            inp = int(env["INPUT_LEN"])
            out = int(env["OUTPUT_LEN"])
        except ValueError:
            pass
        else:
            if mml < inp + out:
                issues.append((
                    "error",
                    f"MAX_MODEL_LEN={mml} < INPUT_LEN+OUTPUT_LEN="
                    f"{inp}+{out}={inp + out}. vllm bench will reject "
                    f"requests at runtime; raise MAX_MODEL_LEN to at "
                    f"least {inp + out}.",
                ))

    return issues


def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        description="Validate a .workload schema.",
    )
    p.add_argument("workload", type=Path)
    args = p.parse_args(argv)

    issues = validate(args.workload)
    n_errors = sum(1 for sev, _ in issues if sev == "error")
    n_warnings = sum(1 for sev, _ in issues if sev == "warning")
    for severity, msg in issues:
        prefix = "ERROR" if severity == "error" else "WARN "
        print(f"{prefix}: {msg}", file=sys.stderr)
    if not issues:
        print("OK", file=sys.stderr)
    elif n_errors == 0:
        print(f"{n_warnings} warning(s); no errors.", file=sys.stderr)
    return 1 if n_errors > 0 else 0


if __name__ == "__main__":   # pragma: no cover
    sys.exit(main())
