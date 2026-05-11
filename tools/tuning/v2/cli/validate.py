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
    "MAX_NUM_SEQS":          (True, 1),
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
    """Source the `.workload` bash file and return its env as a dict.

    Uses `set -a; source <file>; env` in a sub-shell. Returns only the
    keys the file actually set (filters out the inherited shell env
    by snapshot-diffing).
    """
    import subprocess
    # Snapshot the parent shell's env first so we can subtract it.
    snapshot_proc = subprocess.run(
        ["bash", "-c", "env"],
        capture_output=True, text=True, check=True,
    )
    before: dict[str, str] = {}
    for line in snapshot_proc.stdout.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            before[k] = v

    sourced_proc = subprocess.run(
        ["bash", "-c", f"set -a; source '{workload_path}'; env"],
        capture_output=True, text=True, check=True,
    )
    after: dict[str, str] = {}
    for line in sourced_proc.stdout.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            after[k] = v

    # Vars the workload file set or changed.
    set_by_workload: dict[str, str] = {}
    for k, v in after.items():
        if k not in before or before[k] != v:
            set_by_workload[k] = v
    return set_by_workload


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

    for var in WARN_PRESENT:
        if var in env:
            issues.append((
                "warning",
                f"{var} should not be set in .workload — it belongs in "
                f".service (sweep dim) or .kernel (derived pin). See "
                f"docs/tuning_architecture.md §2 for the variable "
                f"classification.",
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
