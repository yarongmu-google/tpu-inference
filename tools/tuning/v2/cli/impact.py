# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Bottom-up impact-analysis CLI (architecture doc §1 goal 5 + §7).

Given a low-level change (kernel-source SHA bump, a specific
tunable param value, a service combo value), answer which
workloads / deployments are affected. The doc lists three concrete
queries that today are `grep -r` against git-tracked JSON files;
this module formalises them.

Subcommands:

  by-kernel-key   <field> <value> [--root <dir>]
    Find .kernel files whose winners contain
    tuning_key[field] == value (or tunable_params[field] == value).
    Example:
      python3 -m tools.tuning.v2.cli.impact by-kernel-key \\
        kernel_K 256 --root tools/benchmark/cases

  by-service-combo <field> <value> [--root <dir>]
    Find .service files whose winners[*].combo[field] == value.
    Example:
      python3 -m tools.tuning.v2.cli.impact by-service-combo \\
        MAX_NUM_BATCHED_TOKENS 131072 --root tools/benchmark/cases

  stale-tunes      <current_sha> [--root <dir>]
    Find .kernel files whose winners have code_revision != current_sha.
    These workloads have stale kernel tunes and need re-tuning.
    Example:
      python3 -m tools.tuning.v2.cli.impact stale-tunes \\
        abc12345 --root tools/benchmark/cases

Output: one match per line, `<workload_path>` or
`<workload_path>:<winner_summary>`. Exits 0 iff at least one match.

Each subcommand is a thin wrapper over a library function exposed
from this module — same `(rows -> list of paths)` shape — so other
tooling (CI, regression checks) can import and reuse.
"""

import json
import sys
from pathlib import Path
from typing import Any, Callable, Iterable


def find_kernel_files(root: Path) -> list[Path]:
    """All `<workload>.kernel` files under `root` (excluding
    `production.kernel` accumulator outputs).

    Walks the tree; useful as the input to every by-* query.
    """
    if not root.exists():
        return []
    out: list[Path] = []
    for p in sorted(root.rglob("*.kernel")):
        if not p.is_file():
            continue
        if p.name == "production.kernel":
            continue
        out.append(p)
    return out


def find_service_files(root: Path) -> list[Path]:
    """All `<workload>.service` files under `root` (excluding
    `production.service`)."""
    if not root.exists():
        return []
    out: list[Path] = []
    for p in sorted(root.rglob("*.service")):
        if not p.is_file():
            continue
        if p.name == "production.service":
            continue
        out.append(p)
    return out


def _coerce(value: str) -> Any:
    """Best-effort: try int, then float, then keep as string. JSON
    fields are typed; CLI args arrive as strings. Without coercion
    the operator's `kernel_K 256` query would never match an int
    256 stored in JSON."""
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def kernel_winners_matching(
    kernel_doc: dict[str, Any],
    field: str,
    value: Any,
) -> list[dict[str, Any]]:
    """Return winners whose tuning_key OR tunable_params has
    `field == value`. Both sub-dicts are searched so the operator
    doesn't have to remember which side a given field lives on."""
    out: list[dict[str, Any]] = []
    for w in kernel_doc.get("winners", []):
        tk = w.get("tuning_key", {}) or {}
        tp = w.get("tunable_params", {}) or {}
        if tk.get(field) == value or tp.get(field) == value:
            out.append(w)
    return out


def service_winners_matching(
    service_doc: dict[str, Any],
    field: str,
    value: Any,
) -> list[tuple[str, dict[str, Any]]]:
    """Return (objective, winner) pairs whose winner.combo[field] ==
    value. None winners (objective not yet swept) are skipped."""
    out: list[tuple[str, dict[str, Any]]] = []
    for obj, winner in (service_doc.get("winners") or {}).items():
        if winner is None:
            continue
        combo = winner.get("combo", {}) or {}
        if combo.get(field) == value:
            out.append((obj, winner))
    return out


def by_kernel_key(
    root: Path, field: str, value: Any,
) -> list[tuple[Path, list[dict[str, Any]]]]:
    """For every `<workload>.kernel` under root, return (path, matches).
    Files with zero matches are omitted from the result."""
    out: list[tuple[Path, list[dict[str, Any]]]] = []
    for kpath in find_kernel_files(root):
        try:
            with open(kpath, "r", encoding="utf-8") as f:
                doc = json.load(f)
        except json.JSONDecodeError:
            continue
        matches = kernel_winners_matching(doc, field, value)
        if matches:
            out.append((kpath, matches))
    return out


def by_service_combo(
    root: Path, field: str, value: Any,
) -> list[tuple[Path, list[tuple[str, dict[str, Any]]]]]:
    """For every `<workload>.service` under root, return (path, matches).
    Each match is (objective_name, winner_dict)."""
    out: list[tuple[Path, list[tuple[str, dict[str, Any]]]]] = []
    for spath in find_service_files(root):
        try:
            with open(spath, "r", encoding="utf-8") as f:
                doc = json.load(f)
        except json.JSONDecodeError:
            continue
        matches = service_winners_matching(doc, field, value)
        if matches:
            out.append((spath, matches))
    return out


def stale_tunes(
    root: Path, current_sha: str,
) -> list[tuple[Path, list[str]]]:
    """For every `<workload>.kernel`, return (path, stale_revisions).
    `stale_revisions` is the distinct set of code_revision values
    present in the file's winners that aren't equal to current_sha.
    Files where every winner is current are omitted."""
    out: list[tuple[Path, list[str]]] = []
    for kpath in find_kernel_files(root):
        try:
            with open(kpath, "r", encoding="utf-8") as f:
                doc = json.load(f)
        except json.JSONDecodeError:
            continue
        stale: set[str] = set()
        for w in doc.get("winners", []):
            rev = (w.get("tuning_key", {}) or {}).get("code_revision")
            if rev and rev != current_sha:
                stale.add(rev)
        if stale:
            out.append((kpath, sorted(stale)))
    return out


def _fmt_kernel_winner(w: dict[str, Any]) -> str:
    """Short one-line summary for terminal output."""
    tk = w.get("tuning_key", {}) or {}
    case = tk.get("case", "?")
    return (
        f"case={case} page_size={tk.get('page_size')} "
        f"kernel_K={tk.get('kernel_K')} mnss="
        f"{(w.get('tunable_params') or {}).get('mnss')} "
        f"rev={tk.get('code_revision')}"
    )


def _fmt_service_winner(obj: str, w: dict[str, Any]) -> str:
    combo = w.get("combo", {}) or {}
    return f"{obj}: " + " ".join(f"{k}={v}" for k, v in sorted(combo.items()))


def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        description="Bottom-up impact-analysis queries (arch doc §7).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_kk = sub.add_parser(
        "by-kernel-key",
        help="Find .kernel files whose winners match a tuning_key / "
             "tunable_params field.",
    )
    p_kk.add_argument("field")
    p_kk.add_argument("value")
    p_kk.add_argument("--root", type=Path, default=Path("tools/benchmark/cases"))

    p_sc = sub.add_parser(
        "by-service-combo",
        help="Find .service files whose winners match a combo field.",
    )
    p_sc.add_argument("field")
    p_sc.add_argument("value")
    p_sc.add_argument("--root", type=Path, default=Path("tools/benchmark/cases"))

    p_st = sub.add_parser(
        "stale-tunes",
        help="Find .kernel files whose winners are not on the given SHA.",
    )
    p_st.add_argument("current_sha")
    p_st.add_argument("--root", type=Path, default=Path("tools/benchmark/cases"))

    args = p.parse_args(argv)

    if args.cmd == "by-kernel-key":
        value = _coerce(args.value)
        matches = by_kernel_key(args.root, args.field, value)
        for path, winners in matches:
            for w in winners:
                print(f"{path}: {_fmt_kernel_winner(w)}")
        return 0 if matches else 1

    if args.cmd == "by-service-combo":
        value = _coerce(args.value)
        matches = by_service_combo(args.root, args.field, value)
        for path, hits in matches:
            for obj, w in hits:
                print(f"{path}: {_fmt_service_winner(obj, w)}")
        return 0 if matches else 1

    if args.cmd == "stale-tunes":
        matches = stale_tunes(args.root, args.current_sha)
        for path, revs in matches:
            print(f"{path}: stale revisions {revs}")
        return 0 if matches else 1

    # Unreachable: argparse `required=True` on the subparser
    # rejects missing/unknown subcommands before we get here.
    p.error(f"unknown subcommand {args.cmd!r}")    # pragma: no cover
    return 1                                        # pragma: no cover


if __name__ == "__main__":   # pragma: no cover
    sys.exit(main())
