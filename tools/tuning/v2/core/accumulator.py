# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Cross-workload accumulator: per-workload {.kernel,.service} → production.

Each workload owns its own `<workload>.kernel` and `<workload>.service`
(produced by the projection step). The accumulator unions them into a
`production.kernel` and `production.service` at the per-model level —
the file lookup the existing v1 sweep/auto-link already consults, and
the substrate for impact-analysis queries ("which workloads have a
LOGICAL winner at K=256?", §7 of the architecture doc).

Pure data movement:

  union_per_workload(workload_files, parse=json.loads) -> dict[str, dict]
    {workload_name: parsed_content_of_that_workload_file}

  discover_workload_files(model_dir, suffix) -> list[Path]
    All `<workload><suffix>` in `model_dir`, excluding `production<suffix>`.

The CLI layer (tools/tuning/v2/cli/aggregate.py) composes these with
file I/O + git_atomic.commit_and_push to produce the final
`production.kernel` and `production.service`. Keeping the union as a
pure function makes it trivial to test.
"""

import json
from pathlib import Path
from typing import Any, Callable


def discover_workload_files(
    model_dir: Path,
    suffix: str,
) -> list[Path]:
    """Return sorted list of `<workload><suffix>` files in `model_dir`.

    Excludes any file whose name starts with `production` so the
    accumulator output doesn't loop back into the input. Excludes the
    `.raw/` directories (which are dirs, not files matching the glob).

    Args:
      model_dir: A model directory, e.g. `tools/benchmark/cases/v7x/llama3_8b/`.
      suffix: e.g. `.kernel` or `.service`. Leading dot included.

    Returns:
      Sorted list of paths (alphabetical workload name).
    """
    if not model_dir.exists() or not model_dir.is_dir():
        return []
    # Use glob with the suffix directly so directories like
    # `<workload>.kernel.raw/` are not matched (they don't end in
    # the suffix as a file). Exact-match exclusion of the production
    # output (fix #13): a file like `production_v2.kernel` is a legit
    # workload-named source, not the accumulator's output. Old code
    # used `startswith("production")` which dropped it incorrectly.
    production_name = f"production{suffix}"
    candidates = sorted(
        p for p in model_dir.glob(f"*{suffix}")
        if p.is_file() and p.name != production_name
    )
    return candidates


def _workload_name(path: Path, suffix: str) -> str:
    """Strip the suffix from path.name. Default to `.stem` semantics.

    Empty `suffix` falls back to `.stem` (which strips only the last
    extension; "foo.bar.kernel" -> "foo.bar"). Without this guard the
    `name.endswith("")` would always match and slice to the empty string.
    """
    name = path.name
    if suffix and name.endswith(suffix):
        return name[: -len(suffix)]
    return path.stem


def union_per_workload(
    workload_files: list[Path],
    suffix: str = "",
    parse: Callable[[str], Any] = json.loads,
) -> dict[str, Any]:
    """Read each workload file, return a dict keyed by workload name.

    Args:
      workload_files: List of paths, typically from
                      `discover_workload_files`.
      suffix: The suffix to strip from each filename to derive the
              workload name (e.g. `.kernel`). Empty string falls back
              to `Path.stem` (which strips the LAST `.` component only).
      parse: Function (file_text -> parsed_value). Defaults to JSON
             parsing; override for non-JSON formats.

    Returns:
      `{workload_name: parsed_content}` for every file. Workload names
      are sorted (insertion order matches caller's sorted list).

    Errors propagate: file-not-found, JSON-decode error, etc. The
    caller (CLI layer) chooses whether to ignore or surface them.
    """
    out: dict[str, Any] = {}
    for path in workload_files:
        name = _workload_name(path, suffix)
        with open(path, "r", encoding="utf-8") as f:
            out[name] = parse(f.read())
    return out


def build_production(
    model_dir: Path,
    suffix: str,
    *,
    topo: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Build a `production<suffix>` document from a model directory.

    Combines discovery + union with a small wrapping envelope that
    records `topo` / `model` (for impact-analysis queries) and the
    schema version. The result is JSON-serialisable.

    Args:
      model_dir: Path to a model directory.
      suffix: `.kernel` or `.service`.
      topo, model: Optional human-readable identity. If not supplied,
                   inferred from the path (`.../cases/<topo>/<model>/`).

    Returns:
      Dict shaped:
        {
          "schema_version": 1,
          "topo": <str>,
          "model": <str>,
          "by_workload": {
            "<workload>": <contents of <workload><suffix>>,
            ...
          }
        }
    """
    files = discover_workload_files(model_dir, suffix)
    by_workload = union_per_workload(files, suffix=suffix)
    if topo is None or model is None:
        inferred_topo, inferred_model = _infer_topo_model(model_dir)
        if topo is None:
            topo = inferred_topo
        if model is None:
            model = inferred_model
    # `schema_version` here = PRODUCTION-envelope schema version,
    # distinct from the per-row ROW_SCHEMA_VERSION stamped inside
    # tuning_key (core/discriminator.py). Same field name, different
    # schema: this one versions the `{topo, model, by_workload}`
    # wrapper; the row one versions individual tuning_key shapes.
    return {
        "schema_version": 1,
        "topo": topo,
        "model": model,
        "by_workload": by_workload,
    }


def _infer_topo_model(model_dir: Path) -> tuple[str, str]:
    """Try to infer (topo, model) from a `cases/<topo>/<model>/` path.

    Walks up at most two levels. Falls back to `("unknown", "unknown")`
    if the conventional path shape isn't found.
    """
    resolved = model_dir.resolve()
    parts = resolved.parts
    # Look for "cases" in path; the two segments after it are topo, model.
    try:
        idx = parts.index("cases")
    except ValueError:
        return "unknown", "unknown"
    if idx + 2 >= len(parts):
        return "unknown", "unknown"
    return parts[idx + 1], parts[idx + 2]
