# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Pure-CPU projection: raw measurements → winners.

The projection step is the second half of every tuning pipeline:

  raw_store.read_rows(.kernel.raw/<sha>.jsonl)
      → project_winners(group=tuning_key, objective=latency_us, ascending)
      → write <workload>.kernel

  raw_store.read_rows(.service.raw/<sha>.jsonl)
      → project_winners(group=workload_pin_keys, objective=req/s, descending)
      → write <workload>.service[throughput_max]
      → project_winners(group=workload_pin_keys, objective=ttft_mean, ascending)
      → write <workload>.service[ttft_min]
      … etc per objective.

This module defines the pure-CPU projector. Kernel and service layers
(in tools/tuning/v2/kernel/project.py and service/project.py) compose
this with their domain-specific group / objective / skip functions.

The projector is **idempotent**: same input rows + same parameters →
same output. No side effects, no I/O, no globals. Easy to test.

**TODO when the second plugin lands** (architecture doc §13): the
kernel-side group_fn in `kernel/project._kernel_group_key` does NOT
include `kernel_variant` / `hardware` today, because only one valid
value of each exists. If two plugins ever produce rows for the same
workload (e.g. an rpa_v3 winner and an rpa_v3_hd64 winner during a
migration window), their rows collapse into one group and one of
the two winners gets silently dropped. Add `kernel_variant` and
`hardware` to the group key as soon as KNOWN_KERNEL_VARIANTS grows
past size 1. Same concern applies to `service/project` if service
sweeps ever cross-pollinate plugins through pin_keys.
"""

from typing import Any, Callable, Hashable, Iterable

from tools.tuning.v2.core.keyset import canonical_json


def project_winners(
    rows: Iterable[dict[str, Any]],
    group_fn: Callable[[dict[str, Any]], Hashable],
    objective_fn: Callable[[dict[str, Any]], float],
    descending: bool = False,
    skip_if: Callable[[dict[str, Any]], bool] | None = None,
) -> list[dict[str, Any]]:
    """Group rows by `group_fn`, pick the winner in each group.

    Args:
      rows: iterable of raw-row dicts (typically from raw_store.read_rows).
      group_fn: row → group key. Hashable; the canonical_json of this
                key determines sort order in the result.
      objective_fn: row → numeric score. Compared with `<` (ascending)
                    or `>` (descending) to pick the per-group winner.
      descending: False (default) → smaller score wins, e.g. latency.
                  True → larger score wins, e.g. throughput.
      skip_if: optional predicate. Rows for which `skip_if(row)` is
               True are dropped before grouping. Use to filter by
               status (`r.get('status') != 'SUCCESS'`) or to drop
               rows missing the objective field. Caller's choice.

    Returns:
      List of winning rows, one per unique group key. Stable order:
      sorted by `canonical_json(group_key)`.

    Determinism: ties (equal objective values) preserve the FIRST row
    seen in input order. Same input → same output, regardless of how
    many times the function is called.

    Errors propagate. If `objective_fn` raises on a row that wasn't
    filtered by `skip_if`, the projection raises. The caller's
    `skip_if` is the contract for "rows that survive the filter have
    a valid objective".
    """
    best: dict[Hashable, tuple[float, dict[str, Any]]] = {}
    for row in rows:
        if skip_if is not None and skip_if(row):
            continue
        score = objective_fn(row)
        key = group_fn(row)
        if key not in best:
            best[key] = (score, row)
            continue
        current_score, _ = best[key]
        if descending:
            wins = score > current_score
        else:
            wins = score < current_score
        if wins:
            best[key] = (score, row)
    return [
        row for _key, (_score, row) in sorted(
            best.items(),
            key=lambda kv: canonical_json(kv[0]),
        )
    ]
