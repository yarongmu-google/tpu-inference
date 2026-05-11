# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Combo enumerator for the LOGICAL (decoupled-K) kernel case.

Generates `(tuning_key, tunable_params)` pairs from a search space,
applying the LOGICAL-case constraints inline:

  1. `mnss >= max_num_seqs`              # prefetch arrays must fit at
                                            least one slot per phys req.
  2. `per_phys_q = K × (mnss/mns - 1) <= MNB`
                                          # synthetic workload sizing
                                            (logical_workload_sizing.py:101).
  3. `bq_sz <= kernel_K`                  # Q-tile can't exceed the
                                            kernel's static Q-axis.
  4. `bq_csz <= bq_sz`, `bkv_csz <= bkv_sz`
                                          # VReg sub-tile fits VMEM tile.

Combos that fail any constraint are silently skipped (not yielded).
Constraint #2 is the lesson from the rpa3_2 Phase 4 case: at MNS=1 /
MNB=8192 / K=256, only mnss=33 is valid (per_phys_q=8192 just fits);
larger mnss values overflow the synthetic workload.

Sibling enumerators for DECODE / MIXED / PREFILL come in later
commits. The runner (kernel/tune.py) accepts any callable matching
this signature, so each case is pluggable.
"""

from typing import Any, Iterator


def enumerate_logical_combos(
    *,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    model_shape: dict[str, Any],
    code_revision: str,
    search_space: dict[str, list[int]],
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """Yield each valid `(tuning_key, tunable_params)` pair for LOGICAL.

    Args:
      max_num_seqs: workload's MAX_NUM_SEQS.
      max_num_batched_tokens: workload's MAX_NUM_BATCHED_TOKENS.
      model_shape: dict with `num_q_heads`, `num_kv_heads`, `head_dim`,
                   `max_model_len`, `q_dtype`, `kv_dtype`,
                   `sliding_window`.
      code_revision: kernel-source SHA (8-char).
      search_space: dict from `kernel/search_space.py`. Required keys:
                    `page_size`, `kernel_K`, `mnss`, `bq_sz`, `bkv_sz`,
                    `bq_csz`, `bkv_csz`.

    Yields:
      `(tuning_key, tunable_params)` tuples. Order: nested-loop iteration
      from outermost (`page_size`) to innermost (`bkv_csz`).
    """
    for page_size in search_space["page_size"]:
        for kernel_K in search_space["kernel_K"]:
            for mnss in search_space["mnss"]:
                if mnss < max_num_seqs:
                    continue
                m_per_phys = max(1, mnss // max_num_seqs - 1)
                per_phys_q = kernel_K * m_per_phys
                if per_phys_q > max_num_batched_tokens:
                    continue
                for bq_sz in search_space["bq_sz"]:
                    if bq_sz > kernel_K:
                        continue
                    for bkv_sz in search_space["bkv_sz"]:
                        for bq_csz in search_space["bq_csz"]:
                            if bq_csz > bq_sz:
                                continue
                            for bkv_csz in search_space["bkv_csz"]:
                                if bkv_csz > bkv_sz:
                                    continue
                                # Spread-order precedence (fix #14):
                                # this is REORDERING, not namespacing.
                                # `**model_shape` goes first so any
                                # future colliding key in model_shape
                                # is overwritten by the explicit
                                # identity keys below — the dict
                                # literal's later keys win. A true
                                # namespace would nest model_shape
                                # under its own subkey; we don't, to
                                # avoid breaking consumers that read
                                # `tuning_key["num_q_heads"]` directly.
                                # Tested by
                                # test_explicit_keys_win_on_collision.
                                tuning_key = {
                                    **model_shape,
                                    "case":            "logical",
                                    "page_size":       page_size,
                                    "kernel_K":        kernel_K,
                                    "max_num_seqs":    max_num_seqs,
                                    "code_revision":   code_revision,
                                }
                                tunable_params = {
                                    "bq_sz":   bq_sz,
                                    "bkv_sz":  bkv_sz,
                                    "bq_csz":  bq_csz,
                                    "bkv_csz": bkv_csz,
                                    "mnss":    mnss,
                                }
                                yield tuning_key, tunable_params
