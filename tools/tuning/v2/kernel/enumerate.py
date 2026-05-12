# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Combo enumerators for each RPA v3 case (D / P / M / L).

Architecture doc §3 line 93: kernel-tune produces one winner per
`(workload, case)`. This module's four enumerators feed the
kernel-tune loop with each case's search space:

  - DECODE  : bq_sz = bq_csz = 1; sweep bkv_sz / bkv_csz / page_size.
  - PREFILL : full block-size sweep + kernel_K (chunk_prefill_size).
              kernel_K % bq_sz == 0 and bq_sz <= kernel_K.
  - MIXED   : full block-size sweep; no kernel_K, no mnss.
  - LOGICAL : PREFILL constraints + mnss + per_phys_q ≤ MNB ceiling.
              (kept in `kernel/enumerate_logical.py` — re-exported here.)

Common constraints (cheap enumeration-time pruning):
  - bq_sz   % bq_csz   == 0
  - bkv_sz  % bkv_csz  == 0
  - bkv_sz  % page_size == 0
  - bkv_csz % page_size == 0
  - bkv_sz / page_size  <= ceil(max_model_len / page_size)

SMEM / VMEM pruning is deferred to `measurement_fn` (requires jnp
dtypes); the runner records the resulting SKIPPED rows.

The combiner `enumerate_all_combos` yields combos across all
requested cases in a stable order (D, M, P, L by default). Used as
the default enumerator in `run_kernel_tune` so a single tune run
produces all four kernel variants' winners.
"""

import math
from typing import Any, Iterator

from tools.tuning.v2.core.discriminator import (
    DEFAULT_HARDWARE,
    DEFAULT_KERNEL_VARIANT,
    ROW_SCHEMA_VERSION,
)
from tools.tuning.v2.kernel.enumerate_logical import (
    enumerate_logical_combos,
)


CASE_DECODE = "decode"
CASE_PREFILL = "prefill"
CASE_MIXED = "mixed"
CASE_LOGICAL = "logical"

ALL_CASES: tuple[str, ...] = (
    CASE_DECODE, CASE_MIXED, CASE_PREFILL, CASE_LOGICAL,
)


def _common_blocks_ok(
    *, page_size: int, bq_sz: int, bkv_sz: int,
    bq_csz: int, bkv_csz: int, max_model_len: int,
) -> bool:
    """Modulo + page-cap constraints shared by every case.

    Mirrors v1's `_block_sizes_valid` so v2 prunes the same
    combos at enumeration time that v1 did. Saves the JAX
    compile-and-OOM round-trip on combos with bad divisibility.
    """
    if bq_sz % bq_csz != 0:
        return False
    if bkv_sz % bkv_csz != 0:
        return False
    if bkv_sz % page_size != 0:
        return False
    if bkv_csz % page_size != 0:
        return False
    pages_per_seq = math.ceil(max_model_len / page_size)
    if bkv_sz // page_size > pages_per_seq:
        return False
    return True


def _build_tuning_key(
    *, model_shape: dict[str, Any], kernel_variant: str, hardware: str,
    case: str, page_size: int, kernel_K: int, max_num_seqs: int,
    code_revision: str,
) -> dict[str, Any]:
    """Common tuning_key construction for any case.

    Three-tier precedence (architecture doc §13.4):
      1. model_shape spread — lowest precedence (workload metadata).
      2. discriminators (kernel_variant, hardware, schema_version) —
         override (1).
      3. explicit identity keys (case, page_size, ...) — override (2).

    Tested by enumerate_logical's
    `test_explicit_keys_win_on_collision`.
    """
    return {
        **model_shape,
        "kernel_variant":  kernel_variant,
        "hardware":        hardware,
        "schema_version":  ROW_SCHEMA_VERSION,
        "case":            case,
        "page_size":       page_size,
        "kernel_K":        kernel_K,
        "max_num_seqs":    max_num_seqs,
        "code_revision":   code_revision,
    }


def enumerate_decode_combos(
    *,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    model_shape: dict[str, Any],
    code_revision: str,
    search_space: dict[str, list[int]],
    kernel_variant: str = DEFAULT_KERNEL_VARIANT,
    hardware: str = DEFAULT_HARDWARE,
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """DECODE: bq_sz = bq_csz = 1 (single q-token per req).

    `kernel_K` is stamped as 0 on the tuning_key (the field is a
    PREFILL/LOGICAL artifact); the measurement adapter translates
    that to v1's `chunk_prefill_size=0`. No mnss in tunable_params.

    `max_num_batched_tokens` unused — DECODE doesn't multiply per-
    phys q tokens; the MNB ceiling doesn't constrain it.
    """
    del max_num_batched_tokens
    max_model_len = model_shape.get("max_model_len", 0)
    bq_sz, bq_csz = 1, 1
    for page_size in search_space["page_size"]:
        for bkv_sz in search_space["bkv_sz"]:
            for bkv_csz in search_space["bkv_csz"]:
                if not _common_blocks_ok(
                    page_size=page_size, bq_sz=bq_sz, bkv_sz=bkv_sz,
                    bq_csz=bq_csz, bkv_csz=bkv_csz,
                    max_model_len=max_model_len,
                ):
                    continue
                tk = _build_tuning_key(
                    model_shape=model_shape,
                    kernel_variant=kernel_variant,
                    hardware=hardware,
                    case=CASE_DECODE,
                    page_size=page_size,
                    kernel_K=0,
                    max_num_seqs=max_num_seqs,
                    code_revision=code_revision,
                )
                tp = {
                    "bq_sz":   bq_sz,
                    "bkv_sz":  bkv_sz,
                    "bq_csz":  bq_csz,
                    "bkv_csz": bkv_csz,
                }
                yield tk, tp


def enumerate_mixed_combos(
    *,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    model_shape: dict[str, Any],
    code_revision: str,
    search_space: dict[str, list[int]],
    kernel_variant: str = DEFAULT_KERNEL_VARIANT,
    hardware: str = DEFAULT_HARDWARE,
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """MIXED: full block-size sweep; no kernel_K, no mnss.

    Same shape as DECODE in terms of tunable_params (bq + bkv +
    block sizes only), but bq_sz / bq_csz get the full sweep
    instead of being pinned to 1.
    """
    del max_num_batched_tokens
    max_model_len = model_shape.get("max_model_len", 0)
    for page_size in search_space["page_size"]:
        for bq_sz in search_space["bq_sz"]:
            for bkv_sz in search_space["bkv_sz"]:
                for bq_csz in search_space["bq_csz"]:
                    if bq_csz > bq_sz:
                        continue
                    for bkv_csz in search_space["bkv_csz"]:
                        if bkv_csz > bkv_sz:
                            continue
                        if not _common_blocks_ok(
                            page_size=page_size, bq_sz=bq_sz,
                            bkv_sz=bkv_sz, bq_csz=bq_csz,
                            bkv_csz=bkv_csz,
                            max_model_len=max_model_len,
                        ):
                            continue
                        tk = _build_tuning_key(
                            model_shape=model_shape,
                            kernel_variant=kernel_variant,
                            hardware=hardware,
                            case=CASE_MIXED,
                            page_size=page_size,
                            kernel_K=0,
                            max_num_seqs=max_num_seqs,
                            code_revision=code_revision,
                        )
                        tp = {
                            "bq_sz":   bq_sz,
                            "bkv_sz":  bkv_sz,
                            "bq_csz":  bq_csz,
                            "bkv_csz": bkv_csz,
                        }
                        yield tk, tp


def enumerate_prefill_combos(
    *,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    model_shape: dict[str, Any],
    code_revision: str,
    search_space: dict[str, list[int]],
    kernel_variant: str = DEFAULT_KERNEL_VARIANT,
    hardware: str = DEFAULT_HARDWARE,
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """PREFILL: full block-size sweep + kernel_K sweep.

    PREFILL-specific constraints (mirror v1):
      - kernel_K % bq_sz == 0  (no within-iter padding waste)
      - bq_sz <= kernel_K      (one q-block fits per iter)

    No mnss in tunable_params (PREFILL is coupled-K).
    `max_num_batched_tokens` not used as a pruning constraint here —
    PREFILL doesn't pack multiple physical requests per iter, so the
    LOGICAL `per_phys_q ≤ MNB` rule doesn't apply.
    """
    del max_num_batched_tokens
    # prev-2a: PREFILL is coupled-K so max_num_subseqs collapses to
    # max_num_seqs. Reuse the LOGICAL enumerator's VMEM/SMEM prune
    # helper for symmetry (audit's `feedback_symmetric_layer_hygiene`
    # rule). The helper does NOT do an HBM-program-memory check
    # because the available empirical formula over-rejects known-
    # working winners — see _static_prune_pass docstring.
    from tools.tuning.v2.kernel.enumerate_logical import _static_prune_pass
    max_model_len = model_shape.get("max_model_len", 0)
    for page_size in search_space["page_size"]:
        for kernel_K in search_space["kernel_K"]:
            for bq_sz in search_space["bq_sz"]:
                if bq_sz > kernel_K:
                    continue
                if kernel_K % bq_sz != 0:
                    continue
                for bkv_sz in search_space["bkv_sz"]:
                    # Static prune (block-size-dependent VMEM + the
                    # MNS-coupled HBM check). Cheap; clamps the inner
                    # bq_csz / bkv_csz loops below to combos that can
                    # actually compile.
                    if not _static_prune_pass(
                        model_shape=model_shape,
                        max_num_seqs=max_num_seqs,
                        page_size=page_size,
                        kernel_K=kernel_K,
                        # PREFILL is coupled-K — pass max_num_seqs in
                        # place of mnss so the SMEM/HBM estimators
                        # treat each iter as one phys seq (the
                        # coupled-K identity).
                        mnss=max_num_seqs,
                        bq_sz=bq_sz,
                        bkv_sz=bkv_sz,
                    ):
                        continue
                    for bq_csz in search_space["bq_csz"]:
                        if bq_csz > bq_sz:
                            continue
                        for bkv_csz in search_space["bkv_csz"]:
                            if bkv_csz > bkv_sz:
                                continue
                            if not _common_blocks_ok(
                                page_size=page_size, bq_sz=bq_sz,
                                bkv_sz=bkv_sz, bq_csz=bq_csz,
                                bkv_csz=bkv_csz,
                                max_model_len=max_model_len,
                            ):
                                continue
                            tk = _build_tuning_key(
                                model_shape=model_shape,
                                kernel_variant=kernel_variant,
                                hardware=hardware,
                                case=CASE_PREFILL,
                                page_size=page_size,
                                kernel_K=kernel_K,
                                max_num_seqs=max_num_seqs,
                                code_revision=code_revision,
                            )
                            tp = {
                                "bq_sz":   bq_sz,
                                "bkv_sz":  bkv_sz,
                                "bq_csz":  bq_csz,
                                "bkv_csz": bkv_csz,
                            }
                            yield tk, tp


_CASE_TO_ENUMERATOR = {
    CASE_DECODE:  enumerate_decode_combos,
    CASE_MIXED:   enumerate_mixed_combos,
    CASE_PREFILL: enumerate_prefill_combos,
    CASE_LOGICAL: enumerate_logical_combos,
}


def enumerate_all_combos(
    *,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    model_shape: dict[str, Any],
    code_revision: str,
    search_space: dict[str, list[int]],
    kernel_variant: str = DEFAULT_KERNEL_VARIANT,
    hardware: str = DEFAULT_HARDWARE,
    cases: tuple[str, ...] = ALL_CASES,
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """Yield combos across every requested case.

    Default `cases = (decode, mixed, prefill, logical)` — the order
    is operator-readable (D first, complex L last). Per-case
    enumerators are responsible for their own constraint pruning;
    this combiner just chains them.
    """
    for case in cases:
        try:
            fn = _CASE_TO_ENUMERATOR[case]
        except KeyError:
            raise ValueError(
                f"unknown case {case!r}; valid: "
                f"{sorted(_CASE_TO_ENUMERATOR)}",
            ) from None
        yield from fn(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            model_shape=model_shape,
            code_revision=code_revision,
            search_space=search_space,
            kernel_variant=kernel_variant,
            hardware=hardware,
        )
