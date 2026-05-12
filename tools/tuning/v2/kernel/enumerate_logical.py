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


from tools.tuning.v2.core.discriminator import (
    DEFAULT_HARDWARE,
    DEFAULT_KERNEL_VARIANT,
    ROW_SCHEMA_VERSION,
)


# Static-prune estimators from the kernel module. Lazy/optional so this
# enumerator stays importable on a laptop without vllm/TPU deps (the
# MOCK_TPU path uses enumerate to verify pipeline wiring without ever
# calling the estimators in anger). On a TPU host the imports succeed
# and we filter infeasible combos at enumeration time.
#
# Scope: only VMEM (block-size-dependent tile footprint) and SMEM
# (scalar-prefetch arrays). NOT HBM program memory — the
# `get_hbm_program_memory_estimate_bytes` formula in the kernel module
# is empirical and was found to over-reject known-working configs (the
# Phase 5 LOGICAL winner at mnss=4224, K=256 fits in v7x's region but
# the formula predicts 66 GB). Including it here would block legit
# winners. Future: replace with a first-principles estimator derived
# from the actual pallas_call BlockSpec.
try:
    from tpu_inference.kernels.ragged_paged_attention.v3.kernel import (
        get_smem_estimate_bytes,
        get_vmem_estimate_bytes,
    )
    _STATIC_PRUNE_AVAILABLE = True
except ImportError:
    _STATIC_PRUNE_AVAILABLE = False


# Same as v1's `tools/kernel/tuner/v1/rpa_v3_kernel_tuner.py`:
# VMEM_LIMIT_BYTES = 60 MB, SMEM_LIMIT_BYTES = 0.9 MB. Duplicated here
# rather than imported from v1 because v2 should not depend on v1 for
# constants — when v1 retires (task #14), these stay valid.
# TODO: hardware-aware. v6e has ~16 MB VMEM; this 60 MB ceiling
# silently lets v6e-infeasible combos through to the runtime check.
_VMEM_LIMIT_BYTES = 60 * 1024 * 1024
_SMEM_LIMIT_BYTES = int(0.9 * 1024 * 1024)


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _static_prune_pass(
    *,
    model_shape: dict[str, Any],
    max_num_seqs: int,
    page_size: int,
    kernel_K: int,
    mnss: int,
    bq_sz: int,
    bkv_sz: int,
) -> bool:
    """Return True if the combo is feasible against VMEM/SMEM/HBM
    budgets (or the estimators aren't available locally — fall open).

    Skipped combos: caller `continue`s the inner loop, so the row is
    never yielded and never round-trips through the measurement
    pipeline. v1's run() runtime check is the backstop for anything
    that slips through (e.g. an estimator under-counts).
    """
    if not _STATIC_PRUNE_AVAILABLE:
        return True
    num_q_heads = model_shape["num_q_heads"]
    num_kv_heads = model_shape["num_kv_heads"]
    head_dim = model_shape["head_dim"]
    max_model_len = model_shape["max_model_len"]
    # CRITICAL: dtypes come from workload env as STRINGS ("bfloat16"),
    # but the kernel-module estimators call get_dtype_packing(dtype)
    # which expects a JAX dtype object (jax.dtypes.itemsize_bits would
    # raise on a raw string). Without this translation the prune
    # would crash on TPU — falling open silently because the caller
    # has no exception handler, putting us right back in the May-11
    # failure mode. Mirror the same translation v2/kernel/measurement_tpu
    # does for the TuningKey construction path.
    import jax.numpy as jnp
    _DTYPE_MAP = {
        "bfloat16":      jnp.bfloat16,
        "float16":       jnp.float16,
        "float32":       jnp.float32,
        "float8_e4m3fn": jnp.float8_e4m3fn,
    }
    q_dtype = _DTYPE_MAP.get(model_shape["q_dtype"], model_shape["q_dtype"])
    kv_dtype = _DTYPE_MAP.get(model_shape["kv_dtype"], model_shape["kv_dtype"])
    pages_per_seq = _cdiv(max_model_len, page_size)

    vmem = get_vmem_estimate_bytes(
        actual_num_kv_heads=num_kv_heads,
        actual_num_q_heads_per_kv_head=num_q_heads // num_kv_heads,
        actual_head_dim=head_dim,
        bq_sz=bq_sz,
        bkv_sz=bkv_sz,
        q_dtype=q_dtype,
        kv_dtype=kv_dtype,
    )
    if vmem > _VMEM_LIMIT_BYTES:
        return False
    smem = get_smem_estimate_bytes(
        max_num_seqs,
        pages_per_seq,
        max_num_subseqs=mnss,
    )
    if smem > _SMEM_LIMIT_BYTES:
        return False
    # HBM program-memory check INTENTIONALLY OMITTED — the empirical
    # formula in `get_hbm_program_memory_estimate_bytes` over-rejects
    # known-working winners (Phase 5 LOGICAL at mnss=4224, K=256). The
    # May-11 OOM mode is still caught by: v1's runtime check (after
    # buffer alloc), prev-1 (FAILED_OOM classification), prev-4
    # (dynamic prune after N consecutive non-SUCCESS in the same
    # subspace), and prev-5 (zero-SUCCESS health check at end of tune).
    # Adding a correct HBM static prune requires a first-principles
    # estimator from the kernel author.
    return True


def enumerate_logical_combos(
    *,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    model_shape: dict[str, Any],
    code_revision: str,
    search_space: dict[str, list[int]],
    kernel_variant: str = DEFAULT_KERNEL_VARIANT,
    hardware: str = DEFAULT_HARDWARE,
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
      kernel_variant: plugin discriminator (architecture doc §13.4).
                      Defaults to `"rpa_v3"`. When a second TPU kernel
                      (e.g. `"rpa_v3_hd64"`, `"mla"`) or a GPU plugin
                      lands, the caller passes its own identifier.
      hardware: hardware partition discriminator (architecture doc
                §13.4.2). Defaults to `"tpu_v7x"`. Inferred by the
                caller from the workload's path
                (`cases/<topo>/<model>/`).

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
                        # Static VMEM/SMEM/HBM prune: every doomed combo
                        # caught here is one less round-trip through
                        # generate_inputs (GB-scale HBM alloc) + v1's
                        # runtime checks. Block-size-independent
                        # bounds (SMEM, HBM) are still re-evaluated per
                        # bkv_sz — cheap; structurally clean.
                        if not _static_prune_pass(
                            model_shape=model_shape,
                            max_num_seqs=max_num_seqs,
                            page_size=page_size,
                            kernel_K=kernel_K,
                            mnss=mnss,
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
                                    # Tier 1 (lowest precedence):
                                    # workload pass-through.
                                    **model_shape,
                                    # Tier 2: discriminators
                                    # (architecture doc §13.4).
                                    "kernel_variant":  kernel_variant,
                                    "hardware":        hardware,
                                    # ROW_SCHEMA_VERSION (distinct
                                    # from the production-envelope
                                    # schema_version in accumulator /
                                    # migrate — same name, different
                                    # schema). Stamped today but not
                                    # YET read; the version-2 reader
                                    # arrives with the next row-shape
                                    # change.
                                    "schema_version":  ROW_SCHEMA_VERSION,
                                    # Tier 3 (highest precedence):
                                    # explicit tuning identity.
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
