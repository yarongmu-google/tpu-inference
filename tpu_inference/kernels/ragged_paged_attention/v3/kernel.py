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
"""TPU-Friendly Ragged Paged Attention kernel.

This kernel offers a highly optimized implementation of ragged paged attention,
specifically designed for TPU and compatible with a wide range of model
specifications. It supports mixed prefill and decoding, enhancing throughput
during inference.
"""
import functools
from enum import Enum
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.ragged_paged_attention.v3.util import (
    align_to, cdiv, get_dtype_packing, get_tpu_version, next_power_of_2)


class RpaCase(Enum):
    """Represents the different cases for Ragged Paged Attention.

  - DECODE: Sequences are in decode-only mode (q_len = 1).
  - PREFILL: Sequences are in prefill-only mode (q_len > 1, static).
  - MIXED: Sequences can be a mix of prefill and decode (q_len > 1, dynamic).
  - LOGICAL: Decoupled-K PREFILL. Multiple sub-seqs per physical seq via
    phys_seq_indices / q_offsets indirection. q_len per
    sub-seq is static (= K_kernel). LOGICAL replaces the PREFILL bucket
    position in distribution; a step still has D / L / M sub-buckets
    just like D / P / M today. See §6.2 / §6.4 in
    docs/rpa_v3_decoupled_k_kernel_design.md.
  """
    DECODE = 0
    PREFILL = 1
    MIXED = 2
    LOGICAL = 3

    @property
    def symbol(self):
        return {
            RpaCase.DECODE: "d",
            RpaCase.PREFILL: "p",
            RpaCase.MIXED: "m",
            RpaCase.LOGICAL: "l",
        }[self]

    def get_range(self, distribution):
        assert distribution.shape == (3, )
        if self == RpaCase.DECODE:
            return 0, distribution[0]
        elif self == RpaCase.PREFILL or self == RpaCase.LOGICAL:
            # LOGICAL replaces PREFILL at the same bucket position.
            return distribution[0], distribution[1]
        elif self == RpaCase.MIXED:
            return distribution[1], distribution[2]
        else:
            raise ValueError(f"Unsupported RPA case: {self}")


class _IterQuantities(NamedTuple):
    """§6.2 single-branch derivation result.

    NamedTuple → pytree-compatible in JAX, so the same struct flows
    through both pure-JAX (reference) and Pallas (kernel) tracing
    contexts. Fields:

      - `phys_seq_idx`, `q_offset`: raw indirection from the iter-keyed
        prefetches.
      - `phys_q_start`, `phys_q_end`, `phys_q_len`, `phys_kv_len`,
        `phys_kv_q_gap`: phys-level intermediates (used by callers that
        need the underlying phys quantities, e.g. for paged-cache
        gather setup in the reference).
      - `q_start`, `q_end`, `q_len`, `kv_q_gap`, `kv_len`: iter-level
        quantities (what the kernel body operates on; under coupled-K
        identity prefetches these equal today's per-seq quantities).
    """
    phys_seq_idx: jax.Array
    q_offset: jax.Array
    phys_q_start: jax.Array
    phys_q_end: jax.Array
    phys_q_len: jax.Array
    phys_kv_len: jax.Array
    phys_kv_q_gap: jax.Array
    q_start: jax.Array
    q_end: jax.Array
    q_len: jax.Array
    kv_q_gap: jax.Array
    kv_len: jax.Array


def _derive_iter_quantities(
    seq_idx,
    phys_seq_indices,  # SMEM ref (kernel) or jax.Array (reference)
    q_offsets,         # same
    cu_q_lens,
    kv_lens,
    case: "RpaCase",
    static_q_len: int | None,
) -> _IterQuantities:
    """Single source of truth for §6.2 iter→phys derivation.

    Used by:
      - `_ragged_paged_attention_kernel_loop` outer body and `_fetch_bkv`
        closure (kernel context, SMEM-ref args).
      - `ref_ragged_paged_attention_logical` (pure-JAX context, jax.Array
        args).

    Coupled-K (D/P/M) callers pass identity prefetches:
    `phys_seq_indices = arange(max_num_seqs)`, `q_offsets = zeros(...)`.
    Under those identity inputs the formulas reduce to today's behavior:
    `phys_seq_idx == seq_idx`, `q_offset == 0`, etc.

    LOGICAL callers pass real iter-keyed indirection arrays. The
    `case == RpaCase.LOGICAL` branch overrides `q_end` to be derived
    from `static_q_len` (= K_kernel) instead of phys's q_end, since
    LOGICAL iters handle one K_kernel-sized chunk of phys's full
    contribution (which spans multiple iters).

    Pallas/JAX inlines this helper at trace time — zero perf cost
    vs. inlining the body at every site.
    """
    phys_seq_idx = phys_seq_indices[seq_idx]
    q_offset = q_offsets[seq_idx]
    phys_q_start = cu_q_lens[phys_seq_idx]
    phys_q_end = cu_q_lens[phys_seq_idx + 1]
    phys_q_len = phys_q_end - phys_q_start
    phys_kv_len = kv_lens[phys_seq_idx]
    phys_kv_q_gap = phys_kv_len - phys_q_len
    q_start = phys_q_start + q_offset
    if case == RpaCase.LOGICAL:
        q_end = q_start + static_q_len
    else:
        q_end = phys_q_end
    q_len = q_end - q_start
    kv_q_gap = phys_kv_q_gap + q_offset
    kv_len = kv_q_gap + q_len
    return _IterQuantities(
        phys_seq_idx=phys_seq_idx,
        q_offset=q_offset,
        phys_q_start=phys_q_start,
        phys_q_end=phys_q_end,
        phys_q_len=phys_q_len,
        phys_kv_len=phys_kv_len,
        phys_kv_q_gap=phys_kv_q_gap,
        q_start=q_start,
        q_end=q_end,
        q_len=q_len,
        kv_q_gap=kv_q_gap,
        kv_len=kv_len,
    )


def ref_ragged_paged_attention(
    queries: jax.
    Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim]
    keys: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    values: jax.
    Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    kv_cache: jax.
    Array,  # [total_num_pages, page_size, num_kv_heads_x2 // kv_packing, kv_packing, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    *,
    use_causal_mask: bool = True,
    skip_kv_mask: bool = False,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    out_dtype: Any = None,
    mask_value: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
):
    if out_dtype is None:
        out_dtype = jnp.float32 if queries.dtype == jnp.float32 else jnp.bfloat16

    if mask_value is None:
        # We do not set to -inf directly because (-inf) - (-inf) is nan.
        mask_value = jnp.finfo(out_dtype).min
    dynamic_validate_inputs(
        queries,
        keys,
        values,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        use_causal_mask=use_causal_mask,
        skip_kv_mask=skip_kv_mask,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        out_dtype=out_dtype,
        mask_value=mask_value,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    actual_head_dim = queries.shape[2]
    actual_num_q_heads = queries.shape[1]
    actual_num_kv_heads = keys.shape[1]
    merged_kv = merge_kv(keys, values)
    assert merged_kv.shape[-3:] == kv_cache.shape[-3:]

    _, page_size, num_kv_heads_x2_per_kv_packing, kv_packing, head_dim = (
        kv_cache.shape)
    num_kv_heads_x2 = num_kv_heads_x2_per_kv_packing * kv_packing
    assert num_kv_heads_x2 % 2 == 0
    assert actual_num_q_heads % actual_num_kv_heads == 0
    assert head_dim % 128 == 0
    assert get_dtype_packing(kv_cache.dtype) == kv_packing
    assert num_kv_heads_x2 == align_to(actual_num_kv_heads * 2, kv_packing)
    actual_num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads
    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]
    assert num_page_indices % max_num_seqs == 0
    pages_per_seq = num_page_indices // max_num_seqs
    outputs = []

    for i in range(distribution[-1]):
        q_start = cu_q_lens[i]
        q_end = cu_q_lens[i + 1]
        q_len = q_end - q_start

        kv_len = kv_lens[i]
        indices_start = i * pages_per_seq
        indices_end = indices_start + cdiv(kv_len, page_size)
        indices = page_indices[indices_start:indices_end]
        q = queries[q_start:q_end, :, :actual_head_dim]

        # Update the kv cache.
        assert kv_len - q_len >= 0
        gathered_kv = kv_cache[indices]
        gathered_shape = gathered_kv.shape
        gathered_kv = gathered_kv.reshape(-1, *gathered_shape[-3:])
        gathered_kv = gathered_kv.at[kv_len - q_len:kv_len].set(
            merged_kv[q_start:q_end])
        kv_cache = kv_cache.at[indices].set(
            gathered_kv.reshape(gathered_shape))

        kv = gathered_kv.reshape(
            -1, num_kv_heads_x2,
            head_dim)[:, :actual_num_kv_heads * 2, :].reshape(
                -1, actual_num_kv_heads, head_dim * 2)
        k = kv[:kv_len, :, :head_dim][:, :, :actual_head_dim]
        v = kv[:kv_len, :, head_dim:][:, :, :actual_head_dim]
        k = jnp.repeat(k, actual_num_q_heads_per_kv_head, axis=1)
        v = jnp.repeat(v, actual_num_q_heads_per_kv_head, axis=1)

        if q_scale is not None:
            q = q / q_scale
            if jnp.issubdtype(k.dtype, jnp.floating):
                dtype_info = jnp.finfo(k.dtype)
                minval = float(dtype_info.min)
                maxval = float(dtype_info.max)
                q = jnp.clip(q, min=minval, max=maxval)
            q = q.astype(k.dtype)

        attn = jnp.einsum("qhd,khd->hqk",
                          q,
                          k,
                          preferred_element_type=jnp.float32).astype(out_dtype)
        attn *= sm_scale
        if k_scale is not None:
            attn *= k_scale
        if q_scale is not None:
            attn *= q_scale
        if soft_cap is not None:
            attn = soft_cap * jnp.tanh(attn / soft_cap)

        if use_causal_mask:
            q_span = (kv_len - q_len) + jax.lax.broadcasted_iota(
                jnp.int32, attn.shape, 1)
            kv_span = jax.lax.broadcasted_iota(jnp.int32, attn.shape, 2)
            mask = q_span >= kv_span
            if sliding_window is not None:
                mask = jnp.logical_and(mask, q_span < kv_span + sliding_window)
            attn = jnp.where(mask, attn, mask_value)
        attn = jax.nn.softmax(attn, axis=-1).astype(v.dtype)

        out = jnp.einsum("hqk,khd->qhd", attn, v).astype(out_dtype)
        if v_scale is not None:
            out *= v_scale

        outputs.append(out)

    result = jnp.concatenate(outputs, axis=0)
    return result, kv_cache


def ref_ragged_paged_attention_logical(
    queries: jax.
    Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim]
    keys: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    values: jax.
    Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    kv_cache: jax.
    Array,  # [total_num_pages, page_size, num_kv_heads_x2 // kv_packing, kv_packing, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]                 PHYSICAL
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq] PHYSICAL
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]              PHYSICAL
    phys_seq_indices: jax.Array,  # i32[max_num_subseqs]   LOGICAL → PHYS
    q_offsets: jax.Array,  # i32[max_num_subseqs]       chunk position
    distribution: jax.Array,  # i32[3]: (D_end, L_end, M_end)
    *,
    static_q_len: int,  # K_kernel: per-iter q_len
    use_causal_mask: bool = True,
    skip_kv_mask: bool = False,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    out_dtype: Any = None,
    mask_value: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
):
    """Pure-JAX reference for the LOGICAL kernel (decoupled-K PREFILL).

    Iterates over LOGICAL sub-seqs in the L bucket
    (`[distribution[0], distribution[1])`), not over physical seqs.
    Each sub-seq is identified by `(phys_seq_indices[i],
    q_offsets[i])` — multiple sub-seqs may share a single phys
    seq under decoupled-K.

    Implements §6.4 Option A: in-step KV reads are sourced from the
    input `keys`/`values` (via `merge_kv`), NOT from `kv_cache`. The
    cache is read only for genuinely-pre-step content (positions <
    `phys_kv_q_gap`). The cache is still WRITTEN per iter for
    cross-step / cross-pallas_call persistence — only this iter's OWN
    slice. Tests can pre-corrupt `kv_cache` at iter-write offsets to
    verify Option A reads from `keys`/`values` instead.

    See §6.2 / §6.4 / §11.2 in
    `docs/rpa_v3_decoupled_k_kernel_design.md`.
    """
    if out_dtype is None:
        out_dtype = jnp.float32 if queries.dtype == jnp.float32 else jnp.bfloat16
    if mask_value is None:
        mask_value = jnp.finfo(out_dtype).min
    dynamic_validate_inputs(
        queries,
        keys,
        values,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        phys_seq_indices,
        q_offsets,
        use_causal_mask=use_causal_mask,
        skip_kv_mask=skip_kv_mask,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        out_dtype=out_dtype,
        mask_value=mask_value,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )

    actual_head_dim = queries.shape[2]
    actual_num_q_heads = queries.shape[1]
    actual_num_kv_heads = keys.shape[1]
    merged_kv = merge_kv(keys, values)
    assert merged_kv.shape[-3:] == kv_cache.shape[-3:]

    _, page_size, num_kv_heads_x2_per_kv_packing, kv_packing, head_dim = (
        kv_cache.shape)
    num_kv_heads_x2 = num_kv_heads_x2_per_kv_packing * kv_packing
    assert num_kv_heads_x2 % 2 == 0
    assert actual_num_q_heads % actual_num_kv_heads == 0
    assert head_dim % 128 == 0
    assert get_dtype_packing(kv_cache.dtype) == kv_packing
    assert num_kv_heads_x2 == align_to(actual_num_kv_heads * 2, kv_packing)
    actual_num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads
    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]
    assert num_page_indices % max_num_seqs == 0
    pages_per_seq = num_page_indices // max_num_seqs

    # LOGICAL bucket: iters [D_end, L_end). DECODE iters [0, D_end) and
    # MIXED iters [L_end, M_end) are handled by their own pallas_calls
    # (not this reference, which mirrors LOGICAL specifically).
    start_iter = int(distribution[0])
    end_iter = int(distribution[1])

    outputs = []
    for iter_idx in range(start_iter, end_iter):
        qs = _derive_iter_quantities(
            iter_idx, phys_seq_indices, q_offsets, cu_q_lens, kv_lens,
            case=RpaCase.LOGICAL, static_q_len=static_q_len)

        # Gather pages covering [0, kv_len).
        indices_start = qs.phys_seq_idx * pages_per_seq
        indices_end = indices_start + cdiv(qs.kv_len, page_size)
        indices = page_indices[indices_start:indices_end]
        gathered_kv = kv_cache[indices]
        gathered_shape = gathered_kv.shape
        gathered_kv = gathered_kv.reshape(-1, *gathered_shape[-3:])

        # §6.4 Option A: overlay all of phys seq's in-step content from
        # `merged_kv`. Cache state at offsets
        # [phys_kv_q_gap, kv_len) is IGNORED — `merged_kv` is the source
        # of truth for in-step content.
        in_step_count = qs.q_offset + qs.q_len
        kv_view = gathered_kv.at[
            qs.phys_kv_q_gap:qs.phys_kv_q_gap + in_step_count
        ].set(merged_kv[qs.phys_q_start:qs.phys_q_start + in_step_count])

        # Compute attention against [0, kv_len).
        q = queries[qs.q_start:qs.q_end, :, :actual_head_dim]
        kv = kv_view.reshape(
            -1, num_kv_heads_x2,
            head_dim)[:, :actual_num_kv_heads * 2, :].reshape(
                -1, actual_num_kv_heads, head_dim * 2)
        k = kv[:qs.kv_len, :, :head_dim][:, :, :actual_head_dim]
        v = kv[:qs.kv_len, :, head_dim:][:, :, :actual_head_dim]
        k = jnp.repeat(k, actual_num_q_heads_per_kv_head, axis=1)
        v = jnp.repeat(v, actual_num_q_heads_per_kv_head, axis=1)

        if q_scale is not None:
            q = q / q_scale
            if jnp.issubdtype(k.dtype, jnp.floating):
                dtype_info = jnp.finfo(k.dtype)
                minval = float(dtype_info.min)
                maxval = float(dtype_info.max)
                q = jnp.clip(q, min=minval, max=maxval)
            q = q.astype(k.dtype)

        attn = jnp.einsum("qhd,khd->hqk",
                          q,
                          k,
                          preferred_element_type=jnp.float32).astype(out_dtype)
        attn *= sm_scale
        if k_scale is not None:
            attn *= k_scale
        if q_scale is not None:
            attn *= q_scale
        if soft_cap is not None:
            attn = soft_cap * jnp.tanh(attn / soft_cap)

        if use_causal_mask:
            q_span = qs.kv_q_gap + jax.lax.broadcasted_iota(
                jnp.int32, attn.shape, 1)
            kv_span = jax.lax.broadcasted_iota(jnp.int32, attn.shape, 2)
            mask = q_span >= kv_span
            if sliding_window is not None:
                mask = jnp.logical_and(mask, q_span < kv_span + sliding_window)
            attn = jnp.where(mask, attn, mask_value)
        attn = jax.nn.softmax(attn, axis=-1).astype(v.dtype)

        out = jnp.einsum("hqk,khd->qhd", attn, v).astype(out_dtype)
        if v_scale is not None:
            out *= v_scale

        outputs.append(out)

        # §6.4 c.2 — write only iter's OWN slice [kv_q_gap, kv_len) to
        # kv_cache. Earlier sub-seqs wrote their slices in their own iter
        # passes; future sub-seqs will write theirs.
        cache_write_view = gathered_kv.at[qs.kv_q_gap:qs.kv_len].set(
            merged_kv[qs.q_start:qs.q_end])
        kv_cache = kv_cache.at[indices].set(
            cache_write_view.reshape(gathered_shape))

    if outputs:
        result = jnp.concatenate(outputs, axis=0)
    else:
        result = jnp.zeros((0, actual_num_q_heads, actual_head_dim),
                           queries.dtype)
    return result, kv_cache


def get_smem_estimate_bytes(
    max_num_seqs,
    pages_per_seq,
    max_num_subseqs: int | None = None,
):
    """Estimate SMEM bytes used by all scalar prefetches.

    `max_num_subseqs` defaults to `max_num_seqs` (coupled-K identity case
    where each iter is one phys seq). Pass a larger value for LOGICAL
    (decoupled-K) where one phys seq spans multiple sub-seqs.
    """
    if max_num_subseqs is None:
        max_num_subseqs = max_num_seqs
    total_bits = (
        # kv_lens_ref: i32[max_num_seqs]                       PHYSICAL
        align_to(max_num_seqs, 128) * 32 +
        # page_indices_ref: i32[max_num_seqs * pages_per_seq]    PHYSICAL
        align_to(max_num_seqs * pages_per_seq, 128) * 32 +
        # cu_q_lens_ref: i32[max_num_seqs + 1]                 PHYSICAL
        align_to(max_num_seqs + 1, 128) * 32 +
        # phys_seq_indices_ref: i32[max_num_subseqs]           NEW iter→phys map
        align_to(max_num_subseqs, 128) * 32 +
        # q_offsets_ref: i32[max_num_subseqs]                  NEW chunk position
        align_to(max_num_subseqs, 128) * 32 +
        # distribution_ref: i32[3]
        128 * 32 +
        # sem_ids_ref: i32[3]
        128 * 32 +
        # bo_ids_ref: i32[4]
        128 * 32 +
        # bkv_update_ids_ref: i32[6]
        128 * 32)
    return cdiv(total_bits, 8)


def get_vmem_estimate_bytes(
    actual_num_kv_heads,
    actual_num_q_heads_per_kv_head,
    actual_head_dim,
    bq_sz,
    bkv_sz,
    q_dtype,
    kv_dtype,
):
    q_packing = get_dtype_packing(q_dtype)
    kv_packing = get_dtype_packing(kv_dtype)
    num_q_heads_per_kv_head = align_to(actual_num_q_heads_per_kv_head,
                                       q_packing)
    bkv_stride = cdiv(actual_num_kv_heads * 2, kv_packing)
    if has_bank_conflicts(bkv_stride):
        bkv_stride += 1
    head_dim = align_to(actual_head_dim, 128)

    total_bits = (
        # bkv_x2_ref
        (2 * bkv_sz * bkv_stride * kv_packing * head_dim) *
        (32 // kv_packing) +
        # bq_x2_ref + bo_x2_ref
        2 * (2 * actual_num_kv_heads * bq_sz * num_q_heads_per_kv_head *
             head_dim) * (32 // q_packing) +
        # l_ref + m_ref
        2 *
        (actual_num_kv_heads * bq_sz * num_q_heads_per_kv_head * 128) * 32 +
        # acc_ref
        (actual_num_kv_heads * bq_sz * num_q_heads_per_kv_head * head_dim) *
        32)
    return cdiv(total_bits, 8)


def get_kv_cache_shape(
    total_num_pages,
    page_size,
    actual_num_kv_heads,
    actual_head_dim,
    kv_dtype,
):
    kv_packing = get_dtype_packing(kv_dtype)
    return (
        total_num_pages,
        page_size,
        align_to(actual_num_kv_heads * 2, kv_packing) // kv_packing,
        kv_packing,
        align_to(actual_head_dim, 128),
    )


def _ragged_paged_attention_kernel(*args, **kwargs):
    # `args` is the scalar_prefetches tuple in the order constructed by
    # `run_rpa_kernel`. distribution_ref is at index 5 (after the 3
    # phys-keyed prefetches and the 2 LOGICAL iter-keyed prefetches).
    # If the prefetch tuple is reordered, this index must move in lockstep.
    distribution_ref = args[5]
    start_seq_idx, end_seq_idx = kwargs["case"].get_range(distribution_ref)

    @pl.loop(start_seq_idx, end_seq_idx)
    def _(seq_idx):
        return _ragged_paged_attention_kernel_loop(
            seq_idx,
            *args,
            **kwargs,
        )


def _ragged_paged_attention_kernel_loop(
    seq_idx,
    # Prefetch
    kv_lens_ref,  # [max_num_seqs]                       PHYSICAL
    page_indices_ref,  # [max_num_seqs * pages_per_seq]    PHYSICAL
    cu_q_lens_ref,  # [max_num_seqs + 1]                   PHYSICAL
    # Iter→phys indirection. Coupled-K (D/P/M) callers pass identity
    # (`arange(max_num_seqs)`, `zeros(max_num_seqs)`); LOGICAL callers
    # pass real values. The kernel reads these uniformly without case
    # branching — identity prefetches make the formulas reduce
    # numerically to today's behavior. See §6.2 in
    # docs/rpa_v3_decoupled_k_kernel_design.md.
    phys_seq_indices_ref,  # [max_num_subseqs]      LOGICAL → PHYSICAL
    q_offsets_ref,  # [max_num_subseqs]          LOGICAL chunk position
    # TODO(jevinjiang): merge these into one so we can save SMEM.
    distribution_ref,  # [3] (decode_end, prefill_end, mixed_end)
    sem_ids_ref,  # [3] (bq_sem_idx, bkv_sem_idx, bo_sem_idx)
    bo_ids_ref,  # [4] (bo_sem_0_seq_idx, bo_sem_1_seq_idx, bo_sem_0_bo_idx, bo_sem_1_bo_idx)
    bkv_update_ids_ref,  # [6] (bkv_sem_0_seq_idx, bkv_sem_1_seq_idx, bkv_sem_0_offset, bkv_sem_1_offset, bkv_sem_0_sz, bkv_sem_1_sz)
    # Input
    q_hbm_ref,  # [actual_num_kv_heads, max_num_tokens, num_q_heads_per_kv_head // q_packing, q_packing, head_dim]
    kv_hbm_ref,  # [max_num_tokens, num_kv_heads_x2 // kv_packing, kv_packing, head_dim]
    kv_cache_hbm_ref,  # [total_num_pages, page_size, num_kv_heads_x2 // kv_packing, kv_packing, head_dim]
    # Output
    o_hbm_ref,  # [actual_num_kv_heads, max_num_tokens, num_q_heads_per_kv_head // q_packing, q_packing, head_dim]
    updated_kv_cache_hbm_ref,  # [total_num_pages, page_size, num_kv_heads_x2 // kv_packing, kv_packing, head_dim]
    # Scratch
    ## Add one extra to handle bank conflicts for strided load if needed.
    bkv_x2_ref,  # [2, bkv_sz, num_kv_heads_x2 // kv_packing (+ 1), kv_packing, head_dim]
    bq_x2_ref,  # [2, actual_num_kv_heads, bq_sz, num_q_heads_per_kv_head // q_packing, q_packing, head_dim]
    bo_x2_ref,  # [2, actual_num_kv_heads, bq_sz, num_q_heads_per_kv_head // q_packing, q_packing, head_dim]
    sems,  # [4, 2]
    l_ref,  # [actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, 128],
    m_ref,  # [actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, 128],
    acc_ref,  # [actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, head_dim],
    *,
    use_causal_mask: bool = True,
    skip_kv_mask: bool = False,
    sm_scale: float,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    static_q_len: int | None = None,
    bq_sz,  # bq fetch size
    bkv_sz,  # bkv prefetch size
    bq_csz,  # bq compute size
    bkv_csz,  # bkv compute size
    case: RpaCase = RpaCase.MIXED,
    debug_mode: bool = False,
):
    assert q_hbm_ref.shape == o_hbm_ref.shape
    assert q_hbm_ref.shape[-1] == kv_cache_hbm_ref.shape[-1]

    if case == RpaCase.DECODE:
        use_causal_mask = False

    out_dtype = acc_ref.dtype
    (
        actual_num_kv_heads,
        max_num_tokens,
        num_q_heads_per_kv_head_per_packing,
        q_packing,
        head_dim,
    ) = q_hbm_ref.shape
    (
        total_num_pages,
        page_size,
        num_kv_heads_x2_per_kv_packing,
        kv_packing,
        _,
    ) = kv_cache_hbm_ref.shape
    bkv_stride = bkv_x2_ref.shape[2]
    assert bkv_stride in (
        num_kv_heads_x2_per_kv_packing,
        num_kv_heads_x2_per_kv_packing + 1,
    )
    max_num_seqs = kv_lens_ref.shape[0]
    num_page_indices = page_indices_ref.shape[0]
    assert num_page_indices % max_num_seqs == 0
    pages_per_seq = num_page_indices // max_num_seqs
    # num_kv_heads_x2 = num_kv_heads_x2_per_kv_packing * kv_packing
    num_q_heads_per_kv_head = num_q_heads_per_kv_head_per_packing * q_packing
    q_dtype = q_hbm_ref.dtype
    kv_dtype = kv_cache_hbm_ref.dtype
    assert o_hbm_ref.dtype == q_dtype
    assert get_dtype_packing(q_dtype) == q_packing
    assert get_dtype_packing(kv_dtype) == kv_packing
    assert head_dim % 128 == 0
    assert bkv_sz % page_size == 0
    bkv_p = bkv_sz // page_size
    start_seq_idx, end_seq_idx = case.get_range(distribution_ref)
    num_seqs = end_seq_idx - start_seq_idx

    # §6.2 derivation: iter_idx → phys_seq_idx + chunk position. Single
    # branch — coupled-K (D/P/M) sees identity prefetches so phys_seq_idx
    # = seq_idx and q_offset = 0 (numerically), reducing the formulas to
    # today's behavior. LOGICAL sees non-identity prefetches and gets
    # genuine indirection.
    # Single source of truth for §6.2 derivation; see _derive_iter_quantities.
    # Unpacked into bare locals so downstream kernel body uses today's names.
    (phys_seq_idx, q_offset, phys_q_start, phys_q_end, phys_q_len,
     phys_kv_len, phys_kv_q_gap, q_start, q_end, q_len, kv_q_gap,
     kv_len) = _derive_iter_quantities(
         seq_idx, phys_seq_indices_ref, q_offsets_ref,
         cu_q_lens_ref, kv_lens_ref, case, static_q_len)
    cur_seq_start_bkv_idx = 0
    next_seq_start_bkv_idx = 0

    if sliding_window is not None:
        # TODO(jevinjiang): can skip by page_size instead of bkv_sz.
        cur_seq_start_bkv_idx = jnp.maximum(kv_q_gap - sliding_window,
                                            0) // bkv_sz
        next_seq_idx = jnp.minimum(seq_idx + 1, end_seq_idx - 1)
        next_qs = _derive_iter_quantities(
            next_seq_idx, phys_seq_indices_ref, q_offsets_ref,
            cu_q_lens_ref, kv_lens_ref, case, static_q_len)
        next_seq_start_bkv_idx = (
            jnp.maximum(next_qs.kv_q_gap - sliding_window, 0) // bkv_sz)

    def debug_print(msg, *args):
        if debug_mode:
            pl.debug_print(msg, *args)

    debug_print("[RPA debug] ======= In loop seq_idx={}", seq_idx)
    debug_print("[RPA debug] start_seq_idx={}", start_seq_idx)
    debug_print("[RPA debug] end_seq_idx={}", end_seq_idx)
    debug_print("[RPA debug] num_seqs={}", num_seqs)
    debug_print("[RPA debug] bkv_p={}", bkv_p)
    debug_print("[RPA debug] page_size={}", page_size)
    debug_print("[RPA debug] pages_per_seq={}", pages_per_seq)
    debug_print("[RPA debug] bkv_sz={}", bkv_sz)
    debug_print("[RPA debug] bq_sz={}", bq_sz)
    debug_print(f"[RPA debug] static_q_len={static_q_len}")
    debug_print("[RPA debug] q_start={}", q_start)
    debug_print("[RPA debug] q_end={}", q_end)
    debug_print("[RPA debug] q_len={}", q_len)
    debug_print("[RPA debug] kv_len={}", kv_len)
    debug_print("[RPA debug] kv_q_gap={}", kv_q_gap)
    debug_print(f"[RPA debug] sliding_window={sliding_window}")
    debug_print("[RPA debug] cur_seq_start_bkv_idx={}", cur_seq_start_bkv_idx)
    debug_print("[RPA debug] next_seq_start_bkv_idx={}",
                next_seq_start_bkv_idx)

    def flash_attention_step1_qk_softmax(
        q,  # [actual_bq_csz * num_q_heads_per_kv_head, head_dim]
        k,  # [bkv_csz, head_dim]
        v,  # [bkv_csz, head_dim]
        l_ref,  # [actual_bq_csz * num_q_heads_per_kv_head, 128]
        m_ref,  # [actual_bq_csz * num_q_heads_per_kv_head, 128]
        *,
        processed_q_len,
        processed_kv_len,
        effective_kv_len,
    ):
        assert len(q.shape) == 2
        assert q.shape[0] % num_q_heads_per_kv_head == 0
        assert q.shape[1] == head_dim
        actual_bq_csz = q.shape[0] // num_q_heads_per_kv_head
        assert k.shape == (bkv_csz, head_dim)
        assert v.shape == (bkv_csz, head_dim)
        assert l_ref.shape == (actual_bq_csz * num_q_heads_per_kv_head, 128)
        assert m_ref.shape == (actual_bq_csz * num_q_heads_per_kv_head, 128)
        assert k.dtype == v.dtype

        # Follow FlashAttention-2 forward pass.
        if q_scale is not None:
            q = q / q_scale
            if jnp.issubdtype(k.dtype, jnp.floating):
                dtype_info = jnp.finfo(k.dtype)
                minval = float(dtype_info.min)
                maxval = float(dtype_info.max)
                q = jnp.clip(q, min=minval, max=maxval)
            q = q.astype(k.dtype)

        s = jnp.matmul(q, k.T, preferred_element_type=jnp.float32)

        s_scale = sm_scale
        if k_scale is not None:
            s_scale *= k_scale
        if q_scale is not None:
            s_scale *= q_scale

        s *= s_scale

        if soft_cap is not None:
            s = soft_cap * jnp.tanh(s / soft_cap)

        int_ty = jnp.int32
        if get_dtype_packing(q_dtype) != 1 and get_tpu_version() >= 6:
            int_ty = jnp.int16
        processed_q_len_int = processed_q_len.astype(int_ty)
        processed_kv_len_int = processed_kv_len.astype(int_ty)
        effective_kv_len_int = effective_kv_len.astype(int_ty)
        q_span = processed_q_len_int + (lax.broadcasted_iota(
            jnp.int32, s.shape, 0) // num_q_heads_per_kv_head).astype(int_ty)
        k_span = processed_kv_len_int + lax.broadcasted_iota(
            int_ty, s.shape, 1)
        v_span = processed_kv_len_int + lax.broadcasted_iota(
            int_ty, v.shape, 0)

        mask = None
        if use_causal_mask:
            assert not skip_kv_mask
            mask = mask_and(mask, q_span >= k_span)

        if not skip_kv_mask:
            mask = mask_and(mask, k_span < effective_kv_len_int)
            v = jnp.where(v_span < effective_kv_len_int, v, 0.0)

        if sliding_window is not None:
            mask = mask_and(mask, q_span < k_span + sliding_window)

        if mask is not None:
            s = jnp.where(mask, s, mask_value)

        s_rowmax = jnp.max(s, axis=1, keepdims=True)

        # if converting the type too early, there will be accuracy issue.
        s_rowmax = s_rowmax.astype(out_dtype)
        m_prev = m_ref[...]
        m_curr = jnp.maximum(m_prev, s_rowmax)
        m_ref[...] = m_curr
        p = jnp.exp(s - broadcast_minor(m_curr, s.shape))

        p_rowsum = jnp.sum(p, axis=1, keepdims=True, dtype=out_dtype)
        exp_m_diff = jnp.exp(m_prev - m_curr)
        l_prev = l_ref[...]
        l_ref[...] = exp_m_diff * l_prev + p_rowsum

        return p, v, exp_m_diff

    def flash_attention_step2_pv(
            p,  # [actual_bq_csz * num_q_heads_per_kv_head, bkv_csz]
            v,  # [bkv_csz, head_dim]
            exp_m_diff,  # [actual_bq_csz * num_q_heads_per_kv_head, 128]
            o_ref,  # [actual_bq_csz * num_q_heads_per_kv_head, head_dim]
    ):
        assert len(p.shape) == 2
        assert p.shape[0] % num_q_heads_per_kv_head == 0
        assert p.shape[1] == bkv_csz
        actual_bq_csz = p.shape[0] // num_q_heads_per_kv_head
        assert v.shape == (bkv_csz, head_dim)
        assert exp_m_diff.shape == (actual_bq_csz * num_q_heads_per_kv_head,
                                    128)
        assert o_ref.shape == (actual_bq_csz * num_q_heads_per_kv_head,
                               head_dim)
        pv = jnp.matmul(p, v, preferred_element_type=jnp.float32)

        if v_scale is not None:
            pv *= v_scale
        # if converting the type too early, there will be accuracy issue.
        pv = pv.astype(out_dtype)
        o_prev = o_ref[...]
        o_ref[...] = broadcast_minor(exp_m_diff, o_prev.shape) * o_prev + pv

    def _async_copy(src, dst, sem, wait):
        if debug_mode:
            # Skip DMA if debug mode is enabled.
            return
        cp = pltpu.make_async_copy(src, dst, sem)
        if wait:
            cp.wait()
        else:
            cp.start()

    def _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, *, wait=False):
        sem = sems.at[0, bkv_sem_idx]
        vmem_ref = bkv_x2_ref.at[
            bkv_sem_idx, :, :num_kv_heads_x2_per_kv_packing]

        cache_hbm_shape = kv_cache_hbm_ref.shape
        cache_hbm_ref = kv_cache_hbm_ref.reshape(
            cache_hbm_shape[0] * cache_hbm_shape[1], *cache_hbm_shape[2:])
        kv_len_start = bkv_idx * bkv_sz
        kv_p_start = bkv_idx * bkv_p

        # §6.2 derivation via shared helper. See _derive_iter_quantities.
        (phys_seq_idx, q_offset, phys_q_start, phys_q_end, phys_q_len,
         phys_kv_len, phys_kv_q_gap, q_start, q_end, q_len, kv_q_gap,
         kv_len) = _derive_iter_quantities(
             seq_idx, phys_seq_indices_ref, q_offsets_ref,
             cu_q_lens_ref, kv_lens_ref, case, static_q_len)

        kv_left = kv_len - kv_len_start

        # §6.4 Option A — READ split: in-step content (this phys seq's
        # tokens this step, cumulative across iters) is sourced from
        # kv_hbm_ref, NOT kv_cache_hbm_ref. Eliminates same-pallas_call
        # cross-iter read-after-write race on kv_cache_hbm_ref. In
        # coupled-K (q_offset=0), reduces to today's
        # max(kv_left - q_len, 0). See §6.4 in the design doc.
        kv_left_frm_cache_read = jnp.maximum(kv_left - (q_offset + q_len), 0)
        kv_left_frm_new_read = kv_left - kv_left_frm_cache_read
        bkv_sz_frm_cache_read = jnp.minimum(kv_left_frm_cache_read, bkv_sz)
        bkv_sz_frm_new_read = jnp.minimum(bkv_sz - bkv_sz_frm_cache_read,
                                          kv_left_frm_new_read)

        # §6.4 c.2 — WRITE extent: only iter's OWN chunk goes back to
        # kv_cache_hbm_ref via _update_kv_cache. Earlier sub-seqs wrote
        # their slices in their own iter passes; future sub-seqs will
        # write theirs. Preserves O(N) total writes (vs O(N^2) if we
        # wrote the cumulative range). In coupled-K (q_offset=0 SMEM
        # constant), the _read and _write expressions reduce to the
        # same value — XLA's CSE / constant-folding should collapse
        # them in the lowered HLO. Verify in Phase 2c via
        # --xla_mosaic_dump_to and the lowered IR diff vs today.
        kv_left_frm_cache_write = jnp.maximum(kv_left - q_len, 0)
        kv_left_frm_new_write = kv_left - kv_left_frm_cache_write
        bkv_sz_frm_cache_write = jnp.minimum(kv_left_frm_cache_write, bkv_sz)
        bkv_sz_frm_new_write = jnp.minimum(bkv_sz - bkv_sz_frm_cache_write,
                                           kv_left_frm_new_write)

        page_indices_offset = phys_seq_idx * pages_per_seq + kv_p_start

        debug_print(
            "[RPA debug]"
            f" -----------{'wait' if wait else 'start'}_fetch_bkv-----------")
        debug_print("[RPA debug] seq_idx={}", seq_idx)
        debug_print("[RPA debug] bkv_idx={}", bkv_idx)
        debug_print("[RPA debug] bkv_sem_idx={}", bkv_sem_idx)
        debug_print("[RPA debug] kv_len_start={}", kv_len_start)
        debug_print("[RPA debug] kv_p_start={}", kv_p_start)
        debug_print("[RPA debug] kv_left={}", kv_left)
        debug_print("[RPA debug] kv_left_frm_cache_read={}",
                    kv_left_frm_cache_read)
        debug_print("[RPA debug] kv_left_frm_new_read={}", kv_left_frm_new_read)
        debug_print("[RPA debug] bkv_sz_frm_cache_read={}",
                    bkv_sz_frm_cache_read)
        debug_print("[RPA debug] bkv_sz_frm_new_read={}", bkv_sz_frm_new_read)
        debug_print("[RPA debug] bkv_sz_frm_cache_write={}",
                    bkv_sz_frm_cache_write)
        debug_print("[RPA debug] bkv_sz_frm_new_write={}", bkv_sz_frm_new_write)
        debug_print("[RPA debug] page_indices_offset={}", page_indices_offset)

        if not wait:
            # Make sure the current bkv buffer is safe to overwrite.
            wait_update_kv_cache(bkv_sem_idx)

            # Fetch effective kv from kv cache. To pipeline multiple DMA calls, we
            # utilize static for loop instead of dynamic for loop. Use the READ
            # split: under LOGICAL this is a smaller cache portion (in-step
            # content goes through kv_hbm_ref instead); under D/P/M (q_offset=0
            # numerically), reduces to today's split.
            for i in range(bkv_p):
                # Ensure only effective kvs are copied.
                sz = jnp.clip(kv_left_frm_cache_read - i * page_size, 0,
                              page_size)
                # If the page index is out of bound, we set page_idx to the last page.
                # And there will be no copy since sz will be 0.
                page_idx = jnp.minimum(page_indices_offset + i,
                                       num_page_indices - 1)
                _async_copy(
                    cache_hbm_ref.at[pl.ds(
                        page_indices_ref[page_idx] * page_size, sz)],
                    vmem_ref.at[pl.ds(i * page_size, sz)],
                    sem,
                    wait=False,
                )
                debug_print("[RPA debug] loop_body i={}, sz={}", i, sz)

            new_kv_len_start = q_end - kv_left_frm_new_read
            debug_print("[RPA debug] new_kv_len_start={}", new_kv_len_start)
            _async_copy(
                kv_hbm_ref.at[pl.ds(new_kv_len_start, bkv_sz_frm_new_read)],
                vmem_ref.at[pl.ds(bkv_sz_frm_cache_read, bkv_sz_frm_new_read)],
                sem,
                wait,
            )
        else:
            dst = vmem_ref.at[pl.ds(
                0, bkv_sz_frm_cache_read + bkv_sz_frm_new_read)]
            _async_copy(
                src=dst,
                dst=dst,
                sem=sem,
                wait=True,
            )
        # Return the WRITE extent — only iter's OWN chunk gets persisted to
        # kv_cache_hbm_ref via _update_kv_cache. In coupled-K, _write equals
        # _read; in LOGICAL, _write is narrower (no redundant cross-iter
        # writeback).
        return kv_len_start + bkv_sz_frm_cache_write, bkv_sz_frm_new_write

    def _update_kv_cache(seq_idx,
                         bkv_sem_idx,
                         offset,
                         update_sz,
                         *,
                         wait=False):
        sem = sems.at[3, bkv_sem_idx]
        vmem_ref = bkv_x2_ref.at[
            bkv_sem_idx, :, :num_kv_heads_x2_per_kv_packing]
        bkv_id = offset // bkv_sz
        kv_p_start = offset // page_size
        kv_p_end = cdiv(offset + update_sz, page_size)
        ignore = offset % page_size
        p_ignore = kv_p_start - bkv_id * bkv_p
        # Single-branch iter→phys mapping. In coupled-K identity prefetches
        # make phys_seq_idx numerically equal to seq_idx.
        phys_seq_idx = phys_seq_indices_ref[seq_idx]
        page_indices_offset = phys_seq_idx * pages_per_seq + kv_p_start

        cache_hbm_shape = updated_kv_cache_hbm_ref.shape
        cache_hbm_ref = updated_kv_cache_hbm_ref.reshape(
            cache_hbm_shape[0] * cache_hbm_shape[1], *cache_hbm_shape[2:])

        debug_print(
            "[RPA debug]"
            f" -----------{'wait' if wait else 'start'}_update_kv_cache-----------"
        )
        debug_print("[RPA debug] seq_idx={}", seq_idx)
        debug_print("[RPA debug] bkv_sem_idx={}", bkv_sem_idx)
        debug_print("[RPA debug] offset={}", offset)
        debug_print("[RPA debug] update_sz={}", update_sz)
        debug_print("[RPA debug] bkv_id={}", bkv_id)
        debug_print("[RPA debug] kv_p_start={}", kv_p_start)
        debug_print("[RPA debug] kv_p_end={}", kv_p_end)
        debug_print("[RPA debug] ignore={}", ignore)
        debug_print("[RPA debug] p_ignore={}", p_ignore)
        debug_print("[RPA debug] page_indices_offset={}", page_indices_offset)

        def loop_body(i, states):
            update_sz, ignore = states
            sz = jnp.minimum(page_size - ignore, update_sz)

            _async_copy(
                vmem_ref.at[pl.ds((p_ignore + i) * page_size + ignore, sz)],
                cache_hbm_ref.at[pl.ds(
                    page_indices_ref[page_indices_offset + i] * page_size +
                    ignore,
                    sz,
                )],
                sem,
                wait,
            )
            debug_print("[RPA debug] loop_body i={}, sz={}", i, sz)
            return update_sz - sz, 0

        if not wait:
            lax.fori_loop(
                0,
                kv_p_end - kv_p_start,
                loop_body,
                (update_sz, ignore),  # total transfer size
                unroll=False,
            )
        else:
            dst = cache_hbm_ref.at[pl.ds(0, update_sz)]
            _async_copy(
                src=dst,
                dst=dst,
                sem=sem,
                wait=True,
            )

    def _fetch_bq(seq_idx, bq_idx, bq_sem_idx, *, wait=False):
        sem = sems.at[1, bq_sem_idx]
        vmem_ref = bq_x2_ref.at[bq_sem_idx]
        q_len_start = cu_q_lens_ref[seq_idx] + bq_idx * bq_sz
        q_end = cu_q_lens_ref[seq_idx + 1]
        sz = jnp.minimum(bq_sz, q_end - q_len_start)

        debug_print(
            "[RPA debug]"
            f" -----------{'wait' if wait else 'start'}_fetch_bq-----------")
        debug_print("[RPA debug] seq_idx={}", seq_idx)
        debug_print("[RPA debug] bq_idx={}", bq_idx)
        debug_print("[RPA debug] bq_sem_idx={}", bq_sem_idx)
        debug_print("[RPA debug] q_len_start={}", q_len_start)
        debug_print("[RPA debug] q_end={}", q_end)
        debug_print("[RPA debug] sz={}", sz)

        _async_copy(
            q_hbm_ref.at[:, pl.ds(q_len_start, sz)],
            vmem_ref.at[:, pl.ds(0, sz)],
            sem,
            wait,
        )

    def _send_bo(seq_idx, bo_idx, bo_sem_idx, *, wait=False):
        sem = sems.at[2, bo_sem_idx]
        vmem_ref = bo_x2_ref.at[bo_sem_idx]
        q_len_start = cu_q_lens_ref[seq_idx] + bo_idx * bq_sz
        q_end = cu_q_lens_ref[seq_idx + 1]
        sz = jnp.minimum(bq_sz, q_end - q_len_start)

        debug_print(
            "[RPA debug]"
            f" -----------{'wait' if wait else 'start'}_send_bo-----------")
        debug_print("[RPA debug] seq_idx={}", seq_idx)
        debug_print("[RPA debug] bo_idx={}", bo_idx)
        debug_print("[RPA debug] bo_sem_idx={}", bo_sem_idx)
        debug_print("[RPA debug] q_len_start={}", q_len_start)
        debug_print("[RPA debug] q_end={}", q_end)
        debug_print("[RPA debug] sz={}", sz)

        _async_copy(
            vmem_ref.at[:, pl.ds(0, sz)],
            o_hbm_ref.at[:, pl.ds(q_len_start, sz)],
            sem,
            wait,
        )

    def start_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx):
        return _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx)

    def wait_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx):
        return _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, wait=True)

    def start_fetch_bq(seq_idx, bq_idx, bq_sem_idx):
        return _fetch_bq(seq_idx, bq_idx, bq_sem_idx)

    def wait_fetch_bq(seq_idx, bq_idx, bq_sem_idx):
        return _fetch_bq(seq_idx, bq_idx, bq_sem_idx, wait=True)

    def start_send_bo(seq_idx, bo_idx, bo_sem_idx):
        bo_ids_ref[bo_sem_idx] = seq_idx
        bo_ids_ref[bo_sem_idx + 2] = bo_idx
        _send_bo(seq_idx, bo_idx, bo_sem_idx)

    def wait_send_bo(bo_sem_idx):
        old_seq_idx = bo_ids_ref[bo_sem_idx]
        old_bo_idx = bo_ids_ref[bo_sem_idx + 2]

        @pl.when(
            jnp.logical_and(start_seq_idx <= old_seq_idx, old_seq_idx
                            <= seq_idx))
        def _():
            _send_bo(old_seq_idx, old_bo_idx, bo_sem_idx, wait=True)

    def start_update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz):
        bkv_update_ids_ref[bkv_sem_idx] = seq_idx
        bkv_update_ids_ref[bkv_sem_idx + 2] = offset
        bkv_update_ids_ref[bkv_sem_idx + 4] = update_sz
        _update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz)

    def wait_update_kv_cache(bkv_sem_idx):
        update_sz = bkv_update_ids_ref[bkv_sem_idx + 4]

        @pl.when(update_sz > 0)
        def _():
            seq_idx = bkv_update_ids_ref[bkv_sem_idx]
            offset = bkv_update_ids_ref[bkv_sem_idx + 2]
            bkv_update_ids_ref[bkv_sem_idx + 4] = 0
            _update_kv_cache(seq_idx,
                             bkv_sem_idx,
                             offset,
                             update_sz,
                             wait=True)

    def strided_load(ref, start, sz, step, *, dtype=None):
        assert get_dtype_packing(ref.dtype) == 1
        assert len(ref.shape) == 2
        r, l = ref.shape  # noqa
        assert l % 128 == 0
        folds = l // 128
        ref = ref.reshape(r * folds, 128)
        start *= folds
        sz *= folds
        step *= folds
        assert sz % step == 0
        vec = jnp.concat(
            [ref[pl.ds(start + i, sz // step, step)] for i in range(folds)],
            axis=1)
        if dtype is not None:
            vec = pltpu.bitcast(vec, dtype)
        return vec

    def strided_store(ref, start, sz, step, val):
        assert get_dtype_packing(ref.dtype) == 1
        assert ref.dtype == val.dtype
        assert ref.shape == val.shape
        assert len(ref.shape) == 2
        r, l = ref.shape  # noqa
        assert l % 128 == 0
        folds = l // 128
        ref = ref.reshape(r * folds, 128)
        start *= folds
        sz *= folds
        step *= folds
        assert sz % step == 0
        for i in range(folds):
            ref[pl.ds(start + i, sz // step,
                      step)] = val[:, i * 128:(i + 1) * 128]

    def load_bq(bq_sem_idx, kv_head_idx, start, sz):
        q_ref = (bq_x2_ref.bitcast(
            jnp.uint32).at[bq_sem_idx, kv_head_idx].reshape(
                bq_sz * num_q_heads_per_kv_head_per_packing, head_dim))
        start *= num_q_heads_per_kv_head_per_packing
        sz *= num_q_heads_per_kv_head_per_packing
        return strided_load(q_ref, start, sz, 1, dtype=q_dtype)

    def load_bkv(bkv_sem_idx, kv_head_idx, start, sz):
        start *= bkv_stride
        sz *= bkv_stride
        step = bkv_stride
        kv_ref = (bkv_x2_ref.bitcast(jnp.uint32).at[bkv_sem_idx].reshape(
            bkv_sz * step, head_dim))

        if kv_packing == 1:
            start += kv_head_idx * 2
            k = strided_load(kv_ref, start, sz, step, dtype=kv_dtype)
            v = strided_load(kv_ref, start + 1, sz, step, dtype=kv_dtype)
            k = pltpu.bitcast(k, kv_dtype)
            v = pltpu.bitcast(v, kv_dtype)
            return k, v

        num_kv_per_load = kv_packing // 2
        offset = kv_head_idx // num_kv_per_load
        kv_idx_in_load = kv_head_idx % num_kv_per_load
        kv = strided_load(kv_ref, start + offset, sz, step)
        bitwidth = 32 // kv_packing
        repack_ty = jnp.dtype(f"uint{bitwidth}")
        k = kv >> (kv_idx_in_load * 2 * bitwidth)
        v = k >> bitwidth
        k = pltpu.bitcast(k.astype(repack_ty), kv_dtype)
        v = pltpu.bitcast(v.astype(repack_ty), kv_dtype)
        return k, v

    def broadcast_minor(src, shape):
        if src.shape == shape:
            return src
        assert src.shape[:-1] == shape[:-1]
        assert src.shape[-1] % 128 == 0
        target_minor = align_to(shape[-1], src.shape[-1])
        # no-op concatenation.
        return jnp.concatenate(
            [src for _ in range(target_minor // src.shape[-1])],
            axis=-1)[..., :shape[-1]]

    def mask_and(mask, new_mask):
        if mask is None:
            return new_mask
        return jnp.logical_and(mask, new_mask)

    def process(static_q_len=None):
        if static_q_len is None:
            actual_bq_sz = bq_sz
            num_bq = cdiv(q_len, actual_bq_sz)
        else:
            actual_bq_sz = min(bq_sz, static_q_len)
            num_bq = cdiv(static_q_len, actual_bq_sz)

        actual_bq_csz = min(bq_csz, actual_bq_sz)

        def get_next_bq_ids(seq_idx, bq_idx, bq_sem_idx):
            next_bq_idx = bq_idx + 1
            is_last_bq = next_bq_idx == num_bq
            next_bq_idx = lax.select(is_last_bq, 0, next_bq_idx)
            next_seq_idx = lax.select(is_last_bq, seq_idx + 1, seq_idx)
            next_bq_sem_idx = lax.select(bq_sem_idx == 0, 1, 0)
            return next_seq_idx, next_bq_idx, next_bq_sem_idx

        def get_next_bkv_ids(seq_idx, bq_idx, bkv_idx, bkv_sem_idx, *,
                             num_bkv):
            next_bkv_idx = bkv_idx + 1
            is_last_bkv = next_bkv_idx == num_bkv
            next_bq_idx = lax.select(is_last_bkv, bq_idx + 1, bq_idx)
            is_last_bq = next_bq_idx == num_bq
            next_bq_idx = lax.select(is_last_bq, 0, next_bq_idx)
            next_seq_idx = lax.select(is_last_bq, seq_idx + 1, seq_idx)
            next_bkv_sem_idx = lax.select(bkv_sem_idx == 0, 1, 0)

            next_bq_start_bkv_idx = 0
            if sliding_window is not None:
                next_bq_start_bkv_idx = (jnp.maximum(
                    kv_q_gap +
                    (bq_idx + 1) * actual_bq_sz - sliding_window, 0) // bkv_sz)
            next_bkv_idx = lax.select(is_last_bkv, next_bq_start_bkv_idx,
                                      next_bkv_idx)
            next_bkv_idx = lax.select(is_last_bq, next_seq_start_bkv_idx,
                                      next_bkv_idx)
            return next_seq_idx, next_bq_idx, next_bkv_idx, next_bkv_sem_idx

        @pl.loop(0, num_bq, unroll=False)
        def compute_with_bq(bq_idx):
            # Re-initialize l, m, acc to 0 before bkv loop.
            l_ref[...] = jnp.full_like(l_ref, 0.0)
            m_ref[...] = jnp.full_like(m_ref, -jnp.inf)
            acc_ref[...] = jnp.full_like(acc_ref, 0.0)

            bq_sem_idx = sem_ids_ref[0]
            next_seq_idx, next_bq_idx, next_bq_sem_idx = get_next_bq_ids(
                seq_idx, bq_idx, bq_sem_idx)

            processed_q_len = kv_q_gap + bq_idx * actual_bq_sz
            start_bkv_idx = 0
            if sliding_window is not None:
                # Recalculate the start_bkv_idx based on the processed_q_len.
                start_bkv_idx = (
                    jnp.maximum(processed_q_len - sliding_window, 0) // bkv_sz)
            if use_causal_mask:
                effective_kv_len = jnp.minimum(kv_len,
                                               processed_q_len + actual_bq_sz)
            else:
                effective_kv_len = kv_len
            end_bkv_idx = cdiv(effective_kv_len, bkv_sz)

            # Prefetch next bq
            @pl.when(next_seq_idx < end_seq_idx)
            def prefetch_next_bq():
                sem_ids_ref[0] = next_bq_sem_idx
                start_fetch_bq(next_seq_idx, next_bq_idx, next_bq_sem_idx)

            @pl.loop(start_bkv_idx, end_bkv_idx, unroll=False)
            def compute_with_bkv(bkv_idx):
                assert bkv_sz % kv_packing == 0

                # Get next bkv ids.
                bkv_sem_idx = sem_ids_ref[1]
                next_seq_idx, _, next_bkv_idx, next_bkv_sem_idx = get_next_bkv_ids(
                    seq_idx, bq_idx, bkv_idx, bkv_sem_idx, num_bkv=end_bkv_idx)
                processed_kv_len = bkv_idx * bkv_sz

                # Prefetch next bkv
                @pl.when(next_seq_idx < end_seq_idx)
                def prefetch_next_bkv():
                    sem_ids_ref[1] = next_bkv_sem_idx
                    start_fetch_bkv(next_seq_idx, next_bkv_idx,
                                    next_bkv_sem_idx)

                # Wait for cur bq if not ready yet
                @pl.when(bkv_idx == start_bkv_idx)
                def wait_cur_bq():
                    wait_fetch_bq(seq_idx, bq_idx, bq_sem_idx)

                # Wait for cur bkv
                offset, update_sz = wait_fetch_bkv(seq_idx, bkv_idx,
                                                   bkv_sem_idx)

                # Start updating bkv to kv cache if applicable.
                # Only needed in last bq loop.
                @pl.when(jnp.logical_and(update_sz > 0, bq_idx == num_bq - 1))
                def update_cur_bkv_to_cache():
                    start_update_kv_cache(seq_idx, bkv_sem_idx, offset,
                                          update_sz)

                debug_print(
                    "[RPA debug] -----------flash attention-----------")
                debug_print("[RPA debug] seq_idx={}", seq_idx)
                debug_print("[RPA debug] bq_idx={}", bq_idx)
                debug_print("[RPA debug] bkv_idx={}", bkv_idx)
                if debug_mode:
                    # Skip flash attention if debug mode is enabled.
                    return

                # Flash attention with cur bkv and bq
                effective_bkv_sz = jnp.minimum(
                    effective_kv_len - bkv_idx * bkv_sz, bkv_sz)
                effective_bkv_sz = jnp.maximum(effective_bkv_sz, 0)

                num_loops = cdiv(effective_bkv_sz, bkv_csz)

                @pl.loop(0, num_loops, unroll=False)
                def attention_loop(idx):
                    prev_lm_slice = None
                    prev_p = None
                    prev_v = None
                    prev_exp_m_diff = None
                    bkv_start = idx * bkv_csz

                    for bq_start in range(0, actual_bq_sz, actual_bq_csz):
                        for kv_head_idx in range(actual_num_kv_heads):
                            bk_c, bv_c = load_bkv(
                                bkv_sem_idx,
                                kv_head_idx,
                                bkv_start,
                                bkv_csz,
                            )
                            bq_c = load_bq(bq_sem_idx, kv_head_idx, bq_start,
                                           actual_bq_csz)

                            lm_slice_start = bq_start * num_q_heads_per_kv_head
                            lm_slice_size = actual_bq_csz * num_q_heads_per_kv_head
                            lm_slice = (kv_head_idx,
                                        pl.ds(lm_slice_start, lm_slice_size))

                            # FlashAttn is divided into `flash_attention_step1_qk_softmax`
                            # and `flash_attention_step2_pv` to pipeline the computation.
                            # `step2_pv` for the previous KV head, which depends on the
                            # softmax output, is overlapped with `step1_qk_softmax` for the
                            # current KV head, reducing overall wait times.
                            cur_p, cur_v, cur_exp_m_diff = flash_attention_step1_qk_softmax(
                                bq_c,
                                bk_c,
                                bv_c,
                                l_ref.at[*lm_slice],
                                m_ref.at[*lm_slice],
                                processed_q_len=processed_q_len + bq_start,
                                processed_kv_len=processed_kv_len + bkv_start,
                                effective_kv_len=effective_kv_len,
                            )
                            if prev_lm_slice is not None:
                                flash_attention_step2_pv(
                                    prev_p,
                                    prev_v,
                                    prev_exp_m_diff,
                                    acc_ref.at[*prev_lm_slice],
                                )
                            prev_lm_slice = lm_slice
                            prev_p = cur_p
                            prev_v = cur_v
                            prev_exp_m_diff = cur_exp_m_diff

                    # Execute pv of last iteration.
                    assert prev_lm_slice is not None
                    flash_attention_step2_pv(
                        prev_p,
                        prev_v,
                        prev_exp_m_diff,
                        acc_ref.at[*prev_lm_slice],
                    )

            # Load acc and calculate final output.
            acc = acc_ref[...]
            l = broadcast_minor(l_ref[...], acc.shape)  # noqa
            out = (acc * pl.reciprocal(l, approx=True) if
                   (l.dtype == jnp.float32 and out_dtype != jnp.float32) else
                   lax.div(acc, l)).astype(out_dtype)

            # Wait for previous bo to be fully sent before storing new bo.
            bo_sem_idx = sem_ids_ref[2]
            sem_ids_ref[2] = lax.select(bo_sem_idx == 0, 1, 0)
            wait_send_bo(bo_sem_idx)

            # Store output from acc to bo.
            out_ref = (bo_x2_ref.at[bo_sem_idx].bitcast(jnp.int32).reshape(
                actual_num_kv_heads * bq_sz *
                num_q_heads_per_kv_head_per_packing,
                head_dim,
            ))
            out = pltpu.bitcast(out, out_ref.dtype).reshape(out_ref.shape)
            strided_store(out_ref, 0, out_ref.shape[0], 1, out)

            # Send cur bo
            start_send_bo(seq_idx, bq_idx, bo_sem_idx)

    ### ------- Kernel start ------- ###

    @pl.when(seq_idx == start_seq_idx)
    def prologue():
        start_fetch_bq(seq_idx=start_seq_idx, bq_idx=0, bq_sem_idx=0)
        start_fetch_bkv(seq_idx=start_seq_idx,
                        bkv_idx=cur_seq_start_bkv_idx,
                        bkv_sem_idx=0)

    @pl.when(jnp.logical_and(start_seq_idx <= seq_idx, seq_idx < end_seq_idx))
    def pipeline():
        process(static_q_len=static_q_len)

    @pl.when(seq_idx == end_seq_idx - 1)
    def epilogue():
        for i in range(2):
            wait_send_bo(bo_sem_idx=i)
            wait_update_kv_cache(bkv_sem_idx=i)

    ### ------- Kernel end ------- ###


def has_bank_conflicts(stride, distance=24, num_banks=32):
    banks = set()
    for i in range(distance):
        bank = (i * stride) % num_banks
        if bank in banks:
            return True
        banks.add(bank)
    return False


def merge_kv(
        k: jax.
    Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim],
        v: jax.
    Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim],
):
    assert k.shape == v.shape
    assert k.dtype == v.dtype
    max_num_tokens, actual_num_kv_heads, actual_head_dim = k.shape
    kv_packing = get_dtype_packing(k.dtype)
    actual_num_kv_heads_x2 = actual_num_kv_heads * 2
    num_kv_heads_x2 = align_to(actual_num_kv_heads_x2, kv_packing)

    head_dim = align_to(actual_head_dim, 128)
    kv = jnp.pad(
        jnp.concat([k, v],
                   axis=-1).reshape(max_num_tokens, actual_num_kv_heads_x2,
                                    actual_head_dim),
        (
            (0, 0),
            (0, num_kv_heads_x2 - actual_num_kv_heads_x2),
            (0, head_dim - actual_head_dim),
        ),
        constant_values=0,
    ).reshape(
        max_num_tokens,
        num_kv_heads_x2 // kv_packing,
        kv_packing,
        head_dim,
    )
    return kv


def prepare_inputs(
        q: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim],
        k: jax.
    Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim],
        v: jax.
    Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim],
):
    max_num_tokens, actual_num_q_heads, actual_head_dim = q.shape
    actual_num_kv_heads = k.shape[1]
    assert actual_num_q_heads % actual_num_kv_heads == 0
    actual_num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads
    q_packing = get_dtype_packing(q.dtype)
    num_q_heads_per_kv_head = align_to(actual_num_q_heads_per_kv_head,
                                       q_packing)
    head_dim = align_to(actual_head_dim, 128)
    q = (
        jnp.pad(
            q.reshape(
                max_num_tokens,
                actual_num_kv_heads,
                actual_num_q_heads_per_kv_head,
                actual_head_dim,
            ),
            (
                (0, 0),
                (0, 0),
                (0, num_q_heads_per_kv_head - actual_num_q_heads_per_kv_head),
                (0, head_dim - actual_head_dim),
            ),
            constant_values=0,
        ).reshape(
            max_num_tokens,
            actual_num_kv_heads,
            num_q_heads_per_kv_head // q_packing,
            q_packing,
            head_dim,
        )
        # TODO(jevinjiang): Explore fusing swapping non-tiling axis to DMA.
        .swapaxes(0, 1))
    # TODO(kyuyeunk, chengjiyao): Add kv quantization here.
    kv = merge_kv(k, v)
    return q, kv


def prepare_outputs(
    out,  # [actual_num_kv_heads, max_num_tokens, num_q_heads_per_kv_head // q_packing, q_packing, head_dim]
    actual_num_q_heads_per_kv_head: int,
    actual_head_dim: int,
):
    (
        actual_num_kv_heads,
        max_num_tokens,
        num_q_heads_per_kv_head_per_q_packing,
        q_packing,
        head_dim,
    ) = out.shape
    actual_num_q_heads = actual_num_q_heads_per_kv_head * actual_num_kv_heads
    return (out.swapaxes(0, 1).reshape(
        max_num_tokens,
        actual_num_kv_heads,
        num_q_heads_per_kv_head_per_q_packing * q_packing,
        head_dim,
    )[:, :, :actual_num_q_heads_per_kv_head, :actual_head_dim].reshape(
        max_num_tokens, actual_num_q_heads, actual_head_dim))


# Expect to run this validation during runtime.
def validate_logical_prefetches(
    phys_seq_indices: jax.Array | None,
    q_offsets: jax.Array | None,
    max_num_seqs: int,
    distribution: jax.Array | None = None,
):
    """Validate the LOGICAL iter→phys indirection prefetches.

    Checks (always run when both prefetches are provided; static only):
      - Both must be present together (or both None).
      - Same shape, both i32 dtype.
      - 1D arrays.
      - shape[0] >= max_num_seqs (coupled-K identity needs at least
        max_num_seqs entries; LOGICAL needs max_num_subseqs > max_num_seqs).

    No-op if both prefetches are None (= caller hasn't enabled the
    indirection; coupled-K with no prefetches passed). Used by both
    static_validate_inputs and dynamic_validate_inputs as a single source
    of LOGICAL-specific checks.

    Note: value-range checks (phys_seq_indices in [0, max_num_seqs),
    q_offsets non-negative) are NOT performed here. They would require
    a Python-level loop over a JAX-traced shape (`distribution[-1]`),
    which crashes under @jax.jit. Value validation belongs in the
    runner-side construction code (subseq_planner), where the prefetches
    are built in concrete Python before being passed to the kernel. The
    `distribution` parameter is kept in the signature only to mirror the
    static/dynamic split convention used elsewhere in the file; it is
    currently unused.
    """
    del distribution  # unused; see docstring note.
    if phys_seq_indices is None and q_offsets is None:
        return
    if phys_seq_indices is None or q_offsets is None:
        raise ValueError(
            "phys_seq_indices and q_offsets must be provided together "
            f"(got {phys_seq_indices=}, {q_offsets=}).")
    if phys_seq_indices.shape != q_offsets.shape:
        raise ValueError(
            f"Expected {phys_seq_indices.shape=} == {q_offsets.shape=}.")
    if len(phys_seq_indices.shape) != 1:
        raise ValueError(
            f"Expected phys_seq_indices to be 1D, got {phys_seq_indices.shape=}.")
    if phys_seq_indices.dtype != jnp.int32:
        raise ValueError(
            f"Expected i32 phys_seq_indices, got {phys_seq_indices.dtype=}.")
    if q_offsets.dtype != jnp.int32:
        raise ValueError(
            f"Expected i32 q_offsets, got {q_offsets.dtype=}.")
    if phys_seq_indices.shape[0] < max_num_seqs:
        raise ValueError(
            f"phys_seq_indices.shape[0]={phys_seq_indices.shape[0]} must be "
            f">= {max_num_seqs=} (need at least one entry per phys seq).")


def dynamic_validate_inputs(
    queries: jax.
    Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim]
    keys: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    values: jax.
    Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    kv_cache: jax.
    Array,  # [total_num_pages, page_size, num_kv_heads_x2 // kv_packing, kv_packing, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    phys_seq_indices: jax.Array | None = None,  # i32[max_num_subseqs]
    q_offsets: jax.Array | None = None,  # i32[max_num_subseqs]
    *,
    use_causal_mask: bool = True,
    skip_kv_mask: bool = False,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    out_dtype: Any = None,
    mask_value: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    # Kernel optimization params.
    chunk_prefill_size: int | None = None,
    # Kernel tuning params.
    d_block_sizes: tuple[int, int, int, int] | None = None,
    p_block_sizes: tuple[int, int, int, int] | None = None,
    m_block_sizes: tuple[int, int, int, int] | None = None,
    vmem_limit_bytes: int | None = None,
):
    q, k, v = queries, keys, values
    static_validate_inputs(
        q,
        k,
        v,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        phys_seq_indices,
        q_offsets,
        use_causal_mask=use_causal_mask,
        skip_kv_mask=skip_kv_mask,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        out_dtype=out_dtype,
        mask_value=mask_value,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        chunk_prefill_size=chunk_prefill_size,
        d_block_sizes=d_block_sizes,
        p_block_sizes=p_block_sizes,
        m_block_sizes=m_block_sizes,
        vmem_limit_bytes=vmem_limit_bytes,
    )
    max_num_tokens = q.shape[0]
    total_num_pages = kv_cache.shape[0]
    page_size = kv_cache.shape[1]
    max_num_seqs = kv_lens.shape[0]
    # LOGICAL prefetches: dynamic value-range checks (static checks already
    # ran inside static_validate_inputs above).
    validate_logical_prefetches(phys_seq_indices, q_offsets, max_num_seqs,
                                distribution=distribution)
    num_page_indices = page_indices.shape[0]
    assert num_page_indices % max_num_seqs == 0
    pages_per_seq = num_page_indices // max_num_seqs

    i, j, k = distribution
    if not (i <= j <= k):
        raise ValueError(f"Invalid distribution: {distribution=}")

    # Under LOGICAL (decoupled-K), k = num_total iters but cu_q_lens /
    # kv_lens / page_indices are PHYS-keyed (length max_num_seqs+1 /
    # max_num_seqs / max_num_seqs * pages_per_seq). Iterate over phys
    # slots, not iter slots — otherwise the loop reads into the
    # padding region of the phys-keyed arrays where q_len = 0 and
    # the (0 < q_len <= kv_len) check fires spuriously.
    #
    # Coupled-K: phys_seq_indices is None and num_iters == num_phys, so
    # num_phys collapses to k and the iteration is identical to the
    # pre-LOGICAL behaviour.
    if phys_seq_indices is not None:
        # phys_seq_indices is iter-keyed in [0, max_num_seqs). Active
        # phys count = max(active iter slots) + 1.
        num_phys = int(jnp.max(phys_seq_indices[:k])) + 1 if k > 0 else 0
    else:
        num_phys = int(k)

    if num_phys > max_num_seqs:
        raise ValueError(f"{num_phys=} must be <= {max_num_seqs=}")

    if num_phys > 0 and cu_q_lens[num_phys] > max_num_tokens:
        raise ValueError(
            f"Total q tokens {cu_q_lens[num_phys]} must be <= "
            f"{max_num_tokens=}.")
    for i in range(num_phys):
        q_len = cu_q_lens[i + 1] - cu_q_lens[i]
        kv_len = kv_lens[i]
        if not (0 < q_len <= kv_len):
            raise ValueError(
                f"Require 0 < {q_len=} <= {kv_len=} at sequence {i}.")
        page_cnt = cdiv(kv_len, page_size)
        if page_cnt > pages_per_seq:
            raise ValueError(
                f"Require {page_cnt=} <= {pages_per_seq=} at sequence {i} where"
                f" {kv_len=} and {page_size=}.")
        for p in range(page_cnt):
            page_idx = page_indices[i * pages_per_seq + p]
            if not (0 <= page_idx < total_num_pages):
                raise ValueError(
                    f"Require 0 <= {page_idx=} < {total_num_pages=} at sequence"
                    f" {i} where {kv_len=} and {page_size=}.")


# Expect to run this validation during compile time.
def static_validate_inputs(
    queries: jax.
    Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim]
    keys: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    values: jax.
    Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    kv_cache: jax.
    Array,  # [total_num_pages, page_size, num_kv_heads_x2 // kv_packing, kv_packing, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    phys_seq_indices: jax.Array | None = None,  # i32[max_num_subseqs]
    q_offsets: jax.Array | None = None,  # i32[max_num_subseqs]
    *,
    use_causal_mask: bool = True,
    skip_kv_mask: bool = False,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    out_dtype: Any = None,
    mask_value: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    # Kernel optimization params.
    chunk_prefill_size: int | None = None,
    # Kernel tuning params.
    d_block_sizes: tuple[int, int, int, int] | None = None,
    p_block_sizes: tuple[int, int, int, int] | None = None,
    m_block_sizes: tuple[int, int, int, int] | None = None,
    vmem_limit_bytes: int | None = None,
):
    """Validate inputs to the RPA kernel statically."""
    if use_causal_mask:
        if skip_kv_mask:
            raise ValueError("Can not skip kv mask when using causal mask.")

    q, k, v = queries, keys, values
    if not (len(q.shape) == len(k.shape) == len(v.shape) == 3):
        raise ValueError(
            f"Expected 3D array for {q.shape=}, {k.shape=}, {v.shape=}")
    if k.shape != v.shape:
        raise ValueError(f"Expected {k.shape=} to be equal to {v.shape=}")
    if not (q.shape[0] == k.shape[0] == v.shape[0]):
        raise ValueError(
            f"Expected {q.shape[0]=} to be equal to {k.shape[0]=} and {v.shape[0]=}"
        )
    if not (q.shape[2] == k.shape[2] == v.shape[2]):
        raise ValueError(
            f"Expected {q.shape[2]=} to be equal to {k.shape[2]=} and {v.shape[2]=}"
        )

    actual_head_dim = q.shape[2]
    actual_num_q_heads = q.shape[1]
    actual_num_kv_heads = k.shape[1]

    if actual_num_q_heads % actual_num_kv_heads != 0:
        raise ValueError(f"Expected {actual_num_q_heads=} to be divisible by"
                         f" {actual_num_kv_heads=}.")

    expected_kv_cache_shape = get_kv_cache_shape(
        kv_cache.shape[0],
        kv_cache.shape[1],
        actual_num_kv_heads,
        actual_head_dim,
        kv_cache.dtype,
    )

    if kv_cache.shape != expected_kv_cache_shape:
        raise ValueError(
            f"Expected {kv_cache.shape=} to be equal to {expected_kv_cache_shape=}"
        )

    (
        _,
        page_size,
        num_kv_heads_x2_per_kv_packing,
        kv_packing,
        head_dim,
    ) = kv_cache.shape

    if head_dim != align_to(actual_head_dim, 128):
        raise ValueError(
            f"Expected {head_dim=} is equal to {align_to(actual_head_dim, 128)=}"
        )
    # Note: we expect the kv quantization happens outside of the RPA kernel.
    if not (kv_cache.dtype == k.dtype == v.dtype):
        raise ValueError(
            f"Expected {kv_cache.dtype=} to be equal to {k.dtype=} and {v.dtype=}."
        )
    # Integer kv quantization is currently not supported.
    if not jnp.issubdtype(kv_cache.dtype, jnp.floating):
        raise ValueError(f"Expected {kv_cache.dtype=} to be a floating point.")
    if kv_packing != get_dtype_packing(kv_cache.dtype):
        raise ValueError(
            f"{kv_packing=} does not match with {kv_cache.dtype=}")

    num_kv_heads_x2 = num_kv_heads_x2_per_kv_packing * kv_packing
    if num_kv_heads_x2 % 2 != 0:
        raise ValueError(
            f"Combined KV heads must be divisible by 2, but got {num_kv_heads_x2}"
        )
    if (num_kv_heads_x2 % kv_packing != 0
            or num_kv_heads_x2 // 2 < actual_num_kv_heads):
        raise ValueError(
            f"Invalid {num_kv_heads_x2=}, {actual_num_kv_heads=}, {kv_packing=}"
        )

    if not (jnp.int32 == kv_lens.dtype == page_indices.dtype == cu_q_lens.dtype
            == distribution.dtype):
        raise ValueError(
            f"Expected int32 dtype for {kv_lens.dtype=}, {page_indices.dtype=},"
            f" {cu_q_lens.dtype=}, {distribution.dtype=}")

    if not (len(kv_lens.shape) == len(page_indices.shape) == len(
            cu_q_lens.shape) == 1):
        raise ValueError(
            f"Expected 1D array for {kv_lens.shape=}, {page_indices.shape=},"
            f" {cu_q_lens.shape=}")

    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]
    if num_page_indices % max_num_seqs != 0:
        raise ValueError(
            f"Expected {num_page_indices=} to be divisible by {max_num_seqs=}."
        )
    if cu_q_lens.shape != (max_num_seqs + 1, ):
        raise ValueError(
            f"Expected {cu_q_lens.shape=} to be ({max_num_seqs + 1},).")
    if distribution.shape != (3, ):
        raise ValueError(f"Expected {distribution.shape=} to be (3,).")

    if page_size % kv_packing != 0:
        raise ValueError(f"{page_size=} must be divisible by {kv_packing=}.")
    if sliding_window is not None and sliding_window <= 0:
        raise ValueError(f"{sliding_window=} must be positive.")
    if soft_cap is not None and soft_cap == 0.0:
        raise ValueError(f"{soft_cap=} must not be 0.0.")
    if chunk_prefill_size is not None and chunk_prefill_size <= 0:
        raise ValueError(f"{chunk_prefill_size=} must be positive.")

    def _validate_block_sizes(block_sizes, prefix):
        if block_sizes is None:
            return
        bq_sz, bkv_sz, bq_csz, bkv_csz = block_sizes
        if not (0 < bq_csz and bq_sz % bq_csz == 0):
            raise ValueError(
                f"{prefix} {bq_csz=} and {bq_sz=} must satisfy (0 < bq_csz and bq_sz"
                " % bq_csz == 0).")
        if not (0 < bkv_csz and bkv_sz % bkv_csz == 0):
            raise ValueError(
                f"{prefix} {bkv_csz=} and {bkv_sz=} must satisfy (0 < bkv_csz and"
                " bkv_sz % bkv_csz == 0).")
        if bkv_sz % page_size != 0:
            raise ValueError(
                f"{prefix} {bkv_sz=} must be divisible by {page_size=}.")
        if bkv_csz % page_size != 0:
            raise ValueError(
                f"{prefix} {bkv_csz=} must be divisible by {page_size=}.")

    _validate_block_sizes(d_block_sizes, "decode")
    _validate_block_sizes(p_block_sizes, "prefill")
    _validate_block_sizes(m_block_sizes, "mixed")

    if vmem_limit_bytes is not None and vmem_limit_bytes <= 0:
        raise ValueError(f"{vmem_limit_bytes=} must be positive.")

    # LOGICAL prefetches: static checks (shape, dtype). No-op if both None.
    validate_logical_prefetches(phys_seq_indices, q_offsets, max_num_seqs,
                                distribution=None)

    # No constraints for the following inputs.
    del sm_scale
    del mask_value
    del out_dtype
    del q_scale
    del k_scale
    del v_scale


def get_default_block_sizes(
    q_dtype,
    kv_dtype,
    actual_num_q_heads,
    actual_num_kv_heads,
    head_dim,
    page_size,
    max_num_tokens,
    max_num_seqs,
    pages_per_seq,
    *,
    case: RpaCase = RpaCase.MIXED,
):
    """Get (bq, bkv_sz, bq_csz, bkv_csz) by some heuristic formulas.

    Note the default block sizes are not necessarily optimal.
    """
    tpu_version = get_tpu_version()

    kv_packing = get_dtype_packing(kv_dtype)
    num_kv_heads_x2 = next_power_of_2(
        align_to(actual_num_kv_heads * 2, kv_packing))
    head_dim = align_to(head_dim, 128)
    num_q_heads_per_kv_head = next_power_of_2(actual_num_q_heads //
                                              actual_num_kv_heads)

    max_q = next_power_of_2(max_num_tokens)
    max_kv = pages_per_seq * page_size

    min_bkv_sz_to_peak = (16 * 1024 * 1024 * kv_packing // 4 // head_dim //
                          num_kv_heads_x2)

    match tpu_version:
        case 5 | 6:
            if case == RpaCase.DECODE:
                bq_sz = 1
                bkv_sz = min(min_bkv_sz_to_peak, max_kv)
                bq_csz = 1
                bkv_csz = min(min_bkv_sz_to_peak, max_kv)
            else:
                bq_sz = min(1024 // num_q_heads_per_kv_head, max_q // 2)
                bkv_sz = min(1024, max_kv)
                bq_csz = min(512 // num_q_heads_per_kv_head, max_q)
                bkv_csz = min(512, align_to(max_kv // 2, page_size))
        case 7:
            if case == RpaCase.DECODE:
                bq_sz = 1
                bkv_sz = min(min_bkv_sz_to_peak, max_kv)
                bq_csz = 1
                bkv_csz = min(min_bkv_sz_to_peak, max_kv)
            else:
                bq_sz = min(2048 // num_q_heads_per_kv_head, max_q // 2)
                bkv_sz = min(2048, max_kv // 2)
                bq_csz = min(1024 // num_q_heads_per_kv_head, max_q // 2)
                bkv_csz = min(512, align_to(max_kv // 2, page_size))
        case _:
            raise NotImplementedError(f"Unsupported {tpu_version=}.")

    return {
        "bq_sz": max(1, bq_sz),
        "bkv_sz": align_to(bkv_sz, page_size),
        "bq_csz": max(1, bq_csz),
        "bkv_csz": align_to(bkv_csz, page_size),
    }


@jax.jit(
    static_argnames=(
        "use_causal_mask",
        "skip_kv_mask",
        "sm_scale",
        "sliding_window",
        "soft_cap",
        "out_dtype",
        "mask_value",
        "q_scale",
        "k_scale",
        "v_scale",
        "chunk_prefill_size",
        "d_block_sizes",
        "p_block_sizes",
        "m_block_sizes",
        "vmem_limit_bytes",
        "debug_mode",
        "disable_bounds_checks",
        "disable_semaphore_checks",
    ),
    donate_argnames=("queries", "keys", "values", "kv_cache"),
)
def ragged_paged_attention(
    queries: jax.
    Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim]
    keys: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    values: jax.
    Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    kv_cache: jax.
    Array,  # [total_num_pages, page_size, num_kv_heads_x2 // kv_packing, kv_packing, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    phys_seq_indices: jax.Array | None = None,  # i32[max_num_subseqs]
    q_offsets: jax.Array | None = None,  # i32[max_num_subseqs]
    *,
    use_causal_mask: bool = True,
    skip_kv_mask: bool = False,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    out_dtype: Any = None,
    mask_value: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    # Kernel optimization params.
    chunk_prefill_size: int | None = None,
    # Kernel tuning params for decode, prefill, and mixed cases.
    # Each case takes a tuple of (bq_sz, bkv_sz, bq_csz, bkv_csz).
    # - bq_sz: the block size for the query fetching.
    # - bkv_sz: the block size for the kv fetching.
    # - bq_csz: the compute size of the block query.
    # - bkv_csz: the compute size of the block kv.
    d_block_sizes: tuple[int, int, int, int] | None = None,
    p_block_sizes: tuple[int, int, int, int] | None = None,
    m_block_sizes: tuple[int, int, int, int] | None = None,
    vmem_limit_bytes: int | None = None,
    # Debug params.
    debug_mode: bool = False,
    disable_bounds_checks: bool = True,
    disable_semaphore_checks: bool = True,
):
    """Ragged paged attention that supports mixed prefill and decode.

  Args:
    queries: concatenated all sequences' queries.
    keys: concatenated all sequences' keys (quantized).
    values: concatenated all sequences' values (quantized).
    kv_cache: paged KV cache with TPU-friendly shape.
    kv_lens: padded kv lengths. Only the first num_seqs values are valid.
    page_indices: flattened page indices look-up table by (seq_id, page_id).
    cu_q_lens: the cumulative sum of the effective query lengths. Similar to
      kv_lens, only the first num_seqs+1 values are valid.
    distribution: (i, j, k) represents that sequences[0:i] are decode-only,
      sequences[i:j] are chunked-prefill-only, and sequences[j:k] are mixed. The
      k is also the total number of sequences.
    use_causal_mask: if true, use causal mask.
    skip_kv_mask: only set to true if use_causal_mask=False and each dynamic
      kv_len % bkv_csz == 0. Set to true can improve performance.
    sm_scale: the softmax scale which will be applied to the Q@K^T.
    sliding_window: the sliding window size for the attention.
    soft_cap: the logit soft cap for the attention.
    out_dtype: the dtype of the output and the accumulator for matmul. Set
      lower for better performance, set higher for better accuracy. If None, it
      uses q.dtype.
    mask_value: mask value for causal mask.
    q_scale: the scale for the query.
    k_scale: the scale for the key.
    v_scale: the scale for the value.
    chunk_prefill_size: the chunk prefill size for the attention.
    d_block_sizes: the block sizes for the decode case.
    p_block_sizes: the block sizes for the prefill case.
    m_block_sizes: the block sizes for the mixed case.
    vmem_limit_bytes: the vmem limit for the pallas kernel.
    debug_mode: if true, RPA does not issue any DMAs or run flash attention but
      print debug info. Need to compile with `--xla_tpu_enable_log_recorder`.
    disable_bounds_checks: if true, disable bounds checks.
    disable_semaphore_checks: if true, disable semaphore checks.

  Returns:
    The output of the attention.
  """
    q, k, v = queries, keys, values
    tpu_version = get_tpu_version()

    if out_dtype is None:
        out_dtype = jnp.float32 if q.dtype == jnp.float32 else jnp.bfloat16

    if mask_value is None:
        # We do not set to -inf directly because (-inf) - (-inf) is nan.
        mask_value = jnp.finfo(out_dtype).min

    if vmem_limit_bytes is None:
        # TODO(jevinjiang, jacobplatin): change this to use
        # `get_vmem_estimate_bytes` when VREG spilling is fixed.
        vmem_limit_bytes = pltpu.get_tpu_info().vmem_capacity_bytes

    static_validate_inputs(
        q,
        k,
        v,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        phys_seq_indices,
        q_offsets,
        use_causal_mask=use_causal_mask,
        skip_kv_mask=skip_kv_mask,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        out_dtype=out_dtype,
        mask_value=mask_value,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        chunk_prefill_size=chunk_prefill_size,
        d_block_sizes=d_block_sizes,
        p_block_sizes=p_block_sizes,
        m_block_sizes=m_block_sizes,
        vmem_limit_bytes=vmem_limit_bytes,
    )

    actual_num_q_heads = q.shape[1]
    actual_head_dim = q.shape[2]
    actual_num_kv_heads = k.shape[1]

    actual_num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads
    q, kv = prepare_inputs(q, k, v)
    (
        _,
        max_num_tokens,
        num_q_heads_per_kv_head_per_q_packing,
        q_packing,
        head_dim,
    ) = q.shape
    page_size = kv_cache.shape[1]
    num_kv_heads_x2_per_kv_packing = kv_cache.shape[2]
    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]
    assert num_page_indices % max_num_seqs == 0
    pages_per_seq = num_page_indices // max_num_seqs
    num_q_heads_per_kv_head = num_q_heads_per_kv_head_per_q_packing * q_packing

    # LOGICAL dispatch gate: presence-based, NOT shape-based. Caller signals
    # "use LOGICAL" by explicitly passing prefetches (any values, any shape).
    # If they pass None, we apply identity defaults and dispatch to today's
    # coupled-K PREFILL. Capturing this BEFORE applying defaults so the
    # signal isn't lost. Avoids the silent footgun where a max_num_seqs-
    # sized but non-identity array would have been routed to PREFILL.
    use_logical_prefill = phys_seq_indices is not None

    # Default the iter→phys prefetches to identity for coupled-K (D/P/M).
    # The kernel reads these uniformly with no case-based branching;
    # identity values make the formulas reduce numerically to today's
    # behavior.
    if phys_seq_indices is None:
        phys_seq_indices = jnp.arange(max_num_seqs, dtype=jnp.int32)
    if q_offsets is None:
        q_offsets = jnp.zeros(max_num_seqs, dtype=jnp.int32)

    # (bq_sem_idx, bkv_sem_idx, bo_sem_idx)
    init_sem_ids = jnp.zeros((3, ), jnp.int32)
    # (bo_sem_0_seq_idx, bo_sem_1_seq_idx, bo_sem_0_bo_idx, bo_sem_1_bo_idx)
    init_bo_ids = jnp.full((4, ), -1, jnp.int32)
    # (bkv_sem_0_seq_idx, bkv_sem_1_seq_idx, bkv_sem_0_offset, bkv_sem_1_offset, bkv_sem_0_sz, bkv_sem_1_sz)
    init_bkv_update_ids = jnp.full((6, ), -1, jnp.int32)

    def run_rpa_kernel(
        q,
        kv_cache,
        *,
        bq_sz,
        bkv_sz,
        bq_csz,
        bkv_csz,
        static_q_len=None,
        case: RpaCase = RpaCase.MIXED,
    ):
        in_specs = [
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ]

        out_specs = [
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ]

        bkv_stride = num_kv_heads_x2_per_kv_packing
        if has_bank_conflicts(bkv_stride):
            bkv_stride += 1

        bkv_double_buf = pltpu.VMEM(
            (2, bkv_sz, bkv_stride, *kv_cache.shape[3:]),
            kv_cache.dtype,
        )

        bq_double_buf = pltpu.VMEM(
            (2, actual_num_kv_heads, bq_sz, *q.shape[2:]),
            q.dtype,
        )

        bo_double_buf = bq_double_buf

        l_scratch = pltpu.VMEM(
            (actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, 128),
            out_dtype,
        )
        m_scratch = l_scratch

        acc_scratch = pltpu.VMEM(
            (actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, head_dim),
            out_dtype,
        )

        scratch_shapes = [
            bkv_double_buf,  # (bkv_x2_ref) Double buffering for kv block.
            bq_double_buf,  # (bq_x2_ref) Double buffering for q block.
            bo_double_buf,  # (bo_x2_ref) Double buffering for output block.
            # Semaphores for double buffering of bkv, bq, bo and bkv_update.
            pltpu.SemaphoreType.DMA((4, 2)),
            # Intermediate buffers per kv head for flash attention.
            l_scratch,
            m_scratch,
            acc_scratch,
        ]

        scalar_prefetches = (
            kv_lens,
            # TODO(jevinjiang): can we use ragged page_indices to save some smem?
            page_indices,
            cu_q_lens,
            # Iter→phys indirection. For coupled-K (D/P/M), caller passes
            # identity (`arange`/`zeros`); for LOGICAL, caller passes real
            # values. The kernel reads these uniformly without case branching.
            phys_seq_indices,
            q_offsets,
            distribution,
            init_sem_ids,
            init_bo_ids,
            init_bkv_update_ids,
        )

        scope_name = f"RPA{case.symbol}-p_{page_size}-bq_{bq_sz}_{bq_csz}-bkv_{bkv_sz}_{bkv_csz}"
        if sliding_window is not None:
            scope_name += f"-sw_{sliding_window}"
        kernel = pl.pallas_call(
            functools.partial(
                _ragged_paged_attention_kernel,
                use_causal_mask=use_causal_mask,
                skip_kv_mask=skip_kv_mask,
                sm_scale=sm_scale,
                sliding_window=sliding_window,
                soft_cap=soft_cap,
                mask_value=mask_value,
                q_scale=q_scale,
                k_scale=k_scale,
                v_scale=v_scale,
                static_q_len=static_q_len,
                bq_sz=bq_sz,
                bkv_sz=bkv_sz,
                bq_csz=bq_csz,
                bkv_csz=bkv_csz,
                case=case,
                debug_mode=debug_mode,
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=len(scalar_prefetches),
                in_specs=in_specs,
                out_specs=out_specs,
                grid=(1, ),
                scratch_shapes=scratch_shapes,
            ),
            compiler_params=pltpu.CompilerParams(
                # TODO(jevinjiang): since each sequence depends on the previous
                # one, we need some extra work to support Megacore mode.
                dimension_semantics=("arbitrary", ),
                vmem_limit_bytes=vmem_limit_bytes,
                # Paged attention invokes multiple small DMAs for each pages
                # instead of a single large DMA. Therefore, the overhead of bounds
                # checking becomes too significant so we disable it.
                disable_bounds_checks=disable_bounds_checks,
                # Only set to true if you gurantee there is no race condition.
                disable_semaphore_checks=disable_semaphore_checks,
            ),
            out_shape=[
                pltpu.HBM(shape=q.shape, dtype=q.dtype),
                pltpu.HBM(shape=kv_cache.shape, dtype=kv_cache.dtype),
            ] if tpu_version >= 7 else [
                jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
                jax.ShapeDtypeStruct(shape=kv_cache.shape,
                                     dtype=kv_cache.dtype),
            ],
            input_output_aliases={
                # Two new prefetches added before q/kv/kv_cache positional
                # args, so each shifts by +2 from the prior (7, 9) layout.
                9: 0,    # q -> output 0
                11: 1,   # kv_cache -> output 1
            },
            name=scope_name,
        )

        if tpu_version >= 7:
            # jit to color the memory since the q, kv are just preprocessed.
            @jax.jit
            def run(scalar_prefetches, q, kv, kv_cache):
                return kernel(
                    *scalar_prefetches,
                    pltpu.with_memory_space_constraint(q, pltpu.HBM),
                    pltpu.with_memory_space_constraint(kv, pltpu.HBM),
                    pltpu.with_memory_space_constraint(kv_cache, pltpu.HBM),
                )
        else:
            # TODO(b/494285697): v6 has issues with pinning aliased memory.
            def run(scalar_prefetches, q, kv, kv_cache):
                return kernel(*scalar_prefetches, q, kv, kv_cache)

        return run(scalar_prefetches, q, kv, kv_cache)

    def _prepare_block_sizes(block_sizes, case):
        if block_sizes is None:
            return get_default_block_sizes(
                q.dtype,
                kv_cache.dtype,
                actual_num_q_heads,
                actual_num_kv_heads,
                head_dim,
                page_size,
                max_num_tokens,
                max_num_seqs,
                pages_per_seq,
                case=case,
            )
        return {
            "bq_sz": block_sizes[0],
            "bkv_sz": block_sizes[1],
            "bq_csz": block_sizes[2],
            "bkv_csz": block_sizes[3],
        }

    # `use_logical_prefill` was computed earlier (presence-based gate). When
    # True, LOGICAL replaces the PREFILL pass; DECODE and MIXED still run.

    # Decode-only
    q, kv_cache = run_rpa_kernel(
        q,
        kv_cache,
        **_prepare_block_sizes(d_block_sizes, RpaCase.DECODE),
        static_q_len=1,
        case=RpaCase.DECODE,
    )
    if chunk_prefill_size is not None:
        if use_logical_prefill:
            # LOGICAL replaces PREFILL: same K-static block-size sweet spot,
            # but iters are LOGICAL sub-seqs (decoupled K_sched / K_kernel).
            q, kv_cache = run_rpa_kernel(
                q,
                kv_cache,
                **_prepare_block_sizes(p_block_sizes, RpaCase.LOGICAL),
                static_q_len=chunk_prefill_size,
                case=RpaCase.LOGICAL,
            )
        else:
            # Prefill-only (today's coupled-K path).
            q, kv_cache = run_rpa_kernel(
                q,
                kv_cache,
                **_prepare_block_sizes(p_block_sizes, RpaCase.PREFILL),
                static_q_len=chunk_prefill_size,
                case=RpaCase.PREFILL,
            )
    # Mixed
    q, kv_cache = run_rpa_kernel(
        q,
        kv_cache,
        **_prepare_block_sizes(m_block_sizes, RpaCase.MIXED),
        static_q_len=None,
        case=RpaCase.MIXED,
    )

    return (
        prepare_outputs(q, actual_num_q_heads_per_kv_head, actual_head_dim),
        kv_cache,
    )
