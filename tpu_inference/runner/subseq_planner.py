# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Sub-sequence planner for decoupled K_sched / K_kernel chunked prefill.

Background
----------
Today the runner uses a single K threaded through both vLLMs scheduler
(``LONG_PREFILL_TOKEN_THRESHOLD``) and the static-K PREFILL kernel pass
(``chunk_prefill_size``). With the same K on both sides, vLLM clamps each
request to K tokens per step and the runner buckets a request into PREFILL
only when its scheduled token count is exactly K — otherwise it lands in
MIXED.

To run the static-K PREFILL kernel for *more* of a requests prefill per
step, we want to **decouple K_sched (vLLMs per-request-per-step cap) from
K_kernel (the kernels static-q-len)**. Concretely: vLLM hands the runner
``n_R`` tokens for request R per step; the runner internally splits ``n_R``
into ``floor(n_R / K_kernel)`` sub-chunks of size ``K_kernel`` (each routed
through PREFILL) plus a single ``n_R % K_kernel`` remainder (routed through
MIXED). The kernels existing single-pallas_call sequential ``@pl.loop``
processes the chunks in order, with KV writes from chunk N visible to chunk
N+1 via the in-place ``input_output_aliases`` on the KV-cache buffer.

This module produces the **sub-sequence plan** for a step: a list of virtual
sub-seq entries that the runner then materialises into the kernels
prefetch arrays (cu_q_lens, kv_lens, page_indices, request_distribution).
The planner is intentionally pure (no JAX, no TPU, no I/O) so it can be
unit-tested without the runtime stack.

Wire-up to the actual prefetch-array construction lives in a follow-up PR;
see TODO marker in ``persistent_batch_manager._reorder_batch``.

Subseq layout & ordering
------------------------
Per real request R with ``n_R`` scheduled tokens:

  - ``n_R == 0``: no sub-seq emitted.
  - ``n_R == 1``: one DECODE sub-seq (q_len=1).
  - ``n_R >= K_kernel``:
      * ``floor(n_R / K_kernel)`` PREFILL sub-seqs, each q_len=K_kernel,
        with monotonically increasing kv_len_at_end values.
      * If ``n_R % K_kernel > 0``: one MIXED sub-seq with q_len equal to
        the remainder.
  - ``1 < n_R < K_kernel``: one MIXED sub-seq (no full chunk). Same as
    todays behaviour for short prefills.

When ``K_kernel is None`` (i.e. the decoupling is disabled), behaviour
matches todays: any non-decode request emits one MIXED sub-seq for its
full ``n_R``. The PREFILL bucket is empty.

The output list is ordered:

  1. All DECODE sub-seqs (ordered by input req_id_order).
  2. All PREFILL sub-seqs (ordered by req_id_order; *within* a single real
     reqs sub-seqs, the chunks are in q-token order so chunk N+1 sees the
     KV writes of chunk N when the kernels @pl.loop iterates them
     sequentially).
  3. All MIXED sub-seqs (ordered by req_id_order).

The kernels case slicing reads (D, D+P, T) from request_distribution and
iterates each slice. Within a slice, sub-seqs run in order; across slices,
kernel calls run in order (DECODE -> PREFILL -> MIXED).
"""

from __future__ import annotations

import dataclasses
import enum
import warnings
from collections.abc import Iterable
from typing import Protocol


class _IntIndexable(Protocol):
    """Anything indexable by int returning a value coercible to int.

    Production use case is ``input_batch.num_computed_tokens_cpu`` (a
    numpy int32 array). Tests pass plain lists. The Protocol exists so
    consumers can substitute either without an ``Any`` annotation
    swallowing legitimate type errors.
    """
    def __getitem__(self, idx: int) -> int | float: ...


class SubSeqKind(enum.Enum):
    """Which kernel pass a sub-seq is routed through."""
    DECODE = "decode"
    PREFILL = "prefill"   # static-q-len = K_kernel
    MIXED = "mixed"       # dynamic q_len; remainder of a chunked req or short prefill


@dataclasses.dataclass(frozen=True)
class SubSeqEntry:
    """One sub-seq entry in the per-step plan.

    Attributes:
      real_req_id: The vLLM request_id this sub-seq belongs to. Multiple
        sub-seqs can share the same real_req_id when a request is split
        into K_kernel chunks.
      kind: Which kernel pass (DECODE / PREFILL / MIXED) this sub-seq runs
        on.
      q_offset_in_real_req: Offset into the real requests scheduled token
        slice for this sub-seqs first token. 0 for the first sub-seq of a
        request, K_kernel for the second, etc.
      q_len: Number of query tokens this sub-seq processes. K_kernel for
        PREFILL sub-seqs; 1 for DECODE; remainder (>=1) for MIXED.
      kv_len_at_end: Cumulative KV length for this requests pages AFTER
        this sub-seq finishes its KV writes. Equals
        ``prior_kv_len + q_offset_in_real_req + q_len``.
      prior_kv_len: The requests num_computed_tokens at the start of this
        scheduler step (before any of this steps tokens are added).
        Carried for the runtimes convenience when populating kv_lens_ref.
    """
    real_req_id: str
    kind: SubSeqKind
    q_offset_in_real_req: int
    q_len: int
    kv_len_at_end: int
    prior_kv_len: int


@dataclasses.dataclass(frozen=True)
class StepPlan:
    """The full per-step sub-seq plan.

    request_distribution = [num_decode, num_decode + num_prefill, num_total]
    matches the ``[D, D+P, T]`` triple the kernel reads from
    ``distribution_ref``. ``num_total = num_decode + num_prefill +
    num_mixed`` and equals ``len(subseqs)``.

    The ``subseqs`` list is ordered: DECODE entries first, then PREFILL,
    then MIXED. Within PREFILL, sub-seqs of the same real_req_id are
    contiguous and in q-token order so the kernels sequential @pl.loop
    sees correct KV-write ordering.
    """
    subseqs: tuple[SubSeqEntry, ...]
    num_decode: int
    num_prefill: int
    num_mixed: int

    @property
    def num_total(self) -> int:
        return self.num_decode + self.num_prefill + self.num_mixed

    @property
    def request_distribution(self) -> tuple[int, int, int]:
        """The ``[D, D+P, T]`` triple consumed by the kernels case slicing."""
        return (self.num_decode, self.num_decode + self.num_prefill,
                self.num_total)


def plan_step(
    *,
    num_scheduled_tokens: dict[str, int],
    prior_kv_lens: dict[str, int],
    req_id_order: Iterable[str],
    K_kernel: int | None,
) -> StepPlan:
    """Build the per-step sub-seq plan.

    Args:
      num_scheduled_tokens: vLLMs per-request scheduled-tokens dict for
        this step. Must contain an entry for every req_id in
        ``req_id_order`` (a 0 entry is allowed and yields no sub-seqs).
      prior_kv_lens: Each requests num_computed_tokens at the START of
        this step (before this steps tokens are added). Required for
        every req_id with a positive scheduled count.
      req_id_order: The persistent-batch-managers stable ordering of
        req_ids. Determines DECODE/PREFILL/MIXED ordering within each
        bucket.
      K_kernel: Static q-len of the PREFILL kernel pass. ``None`` disables
        decoupling: every multi-token scheduled count goes to MIXED
        (matches todays behaviour when chunked-prefill is off or
        K_sched=K_kernel and no remainder is produced). Must be > 1 when
        not None — K_kernel == 1 would degenerate to "every token is its
        own DECODE-shaped sub-seq", which is incorrect since DECODE has
        its own dedicated kernel pass.

    Returns:
      A StepPlan with sub-seqs grouped DECODE -> PREFILL -> MIXED. The
      caller materialises this into prefetch arrays (cu_q_lens, kv_lens,
      page_indices, request_distribution).

    Raises:
      ValueError: K_kernel <= 1, or req_id_order references a req not in
        num_scheduled_tokens, or a positive-count req is missing from
        prior_kv_lens.
    """
    if K_kernel is not None and K_kernel <= 1:
        raise ValueError(
            f"K_kernel must be > 1 when set; got {K_kernel!r}. "
            f"DECODE has its own kernel pass with static q_len=1.")

    decodes: list[SubSeqEntry] = []
    prefills: list[SubSeqEntry] = []
    mixeds: list[SubSeqEntry] = []

    for req_id in req_id_order:
        if req_id not in num_scheduled_tokens:
            raise ValueError(
                f"req_id {req_id!r} in req_id_order but not in "
                f"num_scheduled_tokens")
        n_R = num_scheduled_tokens[req_id]
        if n_R <= 0:
            continue

        if req_id not in prior_kv_lens:
            raise ValueError(
                f"req_id {req_id!r} has {n_R} scheduled tokens but no "
                f"prior_kv_len was supplied")
        prior = prior_kv_lens[req_id]

        # DECODE — short-circuit. q_len=1 always lands here, regardless
        # of K_kernel.
        if n_R == 1:
            decodes.append(SubSeqEntry(
                real_req_id=req_id,
                kind=SubSeqKind.DECODE,
                q_offset_in_real_req=0,
                q_len=1,
                kv_len_at_end=prior + 1,
                prior_kv_len=prior,
            ))
            continue

        # No K_kernel: everything non-decode goes to MIXED (todays behaviour).
        if K_kernel is None:
            mixeds.append(SubSeqEntry(
                real_req_id=req_id,
                kind=SubSeqKind.MIXED,
                q_offset_in_real_req=0,
                q_len=n_R,
                kv_len_at_end=prior + n_R,
                prior_kv_len=prior,
            ))
            continue

        # K_kernel set — split into floor(n_R / K_kernel) PREFILL chunks
        # + (optional) MIXED remainder.
        num_full_chunks = n_R // K_kernel
        remainder = n_R % K_kernel

        for chunk_idx in range(num_full_chunks):
            offset = chunk_idx * K_kernel
            prefills.append(SubSeqEntry(
                real_req_id=req_id,
                kind=SubSeqKind.PREFILL,
                q_offset_in_real_req=offset,
                q_len=K_kernel,
                kv_len_at_end=prior + offset + K_kernel,
                prior_kv_len=prior,
            ))

        if remainder > 0:
            offset = num_full_chunks * K_kernel
            mixeds.append(SubSeqEntry(
                real_req_id=req_id,
                kind=SubSeqKind.MIXED,
                q_offset_in_real_req=offset,
                q_len=remainder,
                kv_len_at_end=prior + offset + remainder,
                prior_kv_len=prior,
            ))

    return StepPlan(
        subseqs=tuple(decodes + prefills + mixeds),
        num_decode=len(decodes),
        num_prefill=len(prefills),
        num_mixed=len(mixeds),
    )


def build_prior_kv_lens(
    *,
    req_ids_in_order: Iterable[str],
    num_scheduled_tokens: dict[str, int],
    req_id_to_index: dict[str, int],
    num_computed_tokens_cpu: _IntIndexable,
) -> dict[str, int]:
    """Build the per-step ``prior_kv_lens`` snapshot for ``plan_step``.

    Pure-Python helper extracted from
    PersistentBatchManager.compute_step_plan so the data-marshalling
    logic is testable without importing the vLLM/JAX runtime stack.

    Args:
      req_ids_in_order: post-reorder ordering of request ids in the
        persistent batch (typically ``input_batch.req_ids``).
      num_scheduled_tokens: vLLMs per-request scheduled-tokens dict for
        this step.
      req_id_to_index: maps request id -> persistent-batch slot index
        (typically ``input_batch.req_id_to_index``).
      num_computed_tokens_cpu: array-like indexable by slot index that
        returns the requests num_computed_tokens at the START of this
        step. Typically ``input_batch.num_computed_tokens_cpu`` (a
        numpy int32 array). The return is coerced to ``int`` so plain
        Python lists also work for offline tests.

    Returns:
      ``{req_id: prior_kv_len}`` for every req with a positive
      scheduled-tokens count and a known persistent-batch slot.

    Skipping rules — note these diverge slightly from ``plan_step``s
    contract on missing-from-scheduled, intentionally as belt-and-
    suspenders:

      - n_R == 0 in num_scheduled_tokens: skipped here (matches
        plan_step, which also skips n_R <= 0).
      - n_R missing from num_scheduled_tokens (req_id absent): SKIPPED
        here, but plan_step **raises ValueError** when it sees an absent
        req_id in req_id_order. In production both calls share the
        same num_scheduled_tokens dict, so this branch is unreachable
        — defensive code that documents the asymmetry rather than
        propagating a corner-case bug.
      - req_id missing from req_id_to_index (race against the
        persistent-batch update during update_states): skipped here
        rather than IndexErroring downstream.

    The defensive skips mean a malformed call wont crash inside
    plan_step with a confusing error — it produces a smaller prior_kv_lens
    dict, and plan_step will raise on its own input-validation pass with
    a clearer message.
    """
    prior: dict[str, int] = {}
    for req_id in req_ids_in_order:
        n = num_scheduled_tokens.get(req_id, 0)
        if n <= 0:
            continue
        idx = req_id_to_index.get(req_id)
        if idx is None:
            continue
        prior[req_id] = int(num_computed_tokens_cpu[idx])
    return prior


@dataclasses.dataclass(frozen=True)
class IterPrefetches:
    """Iter-keyed prefetch arrays for the LOGICAL kernel dispatch.

    These pair with the existing phys-keyed cu_q_lens (= query_start_loc)
    and kv_lens (= seq_lens) on AttentionMetadata. The kernel uses
    phys_seq_indices[seq_idx] to map iter -> phys slot for cu_q_lens /
    kv_lens / page_indices lookup, and q_offsets[seq_idx] together with
    static_q_len to define this iters [q_start, q_end) inside the phys
    reqs scheduled-token slice.

    Both lists have len ``max_num_subseqs`` (sized by
    ``required_max_num_subseqs``); plan.num_total <= max_num_subseqs is
    enforced at build time. Slots [num_total, max_num_subseqs) are
    zero-padded; the kernel never reads them since
    request_distribution[2] = num_total caps the iter count.
    """
    phys_seq_indices: list[int]
    q_offsets: list[int]


def build_iter_prefetches(
    *,
    plan: StepPlan,
    req_id_to_index: dict[str, int],
    max_num_seqs: int,
    max_num_subseqs: int,
    static_q_len: int | None,
    num_scheduled_tokens: dict[str, int] | None = None,
) -> IterPrefetches:
    """Materialise a StepPlan into the kernels iter-keyed prefetches.

    Pure-stdlib. Caller (PersistentBatchManager) converts the int lists
    to int32 numpy arrays / device buffers when packing into the
    AttentionMetadata fields.

    Args:
      plan: StepPlan from plan_step() — its sub-seq order
        (DECODE -> PREFILL -> MIXED) determines the iter dimension.
      req_id_to_index: persistent-batchs slot mapping
        (typically input_batch.req_id_to_index). Maps real_req_id ->
        phys_seq_idx.
      max_num_seqs: persistent-batch capacity. Phys indices must be in
        [0, max_num_seqs).
      max_num_subseqs: prefetch-array sizing bound (see
        required_max_num_subseqs). plan.num_total must be <= this.
      static_q_len: K_kernel when running the LOGICAL pass. None on
        todays coupled-K path; the LOGICAL bounds check below is a
        no-op in that case.
      num_scheduled_tokens: per-req scheduled-token counts for this
        step. Required when static_q_len is set, used to bounds-check
        each PREFILL sub-seqs (q_offset + static_q_len) against the
        phys reqs total scheduled tokens. Pass None on the coupled-K
        path; the check is skipped.

    Raises:
      ValueError: on any of (the kernel cannot recover from these —
      silent OOB / mis-routed reads — so the runner fails loud at
      plan-materialisation time):
        - plan.num_total > max_num_subseqs.
        - sub-seqs real_req_id missing from req_id_to_index.
        - phys index outside [0, max_num_seqs).
        - negative q_offset.
        - LOGICAL prefill iter would read past its phys reqs scheduled
          q-slice (q_offset + static_q_len > n_R).
    """
    n_iters = plan.num_total
    if n_iters > max_num_subseqs:
        raise ValueError(
            f"plan has {n_iters} sub-seqs but max_num_subseqs is "
            f"{max_num_subseqs}; required_max_num_subseqs() "
            "under-sized the prefetch arrays for this workload.")

    phys_seq_indices = [0] * max_num_subseqs
    q_offsets = [0] * max_num_subseqs

    for i, entry in enumerate(plan.subseqs):
        phys = req_id_to_index.get(entry.real_req_id)
        if phys is None:
            raise ValueError(
                f"sub-seq #{i} real_req_id={entry.real_req_id!r} not in "
                "req_id_to_index — persistent batch / plan are out of "
                "sync.")
        if not 0 <= phys < max_num_seqs:
            raise ValueError(
                f"sub-seq #{i} phys slot {phys} outside [0, "
                f"{max_num_seqs}).")
        if entry.q_offset_in_real_req < 0:
            raise ValueError(
                f"sub-seq #{i} has negative q_offset "
                f"{entry.q_offset_in_real_req}.")

        if (static_q_len is not None
                and entry.kind is SubSeqKind.PREFILL):
            if num_scheduled_tokens is None:
                raise ValueError(
                    "num_scheduled_tokens is required when static_q_len "
                    "is set (LOGICAL bounds check).")
            n_R = num_scheduled_tokens.get(entry.real_req_id, 0)
            if entry.q_offset_in_real_req + static_q_len > n_R:
                raise ValueError(
                    f"sub-seq #{i} (PREFILL, real_req_id="
                    f"{entry.real_req_id!r}) would read past its phys "
                    f"reqs q-slice: q_offset="
                    f"{entry.q_offset_in_real_req} + static_q_len="
                    f"{static_q_len} > n_R={n_R}.")

        phys_seq_indices[i] = phys
        q_offsets[i] = entry.q_offset_in_real_req

    return IterPrefetches(phys_seq_indices=phys_seq_indices,
                          q_offsets=q_offsets)


def required_max_num_subseqs(*, max_num_seqs: int, K_sched_cap: int,
                             K_kernel: int) -> int:
    """Compute the prefetch-array sizing bound at runner init.

    Formula:
      max_num_subseqs = max_num_seqs * ceil(K_sched_cap / K_kernel) + max_num_seqs

    The two terms account for, respectively:
      - up to ``ceil(K_sched_cap / K_kernel)`` PREFILL sub-seqs per real
        request (when each request is given the full K_sched_cap budget
        and that budget is a multiple of K_kernel — this is the ceiling).
      - up to one MIXED sub-seq per real request (the remainder of its
        n_R when not a multiple of K_kernel, or the whole thing when
        n_R < K_kernel).

    The +max_num_seqs MIXED slack is a worst-case bound; when
    ``K_sched_cap % K_kernel == 0`` AND every request always produces a
    multiple-of-K_kernel scheduled token count, no remainder is ever
    produced and the slack is wasted. We over-allocate rather than
    track the actual case at runtime — prefetch-array sizes are tiny
    (KB-scale) so the slack is cheap, and a tighter bound would risk
    a silent overflow on the rare workload that produces remainders.

    DECODE sub-seqs are bounded by max_num_seqs but are mutually exclusive
    with PREFILL/MIXED sub-seqs of the same real request (a request is in
    decode OR prefill, not both, in any given step). So decode capacity is
    already covered by the existing max_num_seqs.

    NOTE: this formula gives the MATHEMATICAL bound. The PRACTICAL bound is
    almost always lower — the kernels SMEM budget caps
    ``max_num_subseqs * pages_per_seq`` (page_indices_ref dominates SMEM).
    Use ``compute_smem_bytes()`` and ``validate_smem_fits()`` below to
    verify a chosen ``max_num_subseqs`` fits the SMEM_LIMIT before
    allocating prefetch arrays.

    Returns:
      Required size for the prefetch arrays (kv_lens, cu_q_lens-1,
      page_indices/pages_per_seq, etc.) at runner init.
    """
    if K_kernel <= 1:
        raise ValueError(f"K_kernel must be > 1; got {K_kernel}")
    if K_sched_cap <= 0:
        raise ValueError(f"K_sched_cap must be > 0; got {K_sched_cap}")
    if max_num_seqs <= 0:
        raise ValueError(f"max_num_seqs must be > 0; got {max_num_seqs}")

    # Ceiling division
    max_chunks_per_req = (K_sched_cap + K_kernel - 1) // K_kernel
    return max_num_seqs * max_chunks_per_req + max_num_seqs


# Default conservative SMEM budget for the prefetch arrays. Mirrors the
# kernel tuners constraint at tools/kernel/tuner/v1/rpa_v3_kernel_tuner.py:96.
#
# IMPORTANT: this is a fallback. The actual hardware SMEM on v7x is
# ~1 MiB, queryable at runtime via:
#
#     pltpu.get_tpu_info().smem_capacity_bytes
#
# (See tpu_inference/kernels/experimental/batched_rpa/configs.py:181 for
# the canonical pattern, which subtracts 32 KiB for kernel internals.)
# Runner code should query the runtime API at init and pass the actual
# limit into validate_smem_fits / max_M_under_smem rather than relying
# on this conservative default. The default exists so this module can
# stay pure stdlib (no JAX import) for offline testing and for sweep
# recipe sizing. With a 0.9 MiB ceiling, achievable M_max values come
# out ~15-20% conservative compared to the real ~1 MiB hardware budget.
SMEM_LIMIT_BYTES: int = int(0.9 * 1024 * 1024)


# Reserve carved out of the SMEM hardware capacity for kernel internals
# (sem id arrays, distribution_ref, etc.) when the runtime API is used
# instead of the conservative SMEM_LIMIT_BYTES default. Single
# definition so the runner and any other caller agree.
# Matches the pattern in tpu_inference/kernels/experimental/
# batched_rpa/configs.py:181.
SMEM_RESERVE_BYTES: int = 32 * 1024


# Sub-chunks of the same real request currently DUPLICATE that requests
# pages_per_seq slots into page_indices_ref under todays kernel
# (kernel.py:568 indexes via seq_idx directly: page_indices_offset =
# seq_idx * pages_per_seq + kv_p_start). For sub-chunks of one request,
# every duplicated row holds identical contents — pure waste.
#
# Correctness note (read carefully if you wire this up): the kernel is
# unaware that the duplicated rows alias the same physical pages. It
# faithfully runs its full bkv loop for every seq_idx and writes new
# K/V at the offset implied by ``kv_q_gap = kv_lens_ref[seq_idx] -
# (cu_q_lens_ref[seq_idx+1] - cu_q_lens_ref[seq_idx])``. So duplicating
# page_indices alone is NOT enough — the runner MUST set kv_lens_ref
# and cu_q_lens_ref per sub-seq such that consecutive chunks of the
# same real request write to disjoint offset ranges within the shared
# pages. Concretely, for chunks 0, 1, 2 of a request R with K_kernel
# tokens each, prior_kv_len=p:
#   kv_lens_ref     = [p+K, p+2K, p+3K, ...]
#   cu_q_lens_ref   = [..., x, x+K, x+2K, x+3K, ...]   (q boundaries)
# Then chunk 0 writes [p, p+K), chunk 1 writes [p+K, p+2K), chunk 2
# writes [p+2K, p+3K). Get this wrong and the kernel silently corrupts
# KV cache.
#
# StepPlan emits this per-sub-seq metadata (q_offset_in_real_req +
# q_len + kv_len_at_end on each SubSeqEntry), so the planner already
# anticipates the constraint; the prefetch-array construction code
# (next commits) is where it has to land.
#
# A small kernel change unblocks much higher M_max:
#
#   1. Rename the loop variable from seq_idx (today, ranging
#      [0, max_num_subseqs)) to iter_idx (the same range, but the name
#      now reflects its meaning).
#   2. Add a new prefetch array:
#         real_seq_idx_ref: i32[max_num_subseqs]
#      LENGTH = max_num_subseqs (one entry per iteration).
#      VALUES are integers in [0, max_num_real_seqs) — the real-request
#      index for each iteration. E.g., if real seqs (R0, R1, R2, R3) get
#      sub-chunk counts (2, 1, 3, 1), the array holds [0, 0, 1, 2, 2, 2, 3].
#      Note: do NOT confuse the LENGTH with the VALUE RANGE — the array
#      is indexed by the loop variable (iter_idx), so its length must
#      match the iteration count (max_num_subseqs), even though the
#      values stored inside are real-seq IDs.
#   3. Change the page_indices_offset computation in the kernel:
#         real_idx = real_seq_idx_ref[iter_idx]
#         page_indices_offset = real_idx * pages_per_seq + kv_p_start
#   4. The other per-iteration prefetches (kv_lens_ref, cu_q_lens_ref)
#      stay sized at max_num_subseqs as today — they hold per-iteration
#      cumulative kv_len and per-iteration q boundaries, so their LENGTH
#      genuinely scales with iter count.
#
# This compresses page_indices_ref from
# (max_num_subseqs * pages_per_seq) to (max_num_real_seqs *
# pages_per_seq) — typically a ~30x SMEM reduction at M_max=32, since
# pages_per_seq dwarfs the 4-byte-per-iter cost of real_seq_idx_ref.
# PR3 (planned) lands this kernel change; PR2 wires up the runner with
# todays duplicated-page-indices layout and accepts the limited M_max.


def _align_to(value: int, multiple: int) -> int:
    """Ceiling-align value to the next multiple. Mirrors kernels align_to."""
    return ((value + multiple - 1) // multiple) * multiple


def compute_smem_bytes(*, max_num_seqs: int, pages_per_seq: int) -> int:
    """Estimate SMEM bytes consumed by the prefetch arrays.

    Mirrors the formula in
    ``tpu_inference/kernels/ragged_paged_attention/v3/kernel.py:208``
    (``get_smem_estimate_bytes``). page_indices_ref dominates the total;
    the other prefetch arrays (kv_lens, cu_q_lens, distribution, etc.)
    add a few KB at most.

    Under the sub-chunk model the PRACTICAL ``max_num_seqs`` here is the
    expanded value (= ``max_num_subseqs`` from
    ``required_max_num_subseqs``), since each sub-seq slot needs its own
    pages_per_seq entries — even though many sub-seqs of the same real
    request share physical pages. (The kernel does
    ``page_indices_ref[seq_idx * pages_per_seq + i]`` regardless of
    whether seq_idx is a real request or a sub-chunk; the duplication
    has SMEM cost.)

    Args:
      max_num_seqs: number of seq slots reserved in prefetch arrays.
        Under the sub-chunk model, pass the expanded sub-seq count, not
        the real-request count.
      pages_per_seq: ``ceil(max_model_len / page_size)``.

    Returns:
      Estimated SMEM bytes. Compare against SMEM_LIMIT_BYTES.
    """
    if max_num_seqs <= 0:
        raise ValueError(f"max_num_seqs must be > 0; got {max_num_seqs}")
    if pages_per_seq <= 0:
        raise ValueError(f"pages_per_seq must be > 0; got {pages_per_seq}")

    total_bits = (
        # kv_lens_ref: i32[max_num_seqs]
        _align_to(max_num_seqs, 128) * 32 +
        # page_indices_ref: i32[max_num_seqs * pages_per_seq] — DOMINANT
        _align_to(max_num_seqs * pages_per_seq, 128) * 32 +
        # cu_q_lens_ref: i32[max_num_seqs + 1]
        _align_to(max_num_seqs + 1, 128) * 32 +
        # distribution_ref: i32[3]    -> 128*32 (alignment)
        128 * 32 +
        # sem_ids_ref: i32[3]         -> 128*32
        128 * 32 +
        # bo_ids_ref: i32[4]          -> 128*32
        128 * 32 +
        # bkv_update_ids_ref: i32[6]  -> 128*32
        128 * 32)
    return (total_bits + 7) // 8


def validate_smem_fits(*, max_num_subseqs: int, pages_per_seq: int,
                       smem_limit_bytes: int = SMEM_LIMIT_BYTES) -> None:
    """Assert that the chosen sub-seq sizing fits in SMEM.

    Used at runner init, after ``required_max_num_subseqs()`` produces a
    nominal sub-seq count from K_sched_cap and K_kernel. If the nominal
    count blows SMEM, the caller must either (a) reduce K_sched_cap, (b)
    reduce MAX_NUM_SEQS, or (c) choose a larger page_size to shrink
    pages_per_seq. SMEM cannot be expanded at runtime.

    Raises:
      ValueError: SMEM estimate exceeds the limit. Message includes the
        offending values and the limit so the misconfiguration is
        diagnosable from logs.
    """
    bytes_used = compute_smem_bytes(max_num_seqs=max_num_subseqs,
                                    pages_per_seq=pages_per_seq)
    if bytes_used > smem_limit_bytes:
        raise ValueError(
            f"SMEM budget exceeded under the sub-chunk model: "
            f"max_num_subseqs={max_num_subseqs}, "
            f"pages_per_seq={pages_per_seq}, "
            f"estimated SMEM bytes={bytes_used} > limit={smem_limit_bytes}. "
            f"Reduce K_sched_cap, MAX_NUM_SEQS, or use a larger page_size.")


def max_M_under_smem(*, max_num_seqs: int, pages_per_seq: int,
                     smem_limit_bytes: int = SMEM_LIMIT_BYTES) -> int:
    """Largest M (chunks-per-real-request bound) that fits in SMEM, given
    ``max_num_seqs`` real requests and ``pages_per_seq`` per-seq pages.

    Solves ``compute_smem_bytes(max_num_seqs * (M + 1), pages_per_seq) <=
    smem_limit_bytes`` for the largest integer M where (M + 1) accounts
    for M PREFILL chunks plus 1 MIXED remainder slot per real request.
    Returns 0 if even the today-equivalent layout (no chunking) does not
    fit, which would itself be a misconfiguration.

    Used by the runner / sweep recipes to pick a K_sched_cap that does
    not exceed the achievable M for a given workloads (max_model_len,
    page_size, MAX_NUM_SEQS) shape. K_sched_cap = M * K_kernel.
    """
    # Linear search bounded by what could plausibly fit; SMEM is small
    # enough that this is fast. (M + 1) accounts for M PREFILL chunks
    # plus 1 MIXED remainder slot per real request — same formula as
    # required_max_num_subseqs.
    M = 0
    SAFETY_CAP = 1024
    while True:
        candidate = max_num_seqs * (M + 1)
        bytes_used = compute_smem_bytes(max_num_seqs=candidate,
                                        pages_per_seq=pages_per_seq)
        if bytes_used > smem_limit_bytes:
            return max(M - 1, 0)
        M += 1
        # Safety bound: M cannot exceed K_sched_cap / K_kernel for any
        # reasonable config; cap the search to avoid a runaway loop on
        # a misconfigured smem_limit_bytes (e.g. someone passes
        # sys.maxsize). If we hit the cap, warn — at this point either
        # the budget is wrong or the workload geometry is unusual
        # enough that the caller should be aware.
        if M >= SAFETY_CAP:
            warnings.warn(
                f"max_M_under_smem hit safety cap of {SAFETY_CAP} "
                f"at max_num_seqs={max_num_seqs}, "
                f"pages_per_seq={pages_per_seq}, "
                f"smem_limit_bytes={smem_limit_bytes}. Likely a "
                f"misconfigured smem_limit_bytes; the cap is "
                f"intentionally well above any plausible real M.",
                stacklevel=2)
            return SAFETY_CAP - 1


@dataclasses.dataclass(frozen=True)
class DecoupledKConfig:
    """Resolved configuration for the decoupled-K runner path.

    Produced by ``evaluate_decoupled_k_config()`` from the raw
    K_sched / K_kernel / SMEM budget inputs.

    Attributes:
      K_kernel: the static-q-len of the PREFILL kernel pass. Carried
        over from the input so consumers (PersistentBatchManager,
        AttentionMetadata builders) can reach it from a single
        DecoupledKConfig handle without reading envs.RPA_KERNEL_K
        again.
      K_sched_effective: the actual per-request-per-step token budget
        used to compute requested_M. Equals
        ``min(K_sched, max_num_batched_tokens) if K_sched > 0 else
        max_num_batched_tokens``. Clamping by MNB matters because vLLM
        cannot hand a single request more tokens than its total
        per-step budget — without the clamp, requested_M overestimates
        and the diagnostic warning misreports the effective scheduler
        cap.
      requested_M: ``ceil(K_sched_effective / K_kernel)``  — the
        chunks-per-request-per-step the user asked for.
      achievable_M: largest M that fits in SMEM at the given workload
        geometry. Computed by ``max_M_under_smem()``.
      effective_M: ``min(requested_M, achievable_M)``. The runner caps
        sub-chunk count per request at this value per step. When
        ``requested_exceeds_achievable`` is True (achievable_M <
        requested_M), the **caller** is expected to log a warning;
        this dataclass merely flags the condition via the property.
        ``evaluate_decoupled_k_config`` does not warn directly because
        consumers may want to format the message with their own
        context (workload name, log level, etc.).
      effective_K_sched_cap: ``effective_M * K_kernel``. The maximum
        per-request-per-step token count the runner will actually
        process in chunked form. If vLLM hands us more than this
        (because its own LONG_PREFILL_TOKEN_THRESHOLD is higher), the
        excess goes to MIXED.
      smem_budget: the SMEM ceiling used for this evaluation. Either
        from the runtime API (``pltpu.get_tpu_info().smem_capacity_bytes
        - reserve``) or the conservative default
        (``SMEM_LIMIT_BYTES``).
      pages_per_seq: ``ceil(max_model_len / page_size)`` carried over so
        downstream prefetch-array sizing is consistent.
    """
    K_kernel: int
    K_sched_effective: int
    requested_M: int
    achievable_M: int
    effective_M: int
    effective_K_sched_cap: int
    smem_budget: int
    pages_per_seq: int

    @property
    def requested_exceeds_achievable(self) -> bool:
        """True when SMEM forces clamping to a lower M than requested."""
        return self.requested_M > self.achievable_M


def evaluate_decoupled_k_config(
    *,
    K_kernel: int,
    K_sched: int,
    max_num_batched_tokens: int,
    max_num_seqs: int,
    max_model_len: int,
    page_size: int,
    smem_budget: int = SMEM_LIMIT_BYTES,
) -> DecoupledKConfig:
    """Validate and resolve the decoupled-K config at runner init.

    Inputs come from vLLM scheduler/cache config plus the runtime SMEM
    budget. Output is a DecoupledKConfig the runner stores and uses
    when building per-step prefetch arrays.

    Semantics:
      - K_sched == 0 (vLLM default for "no per-request cap"): use
        max_num_batched_tokens as the worst-case per-request bound,
        since vLLM could in principle hand a single request the full
        MNB budget.
      - K_sched > 0: use that as the per-request cap.

    Always-on hard guard: even at M=1 (no chunking), the prefetch
    arrays must fit in SMEM. If not, the (max_num_seqs, max_model_len,
    page_size) triple is fundamentally incompatible with this hardware
    and the runner should fail loudly rather than silently truncate.

    Returns:
      DecoupledKConfig with resolved effective_M, effective_K_sched_cap,
      and the inputs carried through.

    Raises:
      ValueError: K_kernel <= 1, K_kernel does not divide cleanly into
        a useful range, or even M=1 fails the SMEM budget.
    """
    if K_kernel <= 1:
        raise ValueError(f"K_kernel must be > 1; got {K_kernel}")
    if max_num_batched_tokens <= 0:
        raise ValueError(
            f"max_num_batched_tokens must be > 0; got {max_num_batched_tokens}")
    if max_num_seqs <= 0:
        raise ValueError(f"max_num_seqs must be > 0; got {max_num_seqs}")
    if max_model_len <= 0:
        raise ValueError(f"max_model_len must be > 0; got {max_model_len}")
    if page_size <= 0:
        raise ValueError(f"page_size must be > 0; got {page_size}")

    # ceiling(max_model_len / page_size)
    pages_per_seq = (max_model_len + page_size - 1) // page_size

    # Hard guard: M=1 (no chunking) must fit. If not, the workload
    # geometry is unsupported regardless of K_sched. Wrap the generic
    # SMEM error with a more actionable message — at M=1, lowering
    # K_sched_cap (which is the underlying validate_smem_fits message)
    # cannot help since were already at the minimum.
    try:
        validate_smem_fits(
            max_num_subseqs=required_max_num_subseqs(
                max_num_seqs=max_num_seqs, K_sched_cap=K_kernel,
                K_kernel=K_kernel),
            pages_per_seq=pages_per_seq,
            smem_limit_bytes=smem_budget)
    except ValueError as e:
        raise ValueError(
            f"Workload geometry incompatible with hardware SMEM: even "
            f"M=1 (no sub-chunk inflation) overflows the SMEM budget. "
            f"max_num_seqs={max_num_seqs}, max_model_len={max_model_len}, "
            f"page_size={page_size}, pages_per_seq={pages_per_seq}, "
            f"smem_budget={smem_budget}. Actionable knobs: reduce "
            f"max_num_seqs, increase page_size (smaller pages_per_seq), "
            f"or reduce max_model_len. Reducing K_sched does NOT help "
            f"at this regime. Underlying: {e}") from e

    # K_sched=0 in vLLM means "no per-request cap" — bound by MNB.
    # When K_sched > MNB, vLLM cant actually grant the user-requested
    # cap (per-step token budget is bounded by MNB across all reqs and
    # certainly bounded by it for any single req), so clamp.
    if K_sched > 0:
        K_sched_effective = min(K_sched, max_num_batched_tokens)
    else:
        K_sched_effective = max_num_batched_tokens
    requested_M = (K_sched_effective + K_kernel - 1) // K_kernel

    achievable_M = max_M_under_smem(max_num_seqs=max_num_seqs,
                                    pages_per_seq=pages_per_seq,
                                    smem_limit_bytes=smem_budget)

    effective_M = min(requested_M, achievable_M)
    effective_K_sched_cap = effective_M * K_kernel

    return DecoupledKConfig(
        K_kernel=K_kernel,
        K_sched_effective=K_sched_effective,
        requested_M=requested_M,
        achievable_M=achievable_M,
        effective_M=effective_M,
        effective_K_sched_cap=effective_K_sched_cap,
        smem_budget=smem_budget,
        pages_per_seq=pages_per_seq,
    )
