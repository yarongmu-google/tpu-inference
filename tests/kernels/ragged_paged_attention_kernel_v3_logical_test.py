# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Phase 3 tests for the LOGICAL (decoupled-K) ragged-paged-attention kernel.

Two test classes:

  TestLogicalEquivalence
      Black-box equivalence: ragged_paged_attention(..., phys_seq_indices=,
      q_offsets=) output matches ref_ragged_paged_attention_logical for a
      handful of (num_decode, num_logical_prefill, num_mixed) mixes. This
      is the primary correctness check before tuning.

  TestLogicalKvCacheRaceFix
      The "evil" test for §6.4 Option A: pre-corrupt kv_cache at the
      LOGICAL iters write window (positions
      [phys_kv_q_gap, phys_kv_q_gap + in_step_count)) and verify the
      kernel still produces correct output. If §6.4 is violated — i.e.
      the kernel reads from kv_cache instead of merged_kv at those
      positions — the corruption bleeds into the output and the test
      fails. Without this test, a regression of §6.4 would be invisible
      because the kernel still WRITES correct values to the cache, so
      end-state cache equivalence with the reference would still hold.

Both tests skip on non-TPU. TODO: thread interpret=pltpu.InterpretParams()
through the kernel so these tests can also run on the laptop (per
memory feedback_pallas_interpret_testing.md).
"""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu

from tpu_inference.kernels.ragged_paged_attention.v3.kernel import (
    ragged_paged_attention, ref_ragged_paged_attention_logical)
from tpu_inference.kernels.ragged_paged_attention.v3.util import (
    align_to, cdiv, get_dtype_packing)

jax.config.parse_flags_with_absl()


def _build_logical_workload(
    *,
    seqs,                 # list[(prior_kv_len, q_len, kind)] kind in {"D","L","M"}
    K_kernel,             # int — static_q_len for LOGICAL iters
    num_q_heads,
    num_kv_heads,
    head_dim,
    page_size,
    num_pages,
    q_dtype,
    kv_dtype,
    rng,
    cache_corruptor=None,  # optional callable(kv_cache, info) -> kv_cache
):
    """Materialise a LOGICAL workload into kernel-ready inputs.

    Each seq is one phys request. "D" emits one DECODE iter (q_len=1),
    "L" emits ceil(q_len/K_kernel) LOGICAL iters (each q_len=K_kernel for
    the chunks; tail goes to MIXED if not divisible), "M" emits one MIXED
    iter with the full q_len.

    For simplicity we constrain "L" seqs to have q_len % K_kernel == 0
    (no MIXED tail) — testing the chunking + the chunk/tail interaction
    are separate concerns.

    Returns a dict of (q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens,
    distribution, phys_seq_indices, q_offsets) plus auxiliary info needed
    by the evil test (per-phys page_indices_for_seq, phys_kv_q_gap).
    """
    # --- Phys-keyed bookkeeping ---
    # cu_q_lens (phys-keyed, len = num_phys + 1)
    # kv_lens   (phys-keyed, len = num_phys); = prior + q_len for each
    cu_q_lens = [0]
    phys_kv_lens = []           # full kv_len AFTER this step writes
    phys_kv_q_gaps = []         # = prior_kv_len
    for prior, q_len, _kind in seqs:
        cu_q_lens.append(cu_q_lens[-1] + q_len)
        phys_kv_lens.append(prior + q_len)
        phys_kv_q_gaps.append(prior)
    num_phys = len(seqs)
    total_q_tokens = cu_q_lens[-1]

    # --- Iter ordering: D first, then L, then M (matches kernel) ---
    decode_iters = []     # list[(phys_idx, q_offset)]
    logical_iters = []
    mixed_iters = []
    for phys_idx, (_prior, q_len, kind) in enumerate(seqs):
        if kind == "D":
            assert q_len == 1, "DECODE seq must have q_len==1"
            decode_iters.append((phys_idx, 0))
        elif kind == "L":
            assert q_len % K_kernel == 0, (
                f"LOGICAL seq q_len={q_len} must be divisible by K_kernel="
                f"{K_kernel} in this helper")
            for off in range(0, q_len, K_kernel):
                logical_iters.append((phys_idx, off))
        elif kind == "M":
            mixed_iters.append((phys_idx, 0))
        else:
            raise ValueError(f"Unknown kind {kind!r}")

    iter_phys = [p for (p, _) in decode_iters + logical_iters + mixed_iters]
    iter_qoff = [o for (_, o) in decode_iters + logical_iters + mixed_iters]
    num_decode = len(decode_iters)
    num_logical = len(logical_iters)
    num_mixed = len(mixed_iters)
    num_total = num_decode + num_logical + num_mixed

    # --- Sizes / padding ---
    max_num_seqs = max(align_to(num_phys, 8), 8)
    max_num_subseqs = max(align_to(num_total, 8), 8)
    max_kv = max(phys_kv_lens) if phys_kv_lens else page_size
    pages_per_seq = cdiv(max_kv, page_size)
    max_num_batched_tokens = max(align_to(total_q_tokens, 128), 128)

    # --- Q / K / V buffers (ragged; positions per phys are
    # cu_q_lens[phys] : cu_q_lens[phys+1]). ---
    def gen(shape, dtype):
        return jnp.array(rng.random(size=shape,
                                    dtype=np.float32)).astype(dtype)

    q = gen((max_num_batched_tokens, num_q_heads, head_dim), q_dtype)
    k = gen((max_num_batched_tokens, num_kv_heads, head_dim), kv_dtype)
    v = gen((max_num_batched_tokens, num_kv_heads, head_dim), kv_dtype)

    # --- KV cache + page_indices ---
    kv_packing = get_dtype_packing(kv_dtype)
    padded_head_dim = align_to(head_dim, 128)
    num_kv_heads_x2 = align_to(num_kv_heads * 2, kv_packing)

    # NaN-init matches the existing coupled-K v3 test convention. NaN at
    # cache positions the kernel must NOT read (per §6.4 Option A: the
    # iter-write window) is a stronger test — a future regression where
    # the kernel mistakenly reads from cache at those positions would
    # propagate NaN into the output. We pre-populate positions [0, prior)
    # below for each phys req with finite prior content (legitimately
    # readable from cache); everything else stays NaN.
    kv_cache = jnp.full(
        (num_pages, page_size, num_kv_heads_x2 // kv_packing, kv_packing,
         padded_head_dim),
        jnp.nan,
        dtype=kv_dtype,
    )

    # Per-phys page indices: each phys gets pages_per_seq contiguous pages.
    # Pages 0..num_phys*pages_per_seq-1 are "live"; rest are padding.
    page_indices_for_phys = []
    page_cursor = 0
    for phys_idx in range(num_phys):
        idxs = page_cursor + np.arange(pages_per_seq, dtype=np.int32)
        page_indices_for_phys.append(idxs)
        page_cursor += pages_per_seq

    # Pre-populate "prior" cache content for each phys at positions
    # [0, phys_kv_q_gap). The kernel reads those from the cache (legit
    # pre-step content); both reference and kernel must see the same
    # values there.
    for phys_idx, prior in enumerate(phys_kv_q_gaps):
        if prior == 0:
            continue
        prior_kv = gen((prior, num_kv_heads_x2 // kv_packing, kv_packing,
                        padded_head_dim), kv_dtype)
        # Write into the phys's pages, position-by-position.
        for pos in range(prior):
            page_idx_in_seq = pos // page_size
            slot_in_page = pos % page_size
            global_page = int(page_indices_for_phys[phys_idx][page_idx_in_seq])
            kv_cache = kv_cache.at[global_page, slot_in_page].set(
                prior_kv[pos])

    # Optional cache corruptor for the evil test. Receives the cache and
    # a dict with bookkeeping; returns the (possibly mutated) cache.
    if cache_corruptor is not None:
        kv_cache = cache_corruptor(
            kv_cache,
            dict(page_indices_for_phys=page_indices_for_phys,
                 phys_kv_q_gaps=phys_kv_q_gaps,
                 phys_kv_lens=phys_kv_lens,
                 page_size=page_size))

    # Pad page_indices to (max_num_seqs, pages_per_seq) and flatten.
    page_indices_padded = np.zeros(
        (max_num_seqs, pages_per_seq), dtype=np.int32)
    for phys_idx, idxs in enumerate(page_indices_for_phys):
        page_indices_padded[phys_idx] = idxs
    page_indices = jnp.array(page_indices_padded.reshape(-1))

    cu_q_lens_padded = np.zeros(max_num_seqs + 1, dtype=np.int32)
    cu_q_lens_padded[:len(cu_q_lens)] = cu_q_lens
    cu_q_lens_padded[len(cu_q_lens):] = cu_q_lens[-1]
    cu_q_lens_arr = jnp.array(cu_q_lens_padded)

    kv_lens_padded = np.zeros(max_num_seqs, dtype=np.int32)
    kv_lens_padded[:num_phys] = phys_kv_lens
    kv_lens_arr = jnp.array(kv_lens_padded)

    distribution = jnp.array(
        [num_decode, num_decode + num_logical, num_total], dtype=jnp.int32)

    phys_seq_indices_padded = np.zeros(max_num_subseqs, dtype=np.int32)
    phys_seq_indices_padded[:num_total] = iter_phys
    phys_seq_indices = jnp.array(phys_seq_indices_padded)

    q_offsets_padded = np.zeros(max_num_subseqs, dtype=np.int32)
    q_offsets_padded[:num_total] = iter_qoff
    q_offsets = jnp.array(q_offsets_padded)

    return dict(
        q=q, k=k, v=v, kv_cache=kv_cache,
        kv_lens=kv_lens_arr, page_indices=page_indices,
        cu_q_lens=cu_q_lens_arr, distribution=distribution,
        phys_seq_indices=phys_seq_indices, q_offsets=q_offsets,
        # Aux for the evil test:
        page_indices_for_phys=page_indices_for_phys,
        phys_kv_q_gaps=phys_kv_q_gaps,
        phys_kv_lens=phys_kv_lens,
        total_q_tokens=total_q_tokens,
        num_logical_iters=num_logical,
    )


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class TestLogicalEquivalence(jtu.JaxTestCase):
    """Black-box equivalence: kernel output matches reference within
    bf16 tolerance across a few workload mixes."""

    def _run_and_assert(self, *, seqs, K_kernel, q_dtype=jnp.bfloat16,
                        kv_dtype=jnp.bfloat16):
        if not jtu.is_device_tpu_at_least(version=4):
            self.skipTest("Expect TPUv4+")
        rng = np.random.default_rng(2026)
        w = _build_logical_workload(
            seqs=seqs, K_kernel=K_kernel,
            num_q_heads=8, num_kv_heads=2, head_dim=128,
            page_size=16, num_pages=512,
            q_dtype=q_dtype, kv_dtype=kv_dtype, rng=rng)

        # Reference: pure-JAX LOGICAL implementation.
        expected, _ = ref_ragged_paged_attention_logical(
            queries=w["q"], keys=w["k"], values=w["v"],
            kv_cache=w["kv_cache"], kv_lens=w["kv_lens"],
            page_indices=w["page_indices"], cu_q_lens=w["cu_q_lens"],
            phys_seq_indices=w["phys_seq_indices"],
            q_offsets=w["q_offsets"],
            distribution=w["distribution"],
            static_q_len=K_kernel)

        # Kernel: standard public entry. distribution[0] = num_decode,
        # distribution[1] = num_decode + num_logical, etc. — when num_decode
        # > 0 the LOGICAL pass slices iters [num_decode, num_decode+num_L),
        # which is what the reference walks.
        output, _ = ragged_paged_attention(
            w["q"], w["k"], w["v"], w["kv_cache"],
            w["kv_lens"], w["page_indices"], w["cu_q_lens"],
            w["distribution"],
            w["phys_seq_indices"], w["q_offsets"],
            chunk_prefill_size=K_kernel,
            p_block_sizes=(64, 128, 32, 64),
            d_block_sizes=(64, 128, 32, 64),
            m_block_sizes=(64, 128, 32, 64),
        )
        # Reference returns only LOGICAL iters concatenated; kernel returns
        # the full ragged buffer. Slice the kernel output to match.
        # LOGICAL output spans cu_q_lens[num_decode] :
        # cu_q_lens[num_decode + num_L_phys_seqs] in the q-token layout.
        # For our helper, all LOGICAL iters of one phys are contiguous in
        # q-tokens, and the reference concatenates them in iter order
        # (which matches the q-token layout for LOGICAL-only seqs).
        L_start = int(w["distribution"][0])  # = num_decode iters
        L_end = int(w["distribution"][1])
        # Map iter range -> q-token range via phys_seq_indices/q_offsets.
        # For our happy-path tests, LOGICAL seqs occupy a contiguous q-token
        # range [cu_q_lens[first_L_phys] : cu_q_lens[last_L_phys + 1]).
        first_L_phys = int(w["phys_seq_indices"][L_start])
        last_L_phys = int(w["phys_seq_indices"][L_end - 1])
        q_lo = int(w["cu_q_lens"][first_L_phys])
        q_hi = int(w["cu_q_lens"][last_L_phys + 1])
        kernel_logical_output = output[q_lo:q_hi]

        self.assertEqual(kernel_logical_output.shape, expected.shape)
        # bf16 tolerance per the existing v3 test convention.
        self.assertAllClose(kernel_logical_output, expected, atol=0.2,
                            rtol=0.2)

    @parameterized.parameters(
        # Single phys, 2 LOGICAL chunks.
        dict(seqs=[(0, 8, "L")], K_kernel=4),
        # Single phys, 4 LOGICAL chunks.
        dict(seqs=[(0, 16, "L")], K_kernel=4),
        # Two phys, each chunked.
        dict(seqs=[(0, 8, "L"), (0, 8, "L")], K_kernel=4),
    )
    def test_logical_equivalence_decode_free(self, seqs, K_kernel):
        """LOGICAL-only workloads (no DECODE / MIXED iters)."""
        self._run_and_assert(seqs=seqs, K_kernel=K_kernel)

    def test_logical_equivalence_with_prior_kv(self):
        """LOGICAL iter on a phys req with prior_kv_len > 0 — exercises
        the cache-vs-merged_kv boundary at offset phys_kv_q_gap."""
        self._run_and_assert(
            seqs=[(8, 8, "L")],   # prior=8, this-step q=8 -> 2 chunks
            K_kernel=4)


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class TestLogicalKvCacheRaceFix(jtu.JaxTestCase):
    """Evil test for §6.4 Option A: pre-corrupt kv_cache at every LOGICAL
    iter's write window (positions [phys_kv_q_gap, phys_kv_q_gap +
    in_step_count) per phys) and verify the kernel produces the SAME
    output as the reference run on the same corrupted cache.

    Both kernel and reference receive the corrupted cache; if both
    correctly implement §6.4, both ignore the corruption and read from
    merged_kv. If the kernel regresses (reads from cache instead), the
    corruption bleeds into kernel output and the assertion fails.

    Cache layout: positions [0, phys_kv_q_gap) hold legitimate prior-step
    content (read by both). Positions [phys_kv_q_gap, phys_kv_q_gap +
    in_step_count) are CORRUPTED (must be ignored by both — they are this
    steps own-iter write window where merged_kv is the source of truth).
    """

    def _corruptor(self, kv_cache, info):
        # Replace cache at each phys reqs iter-write window with a large
        # finite "obviously wrong" value (avoiding NaN so we measure
        # numerical drift, not NaN propagation).
        page_indices_for_phys = info["page_indices_for_phys"]
        phys_kv_q_gaps = info["phys_kv_q_gaps"]
        phys_kv_lens = info["phys_kv_lens"]
        page_size = info["page_size"]
        cache = kv_cache
        for phys_idx, (qgap, kv_len) in enumerate(
                zip(phys_kv_q_gaps, phys_kv_lens)):
            for pos in range(qgap, kv_len):
                page_idx_in_seq = pos // page_size
                slot_in_page = pos % page_size
                global_page = int(
                    page_indices_for_phys[phys_idx][page_idx_in_seq])
                # Use a finite "evil" sentinel: 1e3 mapped through dtype.
                evil = jnp.full(cache.shape[2:], 1e3, dtype=cache.dtype)
                cache = cache.at[global_page, slot_in_page].set(evil)
        return cache

    def test_evil_corruption_at_iter_write_window(self):
        """LOGICAL iter on phys with prior_kv_len > 0. Cache at
        [phys_kv_q_gap, kv_len) is corrupted. Kernel + reference both
        ignore it (read merged_kv) and produce identical output."""
        if not jtu.is_device_tpu_at_least(version=4):
            self.skipTest("Expect TPUv4+")

        rng = np.random.default_rng(2026)
        seqs = [(8, 8, "L")]   # prior_kv_len=8, q_len=8 -> 2 LOGICAL iters
        K_kernel = 4

        w = _build_logical_workload(
            seqs=seqs, K_kernel=K_kernel,
            num_q_heads=8, num_kv_heads=2, head_dim=128,
            page_size=16, num_pages=512,
            q_dtype=jnp.bfloat16, kv_dtype=jnp.bfloat16, rng=rng,
            cache_corruptor=self._corruptor)

        # Reference output on corrupted cache: §6.4 Option A in the
        # reference IGNORES cache at [phys_kv_q_gap, kv_len) and reads
        # from merged_kv, so this is the "correct" output.
        expected, _ = ref_ragged_paged_attention_logical(
            queries=w["q"], keys=w["k"], values=w["v"],
            kv_cache=w["kv_cache"], kv_lens=w["kv_lens"],
            page_indices=w["page_indices"], cu_q_lens=w["cu_q_lens"],
            phys_seq_indices=w["phys_seq_indices"],
            q_offsets=w["q_offsets"],
            distribution=w["distribution"],
            static_q_len=K_kernel)

        output, _ = ragged_paged_attention(
            w["q"], w["k"], w["v"], w["kv_cache"],
            w["kv_lens"], w["page_indices"], w["cu_q_lens"],
            w["distribution"],
            w["phys_seq_indices"], w["q_offsets"],
            chunk_prefill_size=K_kernel,
            p_block_sizes=(64, 128, 32, 64),
            d_block_sizes=(64, 128, 32, 64),
            m_block_sizes=(64, 128, 32, 64),
        )
        L_start = int(w["distribution"][0])
        L_end = int(w["distribution"][1])
        first_L_phys = int(w["phys_seq_indices"][L_start])
        last_L_phys = int(w["phys_seq_indices"][L_end - 1])
        q_lo = int(w["cu_q_lens"][first_L_phys])
        q_hi = int(w["cu_q_lens"][last_L_phys + 1])
        kernel_logical_output = output[q_lo:q_hi]

        self.assertEqual(kernel_logical_output.shape, expected.shape)
        self.assertAllClose(kernel_logical_output, expected, atol=0.2,
                            rtol=0.2)
        # Sanity: output must be finite. NaN here would mean the kernel
        # read the evil sentinel and propagated through softmax.
        self.assertTrue(jnp.all(jnp.isfinite(kernel_logical_output)))


if __name__ == "__main__":
    absltest.main()
