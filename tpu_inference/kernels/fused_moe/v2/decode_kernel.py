"""Decode-specific fused MoE kernel (TP, VMEM-resident tokens).

DESIGN PREMISE: MoE is
token-isolated - no cross-token state, no KV walk - so a DECODE step's
entire activation working set is [num_tokens, hidden] (~4 MB at 512x4096
bf16), which fits VMEM. That flips the regime vs prefill:

  decode  (this kernel): tokens stay VMEM-resident for the whole layer;
      dispatch = per-expert row lists consumed by vector loads (no DMA
      descriptors); weights stream ONCE per expert (m ~= B*k/E rows each,
      no reuse possible); the layer is weight-stream-bound and everything
      else must hide under it.
  prefill (NOT this kernel): activations outgrow VMEM; the right shape is
      expert-sorted HBM gathers with same-expert weight-tile caching (see
      kernels/experimental/fused_moe - gmm_fused_rs).

Follows the ragged_paged_attention v3 convention: versioned dir, regime
split explicit; a wrapper can dispatch on num_tokens.

Pipeline (single pallas_call, grid = expert blocks; stage names are
jax.named_scope spans -> hardware trace markers for per-stage xprof):
  routing   (grid step 0): scoring + top-k (argmax-free maxmask: winner by
            value-equality vs the broadcast max - no bf16-argmax gap) ->
            per-expert row lists + gate weights via ONE scalar pass over
            the (token, slot) pairs (B*k iterations - Rupeng-structured,
            not the per-expert rescans of v1).
  gather    per expert: C rows staged from the VMEM-resident tokens by
            scalar-indexed dynamic slices (vector loads, zero DMA).
  gmm1+act  [C, D] @ [D, 2I] -> SwiGLU -> [C, I]
  gmm2      [C, I] @ [I, D] -> [C, D], rows scaled by gate weights
  combine   scatter-add into a VMEM accumulator [T, D] (f32)
  epilogue  accumulator -> output (partial under TP; caller reduce-scatters)

v0 scope: bf16 weights/acts, no shared expert, no in-kernel RS, capacity
C rows/expert with overflow DROPPED (assert-checked in tests; production
needs the spill path from the proposal doc).
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _routing(gating_ref, top_k, renormalize, num_experts):
    """Scoring + argmax-free top-k. Returns (weights[T,k], indices[T,k])."""
    with jax.named_scope("moe_score"):
        scores = jax.nn.softmax(gating_ref[...].astype(jnp.float32), axis=-1)
    t = scores.shape[0]
    iota = lax.broadcasted_iota(jnp.int32, scores.shape, 1)
    valid = iota < num_experts
    x = jnp.where(valid, scores, -jnp.inf)
    weights = []
    indices = []
    for k_id in range(top_k):
        with jax.named_scope(f"moe_topk{k_id}"):
            mx = jnp.max(x, axis=1, keepdims=True)
            win = (x == jnp.broadcast_to(mx, x.shape)) & valid
            idx = jnp.max(jnp.where(win, iota, 0), axis=1, keepdims=True)
            sel = iota == jnp.broadcast_to(idx, x.shape)
            weights.append(mx[:, 0])
            indices.append(idx[:, 0])
            if k_id != top_k - 1:
                x = jnp.where(sel, -jnp.inf, x)
    w = jnp.stack(weights, axis=1)  # [T, k]
    i = jnp.stack(indices, axis=1)  # [T, k]
    if renormalize:
        w = w / jnp.sum(w, axis=1, keepdims=True)
    return w, i


def _decode_moe_kernel(
    # inputs
    tokens_ref,      # [T, D] VMEM-resident (grid-invariant)
    gating_ref,      # [T, E] VMEM-resident
    w1_ref,          # [NE, D, 2I] this grid step's expert block
    w2_ref,          # [NE, I, D]
    # outputs
    out_ref,         # [T, D]
    # scratch
    acc_ref,         # VMEM [T, D] f32 accumulator
    rows_ref,        # SMEM [E, C] i32 per-expert row lists
    gates_ref,       # SMEM [E, C] f32 per-row gate weights
    counts_ref,      # SMEM [E] i32
    gathered_ref,    # VMEM [C, D] staging
    *,
    num_experts: int,
    experts_per_block: int,
    capacity: int,
    top_k: int,
    renormalize: bool,
    act_dtype,
):
    blk = pl.program_id(0)
    num_blocks = pl.num_programs(0)
    t = tokens_ref.shape[0]

    # ---- grid step 0: routing + list building + accumulator init ----
    @pl.when(blk == 0)
    def _prologue():
        acc_ref[...] = jnp.zeros_like(acc_ref)
        counts_zero = jnp.zeros((num_experts,), jnp.int32)
        with jax.named_scope("moe_routing"):
            weights, indices = _routing(gating_ref, top_k, renormalize,
                                        num_experts)
        # ONE scalar pass over the (token, slot) pairs: B*k iterations
        # (Rupeng-structured; v1's per-expert rescan would be B*k*E/8).
        with jax.named_scope("moe_lists"):
            for e in range(num_experts):
                counts_ref[e] = 0

            def _pair(p, carry):
                tok = p // top_k
                k_id = p % top_k
                e = indices[tok, k_id]
                c = counts_ref[e]

                @pl.when(c < capacity)
                def _do():
                    rows_ref[e, c] = tok
                    gates_ref[e, c] = weights[tok, k_id]
                    counts_ref[e] = c + 1

                return carry

            lax.fori_loop(0, t * top_k, _pair, 0)

    # ---- per expert in this block: gather -> gmm1 -> act -> gmm2 -> add ----
    for le in range(experts_per_block):
        e = blk * experts_per_block + le
        cnt = counts_ref[e]

        @pl.when(cnt > 0)
        def _process(le=le, e=e, cnt=cnt):
            with jax.named_scope("moe_gather"):
                def _g(c, carry):
                    row = rows_ref[e, c]

                    @pl.when(c < cnt)
                    def _do():
                        gathered_ref[pl.ds(c, 1), :] = tokens_ref[
                            pl.ds(row, 1), :]

                    @pl.when(c >= cnt)
                    def _pad():
                        gathered_ref[pl.ds(c, 1), :] = jnp.zeros(
                            (1, tokens_ref.shape[1]), act_dtype)

                    return carry

                lax.fori_loop(0, capacity, _g, 0)

            with jax.named_scope("moe_gmm1"):
                x = gathered_ref[...]                       # [C, D]
                w1 = w1_ref[le]                             # [D, 2I]
                h = jnp.dot(x, w1,
                            preferred_element_type=jnp.float32)  # [C, 2I]
                i_half = h.shape[-1] // 2
                gate, up = h[:, :i_half], h[:, i_half:]
                a = (jax.nn.silu(gate) * up).astype(act_dtype)   # [C, I]

            with jax.named_scope("moe_gmm2"):
                w2 = w2_ref[le]                             # [I, D]
                y = jnp.dot(a, w2,
                            preferred_element_type=jnp.float32)  # [C, D]

            with jax.named_scope("moe_combine"):
                def _s(c, carry):
                    @pl.when(c < cnt)
                    def _do():
                        row = rows_ref[e, c]
                        g = gates_ref[e, c]
                        acc_ref[pl.ds(row, 1), :] = (
                            acc_ref[pl.ds(row, 1), :]
                            + g * y[pl.ds(c, 1), :])

                    return carry

                lax.fori_loop(0, capacity, _s, 0)

    # ---- last grid step: emit ----
    @pl.when(blk == num_blocks - 1)
    def _epilogue():
        out_ref[...] = acc_ref[...].astype(out_ref.dtype)


@functools.partial(
    jax.jit,
    static_argnames=("top_k", "renormalize", "capacity", "experts_per_block",
                     "interpret"),
)
def fused_moe_decode(
    tokens: jax.Array,        # [T, D]
    w1: jax.Array,            # [E, D, 2I]  (gate|up concatenated on -1)
    w2: jax.Array,            # [E, I, D]
    gating_output: jax.Array, # [T, E]
    *,
    top_k: int,
    renormalize: bool = True,
    capacity: int = 32,
    experts_per_block: int = 8,
    interpret: bool = False,
) -> jax.Array:
    """Decode-regime fused MoE. Returns [T, D] (full sum; shard the weights
    on I and psum externally for TP)."""
    t, d = tokens.shape
    e, _, i2 = w1.shape
    assert e % experts_per_block == 0, (e, experts_per_block)
    num_blocks = e // experts_per_block

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        grid=(num_blocks,),
        in_specs=[
            pl.BlockSpec(tokens.shape, lambda i: (0, 0)),          # resident
            pl.BlockSpec(gating_output.shape, lambda i: (0, 0)),   # resident
            pl.BlockSpec((experts_per_block, d, i2),
                         lambda i: (i, 0, 0)),                     # streamed
            pl.BlockSpec((experts_per_block, i2 // 2, d),
                         lambda i: (i, 0, 0)),                     # streamed
        ],
        out_specs=pl.BlockSpec((t, d), lambda i: (0, 0)),
        scratch_shapes=[
            pltpu.VMEM((t, d), jnp.float32),                 # acc
            pltpu.SMEM((e, capacity), jnp.int32),            # rows
            pltpu.SMEM((e, capacity), jnp.float32),          # gates
            pltpu.SMEM((e,), jnp.int32),                     # counts
            pltpu.VMEM((capacity, d), tokens.dtype),         # gathered
        ],
    )
    kernel = functools.partial(
        _decode_moe_kernel,
        num_experts=e,
        experts_per_block=experts_per_block,
        capacity=capacity,
        top_k=top_k,
        renormalize=renormalize,
        act_dtype=tokens.dtype,
    )
    return pl.pallas_call(
        kernel,
        grid_spec=grid_spec,
        out_shape=jax.ShapeDtypeStruct((t, d), tokens.dtype),
        interpret=interpret,
    )(tokens, gating_output, w1, w2)
