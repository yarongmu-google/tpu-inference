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


def _moe_compute(
    # views (VMEM refs: kernel inputs or gathered scratch)
    tokens_ref,      # [T, D] the UNION batch
    gating_ref,      # [T, E]
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
    yout_ref,        # VMEM [C, D] f32 gmm2 output (ref-indexable for RMW)
    topk_idx_vmem,   # VMEM [T, K] i32 routing results (vector-writable)
    topk_w_vmem,     # VMEM [T, K] f32
    topk_idx_smem,   # SMEM [T, K] i32 (scalar-readable copy)
    topk_w_smem,     # SMEM [T, K] f32
    copy_sem,        # DMA semaphore for the vmem->smem routing copies
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
            # Traced arrays cannot be dynamically indexed on TPU (registers,
            # not memory) - land them in VMEM (vector store) and copy to
            # SMEM so the list builder can read them with scalar indices
            # (the v1 t2e_routing vmem->smem pattern).
            topk_idx_vmem[...] = indices
            topk_w_vmem[...] = weights
            pltpu.make_async_copy(topk_idx_vmem, topk_idx_smem,
                                  copy_sem).start()
            pltpu.make_async_copy(topk_idx_vmem, topk_idx_smem,
                                  copy_sem).wait()
            pltpu.make_async_copy(topk_w_vmem, topk_w_smem, copy_sem).start()
            pltpu.make_async_copy(topk_w_vmem, topk_w_smem, copy_sem).wait()
        # ONE scalar pass over the (token, slot) pairs: B*k iterations
        # (Rupeng-structured; v1's per-expert rescan would be B*k*E/8).
        with jax.named_scope("moe_lists"):
            for e in range(num_experts):
                counts_ref[e] = 0

            def _pair(p, carry):
                tok = p // top_k
                k_id = p % top_k
                e = topk_idx_smem[tok, k_id]
                c = counts_ref[e]

                @pl.when(c < capacity)
                def _do():
                    rows_ref[e, c] = tok
                    gates_ref[e, c] = topk_w_smem[tok, k_id]
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
                # through a ref: traced arrays cannot be dynamically sliced
                # in the combine loop below.
                yout_ref[...] = jnp.dot(a, w2,
                                        preferred_element_type=jnp.float32)

            with jax.named_scope("moe_combine"):
                def _s(c, carry):
                    @pl.when(c < cnt)
                    def _do():
                        row = rows_ref[e, c]
                        g = gates_ref[e, c]
                        acc_ref[pl.ds(row, 1), :] = (
                            acc_ref[pl.ds(row, 1), :]
                            + g * yout_ref[pl.ds(c, 1), :])

                    return carry

                lax.fori_loop(0, capacity, _s, 0)

    # ---- last grid step: emit ----
    @pl.when(blk == num_blocks - 1)
    def _epilogue():
        out_ref[...] = acc_ref[...].astype(out_ref.dtype)


def _decode_moe_kernel(tokens_ref, gating_ref, w1_ref, w2_ref, out_ref,
                       *scratch, **params):
    """Single-device view: tokens/gating arrive as the union batch."""
    _moe_compute(tokens_ref, gating_ref, w1_ref, w2_ref, out_ref,
                 *scratch, **params)


def _decode_moe_kernel_ag(
    tokens_local_ref,   # [T/P, D] this device's token shard (VMEM)
    gating_local_ref,   # [T/P, E]
    w1_ref, w2_ref, out_ref,
    # AG scratch first, then _moe_compute's scratch
    tokens_full,        # VMEM [T, D] the gathered union
    gating_full,        # VMEM [T, E]
    send_sem,           # DMA
    recv_sem,           # DMA
    *scratch,
    num_devices: int,
    axis_name: str,
    **params,
):
    """In-kernel VMEM-direct all-gather (v0.2): each device remote-copies
    its contiguous token/gating shard straight into every peer's VMEM
    scratch (no HBM staging, no per-row machinery - P-1 static descriptors
    per buffer), then runs the same compute body on the union.

    Idiom per tests/pallas/tpu_pallas_distributed_test.py
    (test_basic_remote_vmem_dma): ready-semaphore handshake before remote
    VMEM writes; memory space rides on the refs, the copy call is the
    same make_async_remote_copy.
    """
    blk = pl.program_id(0)

    @pl.when(blk == 0)
    def _all_gather():
        my_id = lax.axis_index(axis_name)
        t_loc = tokens_local_ref.shape[0]

        with jax.named_scope("moe_ag_barrier"):
            # The GLOBAL barrier semaphore (requires collective_id in
            # CompilerParams): the one semaphore remote signals may target.
            barrier_sem = pltpu.get_barrier_semaphore()
            for p in range(num_devices):
                @pl.when(p != my_id)
                def _sig(p=p):
                    pl.semaphore_signal(barrier_sem, device_id=(jnp.int32(p),),
                                        device_id_type=pl.DeviceIdType.MESH)
            pl.semaphore_wait(barrier_sem, num_devices - 1)

        with jax.named_scope("moe_ag_local"):
            tokens_full[pl.ds(my_id * t_loc, t_loc), :] = tokens_local_ref[...]
            gating_full[pl.ds(my_id * t_loc, t_loc), :] = gating_local_ref[...]

        with jax.named_scope("moe_ag_remote"):
            copies = []
            for p in range(num_devices):
                @pl.when(p != my_id)
                def _send(p=p):
                    for src, dst in ((tokens_local_ref, tokens_full),
                                     (gating_local_ref, gating_full)):
                        pltpu.make_async_remote_copy(
                            src_ref=src,
                            dst_ref=dst.at[pl.ds(my_id * t_loc, t_loc), :],
                            send_sem=send_sem,
                            recv_sem=recv_sem,
                            device_id=(jnp.int32(p),),
                            device_id_type=pl.DeviceIdType.MESH,
                        ).start()
            # drain: 2 sends per peer out, 2 shards per peer in. The wait
            # amount is inferred from the ref shape, so the dummy waits must
            # match the TRANSFERRED shapes (one [T/P,D] token shard + one
            # [T/P,E] gating shard per peer, per direction) - waiting on the
            # full [T,D] buffer would expect P times the bytes and hang.
            for _ in range(num_devices - 1):
                for shard in (tokens_local_ref, gating_local_ref):
                    pltpu.make_async_copy(shard, shard, send_sem).wait()
                    pltpu.make_async_copy(shard, shard, recv_sem).wait()

    _moe_compute(tokens_full, gating_full, w1_ref, w2_ref, out_ref,
                 *scratch, **params)


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
            pltpu.VMEM((capacity, d), jnp.float32),          # yout
            pltpu.VMEM((t, top_k), jnp.int32),               # topk_idx_vmem
            pltpu.VMEM((t, top_k), jnp.float32),             # topk_w_vmem
            pltpu.SMEM((t, top_k), jnp.int32),               # topk_idx_smem
            pltpu.SMEM((t, top_k), jnp.float32),             # topk_w_smem
            pltpu.SemaphoreType.DMA,                         # copy_sem
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


def fused_moe_decode_tp(
    tokens_local: jax.Array,   # [T/P, D] this device's token shard
    w1_local: jax.Array,       # [E, D, 2*I/P] gate|up LOCALLY concatenated
    w2_local: jax.Array,       # [E, I/P, D]
    gating_local: jax.Array,   # [T/P, E]
    *,
    axis_name: str,
    top_k: int,
    renormalize: bool = True,
    capacity: int = 32,
    experts_per_block: int = 8,
    interpret: bool = False,
) -> jax.Array:
    """v0.1 TP wrapper for the DP-attention serving context (call under
    shard_map): all-gather the token/gating shards to the union batch,
    run the decode kernel against this device's I-sharded expert slices,
    reduce-scatter the partial along the TOKEN axis so each device keeps
    the finished rows of its own tokens.

    Weight contract: w1_local is [E, D, 2*I_local] with THIS device's gate
    slice concatenated before its up slice (reshape a global [E, D, 2, I]
    sharded on the last axis). ICI cost: (P-1)/P * [T,D] in + same out
    (~7.3 MB/device at Qwen shapes), overlappable under the weight stream.
    """
    tokens = lax.all_gather(tokens_local, axis_name, axis=0, tiled=True)
    gating = lax.all_gather(gating_local, axis_name, axis=0, tiled=True)
    out = fused_moe_decode(
        tokens, w1_local, w2_local, gating,
        top_k=top_k, renormalize=renormalize, capacity=capacity,
        experts_per_block=experts_per_block, interpret=interpret)
    return lax.psum_scatter(out, axis_name, scatter_dimension=0, tiled=True)


@functools.partial(
    jax.jit,
    static_argnames=("axis_name", "num_devices", "top_k", "renormalize",
                     "capacity", "experts_per_block", "interpret"),
)
def fused_moe_decode_tp_fused(
    tokens_local: jax.Array,   # [T/P, D]
    w1_local: jax.Array,       # [E, D, 2*I/P]
    w2_local: jax.Array,       # [E, I/P, D]
    gating_local: jax.Array,   # [T/P, E]
    *,
    axis_name: str,
    num_devices: int,
    top_k: int,
    renormalize: bool = True,
    capacity: int = 32,
    experts_per_block: int = 8,
    interpret=False,
) -> jax.Array:
    """v0.2: the all-gather fused INTO the kernel as VMEM-direct ICI remote
    copies (P-1 static descriptors per buffer, no HBM staging, no per-row
    machinery), overlapping the first weight-block prefetch. Exit stays an
    external psum_scatter (in-kernel RS = the next step, Rupeng-style
    direct writes)."""
    if interpret is True:
        # Remote DMAs/semaphore signals need the TPU-simulating interpret
        # machine (jax/_src/pallas/mosaic/interpret/); plain interpret=True
        # routes to the discharge interpreter, which raises "Remote signal
        # not implemented" / cannot take mesh device ids.
        interpret = pltpu.InterpretParams(dma_execution_mode="on_wait")
    t_loc, d = tokens_local.shape
    t = t_loc * num_devices
    e, _, i2 = w1_local.shape
    assert e % experts_per_block == 0, (e, experts_per_block)
    num_blocks = e // experts_per_block

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        grid=(num_blocks,),
        in_specs=[
            pl.BlockSpec(tokens_local.shape, lambda i: (0, 0)),
            pl.BlockSpec(gating_local.shape, lambda i: (0, 0)),
            pl.BlockSpec((experts_per_block, d, i2), lambda i: (i, 0, 0)),
            pl.BlockSpec((experts_per_block, i2 // 2, d),
                         lambda i: (i, 0, 0)),
        ],
        out_specs=pl.BlockSpec((t, d), lambda i: (0, 0)),
        scratch_shapes=[
            pltpu.VMEM((t, d), tokens_local.dtype),          # tokens_full
            pltpu.VMEM((t, gating_local.shape[1]),
                       gating_local.dtype),                  # gating_full
            pltpu.SemaphoreType.DMA,                         # send_sem
            pltpu.SemaphoreType.DMA,                         # recv_sem
            pltpu.VMEM((t, d), jnp.float32),                 # acc
            pltpu.SMEM((e, capacity), jnp.int32),            # rows
            pltpu.SMEM((e, capacity), jnp.float32),          # gates
            pltpu.SMEM((e,), jnp.int32),                     # counts
            pltpu.VMEM((capacity, d), tokens_local.dtype),   # gathered
            pltpu.VMEM((capacity, d), jnp.float32),          # yout
            pltpu.VMEM((t, top_k), jnp.int32),               # topk_idx_vmem
            pltpu.VMEM((t, top_k), jnp.float32),             # topk_w_vmem
            pltpu.SMEM((t, top_k), jnp.int32),               # topk_idx_smem
            pltpu.SMEM((t, top_k), jnp.float32),             # topk_w_smem
            pltpu.SemaphoreType.DMA,                         # copy_sem
        ],
    )
    kernel = functools.partial(
        _decode_moe_kernel_ag,
        num_devices=num_devices,
        axis_name=axis_name,
        num_experts=e,
        experts_per_block=experts_per_block,
        capacity=capacity,
        top_k=top_k,
        renormalize=renormalize,
        act_dtype=tokens_local.dtype,
    )
    out = pl.pallas_call(
        kernel,
        grid_spec=grid_spec,
        out_shape=jax.ShapeDtypeStruct((t, d), tokens_local.dtype),
        compiler_params=pltpu.CompilerParams(collective_id=13),
        interpret=interpret,
    )(tokens_local, gating_local, w1_local, w2_local)
    return lax.psum_scatter(out, axis_name, scatter_dimension=0, tiled=True)
