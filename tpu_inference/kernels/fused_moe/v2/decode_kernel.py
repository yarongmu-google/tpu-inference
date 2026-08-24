"""Decode-specific fused MoE kernel (TP, VMEM-resident tokens).

DESIGN PREMISE: MoE is token-isolated - no cross-token state, no KV walk
- so a DECODE step's entire activation working set is [num_tokens,
hidden] (~4 MB at 512x4096 bf16), which fits VMEM. That flips the regime
vs prefill:

  decode  (this kernel): tokens stay VMEM-resident for the whole layer;
      weights stream ONCE per expert (m ~= B*k/E rows each, no reuse
      possible); the layer is weight-stream-bound and everything else
      must hide under it.
  prefill (NOT this kernel): activations outgrow VMEM; the right shape is
      expert-sorted HBM gathers with same-expert weight-tile caching (see
      kernels/experimental/fused_moe - gmm_fused_rs).

NO DYNAMIC ADDRESSING. Dispatch is expressed as matrix products rather
than indexed row copies, so no buffer is ever indexed by a data-dependent
row. That is not a stylistic choice: VMEM is tiled ((16,128) bf16), so a
dynamic row index is illegal for vector loads AND DMAs alike unless it is
provably tile-aligned, and every layout that legalises it (3-D shapes,
flat strided rows) forces the matmul operand through a relayout that
costs more than the work itself. Selection by matmul keeps every buffer
2-D and every index static; the MXU absorbs the routing.

Pipeline (single pallas_call, grid = expert blocks; stage names are
jax.named_scope spans -> hardware trace markers for per-stage xprof):
  routing   (grid step 0, LOCAL shard): router matmul fused in-kernel -
            S[E, T/P] = W @ X_T in one MXU op (W streams as stored, X on
            the transposing push) -> argmax-free maxmask top-k in [E, T]
            (per-token max/mask = sublane reductions; winner by
            value-equality vs the broadcast max - no bf16-argmax gap).
  AG        (grid step 0): VMEM-direct ICI remote copies all-gather the
            token shard + the [T/P, k] top-k results into every peer (3
            static descriptors per peer, no HBM staging), overlapping the
            first weight-block prefetch.
  masks     per grid step: for this step's `be` experts, a one-hot
            dispatch operator OH[be*C, T] (and its gated transpose) built
            from the top-k table with lane cumsums - all vector work on
            [k, T] / [C, T] shapes, no scalar core.
  gather    X[be*C, D] = OH @ tokens - ONE matmul for the whole block;
            the fills amortise across all `be` experts.
  gmm1+act  per expert (static slice of X): [C, D] @ [D, 2I] -> SwiGLU
  gmm2      [C, I] @ [I, D] -> [C, D]
  combine   acc += OHG_T @ Y - one matmul; the gate weights ride in OHG_T
  epilogue  accumulator -> output (partial under TP; caller reduce-scatters)

v0 scope: bf16 weights/acts, no shared expert, no in-kernel RS, capacity
C rows/expert with overflow DROPPED (tokens whose slot >= C simply have
no row in OH; production needs a spill path).
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.common import math as kmath


def _routing(*, logits_et, top_k, renormalize_topk_logits):
    # topk(softmax(l)) == topk(l): softmax is per-token monotone. Under
    # renormalization the full-E denominator Z cancels:
    #   (exp(l_i)/Z) / sum_topk(exp(l_j)/Z) = exp(l_i-m) / sum_topk(exp(l_j-m))
    # so pick on RAW logits and softmax over just the k winners. Without
    # renormalization Z survives - pay the full E-wide softmax.
    if renormalize_topk_logits:
        x = logits_et  # [E, T/P]
    else:
        with jax.named_scope("moe_score"):
            m = jnp.max(logits_et, axis=0, keepdims=True)
            ex = kmath.exp(logits_et - m)
            x = ex / jnp.sum(ex, axis=0, keepdims=True)      # per-token over E
    iota = lax.broadcasted_iota(jnp.int32, x.shape, 0)       # expert ids
    weights = []
    indices = []
    for k_id in range(top_k):
        with jax.named_scope(f"moe_topk{k_id}"):
            mx = jnp.max(x, axis=0, keepdims=True)           # [1, T/P]
            win = x == jnp.broadcast_to(mx, x.shape)
            idx = jnp.max(jnp.where(win, iota, 0), axis=0, keepdims=True)
            sel = iota == jnp.broadcast_to(idx, x.shape)
            weights.append(mx[0])                            # [T/P]
            indices.append(idx[0])
            if k_id != top_k - 1:
                x = jnp.where(sel, -jnp.inf, x)
    w = jnp.stack(weights, axis=1)  # [T/P, k] logits (renorm) or probs
    i = jnp.stack(indices, axis=1)  # [T/P, k]
    if renormalize_topk_logits:
        with jax.named_scope("moe_topk_softmax"):
            # k-wide softmax; w[:, :1] is the max logit (round 0's winner)
            ex = kmath.exp(w - w[:, :1])
            w = ex / jnp.sum(ex, axis=1, keepdims=True)
    return w, i


def _lane_cumsum(x, t):
    """Inclusive Hillis-Steele prefix sum along the lane axis of [1, T]."""
    lane = lax.broadcasted_iota(jnp.int32, x.shape, 1)
    sh = 1
    while sh < t:
        x = x + jnp.where(lane >= sh, pltpu.roll(x, shift=sh, axis=1), 0)
        sh *= 2
    return x


def _dispatch_operators(*, idx_kt, idx_tk, w_tk, e, t, capacity,
                        act_dtype):
    """One expert's dispatch operators, from the top-k table.

    Returns (OH[C, T], OHG_T[T, C]): OH[c, t] = 1 iff token t is the c-th
    token routed to expert e; OHG_T is its transpose scaled by the gate
    weight, so the combine matmul carries the weighting for free. Each
    matmul gets its natural operand layout.

    The two orientations are reduced from the two stored layouts of the
    top-k table - [k, T] for the lane-major operator, [T, k] for the
    sublane-major one - so only `slot` (which needs a prefix sum along T,
    cheap only with T on lanes) is ever transposed.
    """
    eq = idx_kt == e                                         # [k, T]
    mask = jnp.max(eq.astype(jnp.int32), axis=0, keepdims=True)     # [1, T]
    slot = _lane_cumsum(mask, t) - mask                      # exclusive
    live = mask == 1

    iota_c = lax.broadcasted_iota(jnp.int32, (capacity, t), 0)
    oh = jnp.where((iota_c == slot) & live, 1.0, 0.0).astype(act_dtype)

    # transposed operator: T on sublanes, slots on lanes. live/gate come
    # straight from the [T, k] table (T already on sublanes); only slot
    # has to cross.
    eq_t = idx_tk == e                                       # [T, k]
    live_t = jnp.max(eq_t.astype(jnp.int32), axis=1,
                     keepdims=True) == 1                     # [T, 1]
    gate_t = jnp.sum(jnp.where(eq_t, w_tk, 0.0), axis=1,
                     keepdims=True)                          # [T, 1]
    slot_t = slot.T                                          # [T, 1]
    iota_c_t = lax.broadcasted_iota(jnp.int32, (t, capacity), 1)
    ohg_t = jnp.where((iota_c_t == slot_t) & live_t,
                      gate_t, 0.0).astype(act_dtype)
    return oh, ohg_t


def _decode_moe_kernel(
    tokens_local_vmem,   # [T/P, D] this device's token shard
    router_w_vmem,       # [E, D] router weight (upstream [out, in] layout)
    w1_vmem,             # [NE, D, 2I] this grid step's expert block
    w2_vmem,             # [NE, I, D]
    out_vmem,            # [T, D] union partials (caller reduce-scatters)
    tokens_full_vmem,    # VMEM [T, D] the gathered union
    send_sem,            # DMA - all outbound remote copies
    recv_sem_tokens,     # DMA - inbound token shards
    recv_sem_meta,       # DMA - inbound top-k idx/weight shards
    acc_vmem,            # VMEM [T, D] f32 accumulator
    topk_idx_vmem,       # VMEM [T, k] i32 - AG'd routing results
    topk_w_vmem,         # VMEM [T, k] f32
    idx_kt_vmem,         # VMEM [k, T] i32 - transposed once, read per step
    x_vmem,              # VMEM [be*C, D] gathered rows for this block
    y_vmem,              # VMEM [be*C, D] expert outputs for this block
    *,
    axis_name: str,
    mesh_axis_names: tuple,
    num_experts: int,
    be: int,
    capacity: int,
    top_k: int,
    renormalize_topk_logits: bool,
    act_dtype,
    bd1c: int | None = None,
    bd2c: int | None = None,
):
    blk = pl.program_id(0)
    num_blocks = pl.num_programs(0)
    t, d = tokens_full_vmem.shape
    # static under shard_map - same axis env as lax.axis_index below
    num_devices = lax.axis_size(axis_name)

    def _mesh_device_id(p: int):
        # MESH device ids are coordinate tuples, one entry per mesh axis:
        # the target's rank on the TP axis, my OWN coordinate on every
        # other axis (v1's get_mesh_device_id, generalized).
        return tuple(
            jnp.int32(p) if name == axis_name else lax.axis_index(name)
            for name in mesh_axis_names)

    # ---- grid step 0: local routing -> AG ----
    @pl.when(blk == 0)
    def _prologue():
        my_id = lax.axis_index(axis_name)
        t_loc = tokens_local_vmem.shape[0]
        row0 = my_id * t_loc

        with jax.named_scope("moe_routing"):
            logits_et = lax.dot_general(
                router_w_vmem[...],          # [E, D]
                tokens_local_vmem[...],      # [T/P, D]
                dimension_numbers=(((1,), (1,)), ((), ())),
                preferred_element_type=jnp.float32,
            )                                # -> [E, T/P]
            weights, indices = _routing(
                logits_et=logits_et,
                top_k=top_k,
                renormalize_topk_logits=renormalize_topk_logits,
            )                                # -> [T/P, k] each
            topk_idx_vmem[pl.ds(row0, t_loc), :] = indices
            topk_w_vmem[pl.ds(row0, t_loc), :] = weights

        with jax.named_scope("moe_ag_local"):
            tokens_full_vmem[pl.ds(row0, t_loc), :] = tokens_local_vmem[...]

        with jax.named_scope("moe_ag_barrier"):
            # The GLOBAL barrier semaphore (requires collective_id in
            # CompilerParams): the one semaphore remote signals may target.
            # No peer VMEM may be written before its owner enters the
            # kernel. v1's sync_barrier shape: signal ALL peers including
            # self, wait for the full count - no conditionals.
            barrier_sem = pltpu.get_barrier_semaphore()
            for p in range(num_devices):
                pl.semaphore_signal(barrier_sem,
                                    device_id=_mesh_device_id(p),
                                    device_id_type=pl.DeviceIdType.MESH)
            pl.semaphore_wait(barrier_sem, num_devices)

        with jax.named_scope("moe_ag_remote"):
            for p in range(num_devices):
                @pl.when(p != my_id)
                def _send(p=p):
                    for src, dst, rsem in (
                        (tokens_local_vmem,
                         tokens_full_vmem.at[pl.ds(row0, t_loc), :],
                         recv_sem_tokens),
                        (topk_idx_vmem.at[pl.ds(row0, t_loc), :],
                         topk_idx_vmem.at[pl.ds(row0, t_loc), :],
                         recv_sem_meta),
                        (topk_w_vmem.at[pl.ds(row0, t_loc), :],
                         topk_w_vmem.at[pl.ds(row0, t_loc), :],
                         recv_sem_meta),
                    ):
                        pltpu.make_async_remote_copy(
                            src_ref=src,
                            dst_ref=dst,
                            send_sem=send_sem,
                            recv_sem=rsem,
                            device_id=_mesh_device_id(p),
                            device_id_type=pl.DeviceIdType.MESH,
                        ).start()

        # A wait consumes semaphore credits equal to the dummy ref's byte
        # count (the ref's ADDRESS is meaningless) - so one wait shaped
        # "all P-1 inbound shards" drains a buffer in a single call.
        inbound = num_devices - 1

        with jax.named_scope("moe_ag_drain_meta"):
            # ONLY the top-k arrivals: everything up to the gather matmul
            # depends on these and not on the (much larger) token shards,
            # which keep streaming underneath.
            for buf in (topk_idx_vmem, topk_w_vmem):
                pltpu.make_async_copy(
                    buf.at[pl.ds(0, inbound * t_loc), :],
                    buf.at[pl.ds(0, inbound * t_loc), :],
                    recv_sem_meta).wait()

        with jax.named_scope("moe_topk_transpose"):
            # the per-step mask builder reduces over k with T on LANES;
            # one transpose here beats one per expert per grid step.
            idx_kt_vmem[...] = topk_idx_vmem[...].T          # [k, T]
        acc_vmem[...] = jnp.zeros_like(acc_vmem)

    # ---- this step's expert block: masks -> gather -> FFN -> combine ----
    # The token all-gather has been in flight since grid step 0 issued it;
    # settle it as LATE as possible - after the transpose, the accumulator
    # zeroing and the first block's mask building, which need only the
    # top-k tables. The first read of the union is the gather matmul.
    @pl.when(blk == 0)
    def _drain_tokens():
        t_loc = tokens_local_vmem.shape[0]
        inbound = num_devices - 1
        with jax.named_scope("moe_ag_drain_tokens"):
            for sem in (recv_sem_tokens, send_sem):
                pltpu.make_async_copy(
                    tokens_full_vmem.at[pl.ds(0, inbound * t_loc), :],
                    tokens_full_vmem.at[pl.ds(0, inbound * t_loc), :],
                    sem).wait()
            # send-side hygiene: the top-k sends complete too
            for buf in (topk_idx_vmem, topk_w_vmem):
                pltpu.make_async_copy(
                    buf.at[pl.ds(0, inbound * t_loc), :],
                    buf.at[pl.ds(0, inbound * t_loc), :], send_sem).wait()

    with jax.named_scope("moe_masks"):
        idx_kt = idx_kt_vmem[...]
        idx_tk, w_tk = topk_idx_vmem[...], topk_w_vmem[...]
        ohs, ohg_ts = [], []
        for le in range(be):
            oh, ohg_t = _dispatch_operators(
                idx_kt=idx_kt, idx_tk=idx_tk, w_tk=w_tk,
                e=blk * be + le, t=t,
                capacity=capacity, act_dtype=act_dtype)
            ohs.append(oh)
            ohg_ts.append(ohg_t)
        oh_block = jnp.concatenate(ohs, axis=0)              # [be*C, T]
        ohg_t_block = jnp.concatenate(ohg_ts, axis=1)        # [T, be*C]

    with jax.named_scope("moe_gather"):
        # ONE matmul selects every routed row for all `be` experts; the
        # weight fills amortise across the block.
        x_vmem[...] = jnp.dot(
            oh_block, tokens_full_vmem[...],
            preferred_element_type=jnp.float32).astype(act_dtype)

    for le in range(be):
        lo = le * capacity
        with jax.named_scope("moe_gmm1"):
            kc = bd1c or d
            h = jnp.zeros((capacity, w1_vmem.shape[-1]), jnp.float32)
            for k0 in range(0, d, kc):
                h = h + jnp.dot(
                    x_vmem[lo:lo + capacity, k0:k0 + kc],    # [C, kc]
                    w1_vmem[le, k0:k0 + kc, :],              # [kc, 2I]
                    preferred_element_type=jnp.float32)
            i_half = h.shape[-1] // 2
            gate, up = h[:, :i_half], h[:, i_half:]
            a = (jax.nn.silu(gate) * up).astype(act_dtype)   # [C, I]

        with jax.named_scope("moe_gmm2"):
            nc = bd2c or d
            for n0 in range(0, d, nc):
                y_vmem[lo:lo + capacity, n0:n0 + nc] = jnp.dot(
                    a,
                    w2_vmem[le, :, n0:n0 + nc],              # [I, nc]
                    preferred_element_type=jnp.float32).astype(act_dtype)

    with jax.named_scope("moe_combine"):
        # scatter-add as ONE matmul: OHG_T carries the gate weights, so
        # this both routes rows home and applies the top-k weighting.
        acc_vmem[...] = acc_vmem[...] + jnp.dot(
            ohg_t_block, y_vmem[...], preferred_element_type=jnp.float32)

    # ---- last grid step: emit ----
    @pl.when(blk == num_blocks - 1)
    def _epilogue():
        out_vmem[...] = acc_vmem[...].astype(out_vmem.dtype)


@functools.partial(
    jax.jit,
    static_argnames=("mesh", "axis_name", "top_k", "renormalize_topk_logits",
                     "capacity", "be", "bd1c", "bd2c", "vmem_limit_bytes",
                     "interpret"),
)
def fused_moe_decode_tp_fused(
    tokens_local: jax.Array,   # [T/P, D]
    w1_local: jax.Array,       # [E, D, 2*I/P]
    w2_local: jax.Array,       # [E, I/P, D]
    router_w: jax.Array,       # [E, D] router weight, replicated
    *,
    mesh: jax.sharding.Mesh,
    axis_name: str,
    top_k: int,
    renormalize_topk_logits: bool = True,
    capacity: int = 32,
    be: int = 4,
    bd1c: int | None = None,
    bd2c: int | None = None,
    vmem_limit_bytes: int = 64 * 1024 * 1024,
    interpret=False,
) -> jax.Array:
    """v0.2: router-fused + topk-first, the all-gather fused INTO the
    kernel as VMEM-direct ICI remote copies (per peer: token shard +
    [T/P, k] top-k results - the raw gating never exists outside the
    kernel), overlapping the first weight-block prefetch. Exit stays an
    external psum_scatter (in-kernel RS = the next step, gmm_fused_rs-style
    direct writes)."""
    if interpret is True:
        # Remote DMAs/semaphore signals need the TPU-simulating interpret
        # machine (jax/_src/pallas/mosaic/interpret/); plain interpret=True
        # routes to the discharge interpreter, which raises "Remote signal
        # not implemented" / cannot take mesh device ids.
        interpret = pltpu.InterpretParams(dma_execution_mode="on_wait")
    num_devices = mesh.shape[axis_name]
    t_loc, d = tokens_local.shape
    t = t_loc * num_devices
    e, _, i2 = w1_local.shape
    assert e % be == 0, (e, be)
    assert d % (bd1c or d) == 0 and d % (bd2c or d) == 0, (d, bd1c, bd2c)
    assert router_w.shape == (e, d), (router_w.shape, e, d)
    num_blocks = e // be

    # Static VMEM budget: every window and scratch buffer is a
    # compile-time constant, so fail fast with an itemized sum instead of
    # a backend allocation error. Weight blocks are double-buffered by the
    # grid pipeline; the constant-index windows are single-buffered.
    act = jnp.dtype(tokens_local.dtype).itemsize
    weight_block = be * (d * i2 + (i2 // 2) * d) * act
    vmem_need = (
        2 * weight_block                       # w1+w2 stream, double-buffered
        + t_loc * d * act                      # tokens_local window
        + e * d * router_w.dtype.itemsize      # router_w window
        + t * d * act                          # out window
        + t * d * act                          # tokens_full
        + t * d * 4                            # acc (f32)
        + 2 * be * capacity * d * act          # x + y
        + 3 * t * top_k * 4                    # topk tables, both layouts
    )
    assert vmem_need <= vmem_limit_bytes, (
        f"static VMEM need {vmem_need / 2**20:.1f} MiB exceeds "
        f"{vmem_limit_bytes / 2**20:.0f} MiB "
        f"(weight block {2 * weight_block / 2**20:.1f}, be={be}; "
        f"try a smaller be or capacity)")

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        grid=(num_blocks,),
        in_specs=[
            pl.BlockSpec(tokens_local.shape, lambda i: (0, 0)),
            pl.BlockSpec(router_w.shape, lambda i: (0, 0)),
            pl.BlockSpec((be, d, i2), lambda i: (i, 0, 0)),
            pl.BlockSpec((be, i2 // 2, d), lambda i: (i, 0, 0)),
        ],
        out_specs=pl.BlockSpec((t, d), lambda i: (0, 0)),
        scratch_shapes=[
            pltpu.VMEM((t, d), tokens_local.dtype),          # tokens_full
            pltpu.SemaphoreType.DMA,                         # send_sem
            pltpu.SemaphoreType.DMA,                         # recv_sem_tokens
            pltpu.SemaphoreType.DMA,                         # recv_sem_meta
            pltpu.VMEM((t, d), jnp.float32),                 # acc
            pltpu.VMEM((t, top_k), jnp.int32),               # topk_idx
            pltpu.VMEM((t, top_k), jnp.float32),             # topk_w
            pltpu.VMEM((top_k, t), jnp.int32),               # idx_kt
            pltpu.VMEM((be * capacity, d), tokens_local.dtype),   # x
            pltpu.VMEM((be * capacity, d), tokens_local.dtype),   # y
        ],
    )
    kernel = functools.partial(
        _decode_moe_kernel,
        axis_name=axis_name,
        mesh_axis_names=tuple(mesh.axis_names),
        num_experts=e,
        be=be,
        capacity=capacity,
        top_k=top_k,
        renormalize_topk_logits=renormalize_topk_logits,
        act_dtype=tokens_local.dtype,
        bd1c=bd1c,
        bd2c=bd2c,
    )
    out = pl.pallas_call(
        kernel,
        grid_spec=grid_spec,
        out_shape=jax.ShapeDtypeStruct((t, d), tokens_local.dtype),
        compiler_params=pltpu.CompilerParams(
            collective_id=13,
            vmem_limit_bytes=vmem_limit_bytes,
        ),
        interpret=interpret,
    )(tokens_local, router_w, w1_local, w2_local)
    return lax.psum_scatter(out, axis_name, scatter_dimension=0, tiled=True)
