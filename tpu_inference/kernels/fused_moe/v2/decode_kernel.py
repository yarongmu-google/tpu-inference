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
  routing   (grid step 0, LOCAL shard): router matmul fused in-kernel -
            S[E, T/P] = W @ X_T in one MXU op (W streams as stored, X on
            the transposing push) -> argmax-free maxmask top-k in [E, T]
            (per-token max/mask = sublane reductions; winner by
            value-equality vs the broadcast max - no bf16-argmax gap).
  AG        (grid step 0): VMEM-direct ICI remote copies all-gather the
            token shard + the [T/P, k] top-k results into every peer (3
            static descriptors per peer, no HBM staging), overlapping the
            first weight-block prefetch. Drains are SPLIT by dependency:
            the tiny top-k results unblock list building immediately; the
            token drain waits only where the first gather needs it, so
            the token transfer hides under the list building.
  lists     per-expert row lists + gate weights, fully vectorized
            (lane-compress): one-hot masks from the AG'd [T, k] results
            -> exclusive lane-cumsum -> butterfly compaction (roll/where,
            LSB->MSB), landed in SMEM via one bulk copy per table - no
            scalar pass.
  gather    per expert: C rows staged from the VMEM-resident union by
            scalar-indexed dynamic slices (vector loads, zero DMA).
  gmm1+act  [C, D] @ [D, 2I] -> SwiGLU -> [C, I]
  gmm2      [C, I] @ [I, D] -> [C, D], rows scaled by gate weights
  combine   scatter-add into a VMEM accumulator [T, D] (f32)
  epilogue  accumulator -> output (partial under TP; caller reduce-scatters)

v0 scope: bf16 weights/acts, no shared expert, no in-kernel RS, capacity
C rows/expert with overflow DROPPED (assert-checked in tests; production
needs a spill path).
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
    iota = lax.broadcasted_iota(jnp.int32, x.shape, 0)       # expert ids [E, T/P]
    weights = []
    indices = []
    for k_id in range(top_k):
        with jax.named_scope(f"moe_topk{k_id}"):
            mx = jnp.max(x, axis=0, keepdims=True)           # [1, T/P]
            win = x == jnp.broadcast_to(mx, x.shape)
            idx = jnp.max(jnp.where(win, iota, 0), axis=0, keepdims=True)  # [1, T/P]
            sel = iota == jnp.broadcast_to(idx, x.shape)
            weights.append(mx[0])                            # [T]
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


def _row_wait(ref3, sem):
    """Wait one row-DMA's worth of credits (the ref sizes the wait)."""
    pltpu.make_async_copy(ref3.at[pl.ds(0, 1), :, :],
                          ref3.at[pl.ds(0, 1), :, :], sem).wait()


def _gather_rows_dma(*, tokens_full_vmem, g_vmem, rows_smem, e, cnt, row_sem):
    """Plan A gather: one local VMEM->VMEM row DMA per assigned token.

    Both buffers are [rows, D/128, 128] so the dynamically indexed
    dimension 0 is UNTILED (Mosaic tiles only the two minormost dims) -
    dynamic row offsets on a tiled dimension are rejected for DMAs just
    as for vector loads (E2003 / "offsets along tiled dimensions must be
    aligned"). Same shape contract as gmm_fused_rs's dma_gather_gm_start.
    Pad slots (c >= cnt) are never written and never read back.
    """
    def _issue(c, carry):
        row = rows_smem[e, c]
        pltpu.make_async_copy(
            tokens_full_vmem.at[pl.ds(row, 1), :, :],
            g_vmem.at[pl.ds(c, 1), :, :],
            row_sem).start()
        return carry

    lax.fori_loop(0, cnt, _issue, 0)

    def _drain(c, carry):
        _row_wait(g_vmem, row_sem)
        return carry

    lax.fori_loop(0, cnt, _drain, 0)


def _decode_moe_kernel(
    tokens_local_vmem,   # [T/P, DL, L] token shard (D split as DL x L=128
                         #   so dim 0 stays untiled - see _gather_rows_dma)
    router_w_vmem,       # [E, D] router weight (upstream [out, in] layout, VMEM)
    w1_vmem,             # [NE, D, 2I] this grid step's expert block
    w2_vmem,             # [NE, I, D]
    out_vmem,            # [T, DL, L] union partials (caller reshapes + RS)
    tokens_full_vmem,    # VMEM [T, DL, L] the gathered union
    send_sem,           # DMA - all outbound remote copies
    recv_sem_tokens,    # DMA - inbound token shards
    recv_sem_meta,      # DMA - inbound top-k idx/weight shards
    acc_vmem,            # VMEM [T, DL, L] f32 accumulator
    rows_smem,           # SMEM [E, C] i32 per-expert row lists
    counts_smem,         # SMEM [1, E] i32
    gathered_vmem,       # VMEM [2, C, DL, L] staging, ping-pong by parity
    yout_vmem,           # VMEM [2, C, DL, L] f32 gmm2 output, same ping-pong
    temp_vmem,           # VMEM [C, DL, L] f32 combine staging (acc rows)
    topk_idx_vmem,      # VMEM [T, k] i32 - AG'd routing results
    topk_w_vmem,        # VMEM [T, k] f32
    rows_vmem,          # VMEM [E, C] i32 - lane-compress output staging
    gates_t_vmem,       # VMEM [C, E] f32 - TRANSPOSED gate table: static
                        #   column slice [:, e] gives the [C, 1] per-slot
                        #   scale the combine needs sublane-resident
    counts_vmem,        # VMEM [1, E] i32
    copy_sem,           # DMA semaphore for the vmem->smem table copies
    row_sem,            # DMA semaphore for gather/combine row DMAs
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
    t, dl, lanes = tokens_full_vmem.shape
    d = dl * lanes
    # static under shard_map - same axis env as lax.axis_index below
    num_devices = lax.axis_size(axis_name)

    def _mesh_device_id(p: int):
        # MESH device ids are coordinate tuples, one entry per mesh axis:
        # the target's rank on the TP axis, my OWN coordinate on every
        # other axis (v1's get_mesh_device_id, generalized).
        return tuple(
            jnp.int32(p) if name == axis_name else lax.axis_index(name)
            for name in mesh_axis_names)

    # ---- grid step 0: local routing -> AG -> list building ----
    @pl.when(blk == 0)
    def _prologue():
        my_id = lax.axis_index(axis_name)
        t_loc = tokens_local_vmem.shape[0]
        row0 = my_id * t_loc

        with jax.named_scope("moe_routing"):
            logits_et = lax.dot_general(
                router_w_vmem[...],          # [E, D]
                tokens_local_vmem[...].reshape(-1, d),   # [T/P, D]
                dimension_numbers=(((1,), (1,)), ((), ())),
                preferred_element_type=jnp.float32,
            )                               # -> [E, T/P]
            weights, indices = _routing(
                logits_et=logits_et,
                top_k=top_k,
                renormalize_topk_logits=renormalize_topk_logits,
            )                              # -> [T/P, k] each
            topk_idx_vmem[pl.ds(row0, t_loc), :] = indices  # -> [T, k] VMEM
            topk_w_vmem[pl.ds(row0, t_loc), :] = weights    # -> [T, k] VMEM

        with jax.named_scope("moe_ag_local"):
            tokens_full_vmem[pl.ds(row0, t_loc), :, :] = tokens_local_vmem[...]

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
                         tokens_full_vmem.at[pl.ds(row0, t_loc), :, :],
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
            # Drain ONLY the top-k arrivals: list building needs them,
            # not the tokens.
            for buf in (topk_idx_vmem, topk_w_vmem):
                pltpu.make_async_copy(buf.at[pl.ds(0, inbound * t_loc), :],
                                      buf.at[pl.ds(0, inbound * t_loc), :],
                                      recv_sem_meta).wait()

        acc_vmem[...] = jnp.zeros_like(acc_vmem)
        with jax.named_scope("moe_lists"):
            # Vectorized list building (lane-compress): expert one-hot
            # masks -> exclusive lane-cumsum -> butterfly compaction, all
            # roll/where over [E, T] arrays - replaces the former T*k
            # scalar pass. pltpu.roll(x, s, axis=1) is cyclic with
            # out[i] = x[i - s]; shifting LEFT by s is roll by t - s with
            # the wrapped lanes masked off.
            idx = topk_idx_vmem[...]                     # [T, k] i32
            wts = topk_w_vmem[...]                       # [T, k] f32
            lane = lax.broadcasted_iota(jnp.int32, (num_experts, t), 1)
            expert = lax.broadcasted_iota(jnp.int32, (num_experts, t), 0)
            m = jnp.zeros((num_experts, t), jnp.int32)
            gate = jnp.zeros((num_experts, t), jnp.float32)
            for j in range(top_k):
                sel = expert == idx[:, j][None, :]       # [E, T] one-hot
                m = m + sel.astype(jnp.int32)
                gate = jnp.where(sel, wts[:, j][None, :], gate)

            # inclusive Hillis-Steele cumsum along lanes
            inc = m
            sh = 1
            while sh < t:
                prev = pltpu.roll(inc, shift=sh, axis=1)  # prev[i] = inc[i-sh]
                inc = inc + jnp.where(lane >= sh, prev, 0)
                sh *= 2
            slot = inc - m                               # exclusive: list slot
            load = inc[:, t - 1]                         # [E] tokens per expert

            # Butterfly compaction, LSB->MSB: every occupant moves LEFT by
            # bit r of its ORIGINAL distance in round r. Collision-free:
            # after round r all remaining distances are multiples of
            # 2^(r+1), so two occupants cannot land on one lane. dist and
            # the valid bit travel packed (dv = dist | valid<<15), so each
            # round rolls 3 arrays (tok, gate, dv).
            dist = jnp.where(m > 0, lane - slot, 0)
            dv = dist | (m << 15)
            tok = lane
            for r in range((t - 1).bit_length()):
                sh = 1 << r
                src_tok = pltpu.roll(tok, shift=t - sh, axis=1)   # x[i+sh]
                src_gate = pltpu.roll(gate, shift=t - sh, axis=1)
                src_dv = pltpu.roll(dv, shift=t - sh, axis=1)
                moving = ((dv >> r) & 1) * (dv >> 15)
                src_moving = ((src_dv >> r) & 1) * (src_dv >> 15)
                recv = (src_moving == 1) & (lane < t - sh)
                tok = jnp.where(recv, src_tok, tok)
                gate = jnp.where(recv, src_gate, gate)
                valid = jnp.where(recv, 1,
                                  jnp.where(moving == 1, 0, dv >> 15))
                new_dist = jnp.where(recv, (src_dv & 0x7FFF) - sh,
                                     dv & 0x7FFF)
                dv = new_dist | (valid << 15)

            rows_vmem[...] = tok[:, :capacity]           # [E, C]
            # transposed so the combine can take a [C, 1] per-slot scale
            # as a STATIC column slice (sublane-resident)
            gates_t_vmem[...] = gate[:, :capacity].T     # [C, E]
            counts_vmem[...] = jnp.minimum(load, capacity)[None, :]

        with jax.named_scope("moe_lists_smem"):
            # the gather DMA loop reads rows and the expert loop reads
            # counts with scalar indices - land them in SMEM.
            pltpu.make_async_copy(rows_vmem, rows_smem, copy_sem).start()
            pltpu.make_async_copy(counts_vmem, counts_smem, copy_sem).start()

        with jax.named_scope("moe_ag_drain_tokens"):
            # The token transfer streamed in UNDER the list pass above;
            # settle it (and the send side, for semaphore hygiene) before
            # the first gather reads tokens_full_vmem.
            for sem in (recv_sem_tokens, send_sem):
                pltpu.make_async_copy(
                    tokens_full_vmem.at[pl.ds(0, inbound * t_loc), :, :],
                    tokens_full_vmem.at[pl.ds(0, inbound * t_loc), :, :],
                    sem).wait()
            for buf in (topk_idx_vmem, topk_w_vmem):
                pltpu.make_async_copy(buf.at[pl.ds(0, inbound * t_loc), :],
                                      buf.at[pl.ds(0, inbound * t_loc), :],
                                      send_sem).wait()

        with jax.named_scope("moe_lists_smem_drain"):
            # The table copies overlapped the token drain above. Waits
            # must run ONCE (the copies were started once) and must ALL
            # precede ANY table read: a wait consumes matching bytes from
            # ANY copy on copy_sem, so only the all-before-any order makes
            # the anonymity harmless.
            pltpu.make_async_copy(rows_vmem, rows_smem, copy_sem).wait()
            pltpu.make_async_copy(counts_vmem, counts_smem, copy_sem).wait()

    # ---- per expert in this block: gather -> gmm1 -> act -> gmm2 -> add ----
    for le in range(be):
        e = blk * be + le
        cnt = counts_smem[0, e]

        @pl.when(cnt > 0)
        def _process(le=le, e=e, cnt=cnt):
            # Ping-pong staging by expert parity (v1's _x2 idiom): adjacent
            # experts use disjoint gather/yout buffers, so no WAR/RAW
            # hazard chains the experts - the scheduler may run expert
            # le+1's gather (vld) under le's gmms (MXU) under le-1's
            # combine (VALU) concurrently.
            g_vmem = gathered_vmem.at[le % 2]  # [2, C, D] -> [C, D]
            y_vmem = yout_vmem.at[le % 2]      # [2, C, D] -> [C, D]

            with jax.named_scope("moe_gather"):
                _gather_rows_dma(
                    tokens_full_vmem=tokens_full_vmem, g_vmem=g_vmem,
                    rows_smem=rows_smem, e=e, cnt=cnt, row_sem=row_sem)

            with jax.named_scope("moe_gmm1"):
                kc = bd1c or d
                h = jnp.zeros((capacity, w1_vmem.shape[-1]), jnp.float32)
                for k0 in range(0, d, kc):
                    x = g_vmem[:, k0 // lanes:(k0 + kc) // lanes, :]
                    h = h + jnp.dot(
                        x.reshape(capacity, kc),             # [C, kc]
                        w1_vmem[le, k0:k0 + kc, :],          # [kc, 2I]
                        preferred_element_type=jnp.float32)
                i_half = h.shape[-1] // 2
                gate, up = h[:, :i_half], h[:, i_half:]
                a = (jax.nn.silu(gate) * up).astype(act_dtype)   # [C, I]

            with jax.named_scope("moe_gmm2"):
                # gate folded in HERE, while the rows are still aligned.
                # e is TRACED (grid-derived), so column e is extracted by
                # rotating the small [C, E] table (dynamic lane offsets
                # would need an alignment proof; rolls take traced shifts)
                # and slicing column 0 statically.
                gate_col = pltpu.roll(
                    gates_t_vmem[...],
                    shift=(num_experts - e) % num_experts,
                    axis=1)[:, :1]                           # [C, 1] f32
                nc = bd2c or d
                for n0 in range(0, d, nc):
                    y = jnp.dot(
                        a,
                        w2_vmem[le, :, n0:n0 + nc],          # [I, nc]
                        preferred_element_type=jnp.float32) * gate_col
                    y_vmem[:, n0 // lanes:(n0 + nc) // lanes, :] = (
                        y.reshape(capacity, nc // lanes, lanes))

            with jax.named_scope("moe_combine"):
                # Scatter-add via DMA staging (dynamic single-row vector
                # RMW is uncompilable, E2003): stage the cnt live acc rows
                # into temp slots, one aligned full-block add, write back.
                # Rows within an expert are distinct (a token appears at
                # most once per expert), and the drains serialize experts.
                def _stage(c, carry):
                    row = rows_smem[e, c]
                    pltpu.make_async_copy(
                        acc_vmem.at[pl.ds(row, 1), :, :],
                        temp_vmem.at[pl.ds(c, 1), :, :],
                        row_sem).start()
                    return carry

                lax.fori_loop(0, cnt, _stage, 0)

                def _drain(c, carry):
                    _row_wait(temp_vmem, row_sem)
                    return carry

                lax.fori_loop(0, cnt, _drain, 0)

                # pad slots: stale temp + garbage y, computed and discarded
                temp_vmem[...] = temp_vmem[...] + y_vmem[...]

                def _wb(c, carry):
                    row = rows_smem[e, c]
                    pltpu.make_async_copy(
                        temp_vmem.at[pl.ds(c, 1), :, :],
                        acc_vmem.at[pl.ds(row, 1), :, :],
                        row_sem).start()
                    return carry

                lax.fori_loop(0, cnt, _wb, 0)
                lax.fori_loop(0, cnt, _drain, 0)

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
    # D is split as [DL, LANES] wherever a buffer is indexed by a dynamic
    # ROW: Mosaic tiles the two minormost dims, so this keeps dim 0
    # untiled and dynamic row offsets legal (E2003 otherwise).
    lanes = 128
    assert d % lanes == 0, d
    dl = d // lanes
    t = t_loc * num_devices
    e, _, i2 = w1_local.shape
    assert e % be == 0, (e, be)
    assert d % (bd1c or d) == 0 and d % (bd2c or d) == 0, (d, bd1c, bd2c)
    assert (bd1c or lanes) % lanes == 0 and (bd2c or lanes) % lanes == 0, (
        bd1c, bd2c, lanes)
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
        + t * d * act                          # tokens_full_vmem
        + t * d * 4                            # acc (f32)
        + 2 * capacity * d * act               # gathered x2
        + 2 * capacity * d * 4                 # yout x2 (f32)
        + capacity * d * 4                     # temp (combine staging, f32)
        + 2 * t * top_k * 4                    # topk idx/w vmem
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
            pl.BlockSpec((t_loc, dl, lanes), lambda i: (0, 0, 0)),
            pl.BlockSpec(router_w.shape, lambda i: (0, 0)),
            pl.BlockSpec((be, d, i2), lambda i: (i, 0, 0)),
            pl.BlockSpec((be, i2 // 2, d),
                         lambda i: (i, 0, 0)),
        ],
        out_specs=pl.BlockSpec((t, dl, lanes), lambda i: (0, 0, 0)),
        scratch_shapes=[
            pltpu.VMEM((t, dl, lanes),
                       tokens_local.dtype),                  # tokens_full_vmem
            pltpu.SemaphoreType.DMA,                         # send_sem
            pltpu.SemaphoreType.DMA,                         # recv_sem_tokens
            pltpu.SemaphoreType.DMA,                         # recv_sem_meta
            pltpu.VMEM((t, dl, lanes), jnp.float32),         # acc
            pltpu.SMEM((e, capacity), jnp.int32),            # rows
            pltpu.SMEM((1, e), jnp.int32),                   # counts
            pltpu.VMEM((2, capacity, dl, lanes),
                       tokens_local.dtype),                  # gathered x2
            pltpu.VMEM((2, capacity, dl, lanes), jnp.float32),  # yout x2
            pltpu.VMEM((capacity, dl, lanes), jnp.float32),  # temp (combine)
            pltpu.VMEM((t, top_k), jnp.int32),               # topk_idx_vmem
            pltpu.VMEM((t, top_k), jnp.float32),             # topk_w_vmem
            pltpu.VMEM((e, capacity), jnp.int32),            # rows_vmem
            pltpu.VMEM((capacity, e), jnp.float32),          # gates_t_vmem
            pltpu.VMEM((1, e), jnp.int32),                   # counts_vmem
            pltpu.SemaphoreType.DMA,                         # copy_sem
            pltpu.SemaphoreType.DMA,                         # row_sem
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
        out_shape=jax.ShapeDtypeStruct((t, dl, lanes), tokens_local.dtype),
        compiler_params=pltpu.CompilerParams(
            collective_id=13,
            vmem_limit_bytes=vmem_limit_bytes,
        ),
        interpret=interpret,
    )(tokens_local.reshape(t_loc, dl, lanes), router_w, w1_local, w2_local)
    # the [DL, LANES] split is a kernel-internal layout detail; callers see
    # [T/P, D]. Reshapes outside the kernel are free (same linear order).
    return lax.psum_scatter(out.reshape(t, d), axis_name,
                            scatter_dimension=0, tiled=True)
