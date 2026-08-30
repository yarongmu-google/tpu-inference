"""Decode-regime fused MoE kernel (TP, VMEM-resident tokens).

Stage-by-stage documentation lives ON the step functions inside
_decode_moe_kernel (v1's convention: one function per pipeline action).
Only the two invariants that shape everything are stated here:

1. DECODE REGIME: MoE is token-isolated, so a decode step's entire
   activation working set is [T, D] (~4 MB at 512x4096 bf16) and stays
   VMEM-resident for the whole layer; weights stream once per expert
   with no reuse. Prefill outgrows VMEM and belongs to the gmm path,
   not this kernel.

2. NO DYNAMIC ADDRESSING: dispatch is matrix products, never indexed
   row copies. VMEM is tiled ((16,128) bf16), so a data-dependent row
   index is illegal for vector loads and DMAs alike unless provably
   tile-aligned, and every layout that legalises it costs a relayout
   worth more than the work it saves. Selection by matmul keeps every
   buffer 2-D and every index static; the MXU absorbs the routing.
   (Dynamic indices on the UNTILED leading dim of >=3-D scratch are
   fine - the v1 `.at[slot]` idiom.)

v0 scope: bf16 weights/acts, no shared expert, no in-kernel RS;
routing beyond `capacity` rows/expert is DROPPED (needs a spill path
for production).
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.fused_moe.v1.kernel import apply_act_fn

from tpu_inference.kernels.common import math as kmath


def _routing(
    *,
    logits_et: jax.Array,          # [E, T/P] router logits, experts major
    top_k: int,
    renormalize_topk_logits: bool,
) -> tuple[jax.Array, jax.Array]:  # ([T/P, k] f32 weights, [T/P, k] i32 ids)
    """Single chip: top-k over this device's token shard only; the
    all-gather shares the RESULTS (topk-first - routing is never
    recomputed across chips, no [E, T] logits ever exist)."""
    # topk(softmax(l)) == topk(l): softmax is per-token monotone. Under
    # renormalization the full-E denominator Z cancels:
    #   (exp(l_i)/Z) / sum_topk(exp(l_j)/Z) = exp(l_i-m) / sum_topk(exp(l_j-m))
    # so pick on RAW logits and softmax over just the k winners. Without
    # renormalization Z survives - pay the full E-wide softmax.
    if renormalize_topk_logits:
        scores = logits_et                                   # [E, T/P]
    else:
        with jax.named_scope("moe_score"):
            max_logit = jnp.max(logits_et, axis=0, keepdims=True)
            exps = kmath.exp(logits_et - max_logit)
            scores = exps / jnp.sum(exps, axis=0,
                                    keepdims=True)           # per-token over E
    expert_iota = lax.broadcasted_iota(jnp.int32, scores.shape, 0)
    weights = []
    indices = []
    for k_id in range(top_k):
        with jax.named_scope(f"moe_topk{k_id}"):
            round_max = jnp.max(scores, axis=0, keepdims=True)      # [1, T/P]
            is_winner = scores == jnp.broadcast_to(round_max, scores.shape)
            winner_id = jnp.max(jnp.where(is_winner, expert_iota, 0),
                                axis=0, keepdims=True)       # [1, T/P]
            weights.append(round_max[0])                     # [T/P]
            indices.append(winner_id[0])
            if k_id != top_k - 1:
                picked = expert_iota == jnp.broadcast_to(winner_id,
                                                         scores.shape)
                scores = jnp.where(picked, -jnp.inf, scores)
    topk_weights = jnp.stack(weights, axis=1)  # [T/P, k] logits or probs
    topk_indices = jnp.stack(indices, axis=1)  # [T/P, k]
    if renormalize_topk_logits:
        with jax.named_scope("moe_topk_softmax"):
            # k-wide softmax; column 0 is the max logit (round 0's winner)
            exps = kmath.exp(topk_weights - topk_weights[:, :1])
            topk_weights = exps / jnp.sum(exps, axis=1, keepdims=True)
    return topk_weights, topk_indices


def _lane_cumsum(
    vec: jax.Array,       # [1, T] i32 (any [rows, num_lanes] works)
    num_lanes: int,
) -> jax.Array:           # [1, T] inclusive prefix sum along lanes
    """Inclusive Hillis-Steele prefix sum along the lane axis."""
    lane_iota = lax.broadcasted_iota(jnp.int32, vec.shape, 1)
    shift = 1
    while shift < num_lanes:
        vec = vec + jnp.where(lane_iota >= shift,
                              pltpu.roll(vec, shift=shift, axis=1), 0)
        shift *= 2
    return vec


# One expert's dispatch operators are built in three phases, split BY
# EXECUTION UNIT so the driver can software-pipeline them across experts
# (phase j of expert e is independent of phase j of expert e'):
#   _mask_membership  VALU   compares + reductions over the top-k table
#   _mask_slots       XLU    lane prefix-sum (rolls) + the one transpose
#   _mask_operators   VALU   the iota==slot select storms
# Composed in order they equal the old single-function builder:
# OH[c, t] = 1 iff token t is the c-th token routed to expert e; OHG_T is
# its transpose scaled by the gate weight, so the combine matmul carries
# the weighting for free. The two orientations are reduced from the two
# stored layouts of the top-k table - [k, T] for the lane-major operator,
# [T, k] for the sublane-major one - so only `slot` (which needs a prefix
# sum along T, cheap only with T on lanes) ever crosses.


def _mask_membership(
    *,
    topk_idx_kt: jax.Array,   # [k, T] i32 chosen expert ids, k major
    topk_idx_tk: jax.Array,   # [T, k] i32 same table, tokens major
    topk_w_tk: jax.Array,     # [T, k] f32 gate weights, tokens major
    expert_id,                # scalar (may be traced)
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """VALU phase: which tokens picked this expert, and their gates.

    Returns (routed_mask [1, T] i32 0/1, live_t [T, 1] bool,
    gate_t [T, 1] f32)."""
    hit_kt = topk_idx_kt == expert_id                        # [k, T]
    routed_mask = jnp.max(hit_kt.astype(jnp.int32), axis=0,
                          keepdims=True)                     # [1, T]
    hit_tk = topk_idx_tk == expert_id                        # [T, k]
    live_t = jnp.max(hit_tk.astype(jnp.int32), axis=1,
                     keepdims=True) == 1                     # [T, 1]
    gate_t = jnp.sum(jnp.where(hit_tk, topk_w_tk, 0.0), axis=1,
                     keepdims=True)                          # [T, 1]
    return routed_mask, live_t, gate_t


def _mask_slots(routed_mask: jax.Array,
                num_tokens: int) -> tuple[jax.Array, jax.Array]:
    """XLU phase: exclusive prefix sum -> per-token slot, both layouts.

    Returns (slot [1, T] i32, slot_t [T, 1] i32)."""
    slot = _lane_cumsum(routed_mask, num_tokens) - routed_mask   # [1, T]
    slot_t = slot.T                                              # [T, 1]
    return slot, slot_t


def _mask_operators(
    *,
    routed_mask: jax.Array,   # [1, T] i32 0/1
    live_t: jax.Array,        # [T, 1] bool
    gate_t: jax.Array,        # [T, 1] f32
    slot: jax.Array,          # [1, T] i32
    slot_t: jax.Array,        # [T, 1] i32
    num_tokens: int,
    capacity: int,
    act_dtype,
) -> tuple[jax.Array, jax.Array]:
    """VALU phase: expand slots into the two one-hot operators.

    Returns (OH [C, T], OHG_T [T, C]) in act dtype."""
    live = routed_mask == 1
    slot_iota = lax.broadcasted_iota(jnp.int32, (capacity, num_tokens), 0)
    oh = jnp.where((slot_iota == slot) & live, 1.0, 0.0).astype(act_dtype)
    slot_iota_t = lax.broadcasted_iota(jnp.int32, (num_tokens, capacity), 1)
    ohg_t = jnp.where((slot_iota_t == slot_t) & live_t,
                      gate_t, 0.0).astype(act_dtype)
    return oh, ohg_t


def _decode_moe_kernel(
    tokens_local_hbm,    # HBM [T/P, D] this device's token shard
    route_in_hbm,        # HBM [E, D] router weight, or [T/P, E] logits
                         # when not router_fused (upstream [out, in])
    w1_hbm,              # HBM [E, D, 2I] - streamed via the blocked fetch
    w2_hbm,              # HBM [E, I, D]
    out_hbm,             # HBM [T, D] union partials (caller
                         # reduce-scatters) - DMA'd once from out_stage
    tokens_full_vmem,    # VMEM [T, D] the gathered union
    send_sem,            # DMA - all outbound remote copies
    recv_sem_tokens,     # DMA - inbound token shards
    recv_sem_meta,       # DMA - inbound top-k idx/weight shards
    acc_vmem,            # VMEM [T, D] f32 accumulator
    topk_idx_vmem,       # VMEM [T, k] i32 - AG'd routing results
    topk_w_vmem,         # VMEM [T, k] f32
    idx_kt_vmem,         # VMEM [k, T] i32 - transposed once, read per step
    x_vmem,              # VMEM [2, be*C, D] gathered rows, parity slots:
                         # slot k%2 read at step k, slot (k+1)%2 built
    y_vmem,              # VMEM [bg, be*C, D] expert outputs for the group
    ohg_vmem,            # VMEM [2*bg, T, be*C] combine operators - a ring
                         # written one step AHEAD of its group's combine
    b_w1_x2_vmem,        # VMEM [2, be, D, 2I] - x2 buffer of whole
                         # expert-block slabs, parity slots like x
                         # (slot k%2 read at step k)
    b_w2_x2_vmem,        # VMEM [2, be, I, D]
    local_sems,          # DMA (2, 5): 2 x [b_gating_sem, b_w1_sem,
                         # b_w2_sem, b_tokens_sem, b_output_sem] (v1's
                         # layout; index 3 is his b_w3_sem - our w1
                         # fuses w1+w3, so it carries the token fetch)
    tokens_local_vmem,   # VMEM [T/P, D] prologue-fetched copy
    router_w_vmem,       # VMEM route_in-shaped prologue-fetched copy
    out_stage_vmem,      # VMEM [T, D] act-dtype staging for the output
    *,
    axis_name: str,
    mesh_axis_names: tuple,
    num_experts: int,
    be: int,
    bg: int,
    capacity: int,
    top_k: int,
    renormalize_topk_logits: bool,
    act_dtype,
    router_fused: bool = True,
    bd1c: int | None = None,
    bd2c: int | None = None,
    bcT: int | None = None,
    ablate: str = "none",
):
    # `ablate` stubs one stage at a time so wall-clock differences localise
    # the cost. The profiler segfaults in this environment (PLUGIN_Profiler
    # ABI mismatch), so differential timing is the only stage-level signal.
    # NB: under dispatch-ahead, attribution shifts by one block - step k
    # prices FFN(k) + dispatch(k+1); totals per kernel are unaffected.
    # ablate=weights = all compute stubbed with the blocked weight fetch still
    # streaming: the stream's in-situ bandwidth. ablate=all additionally
    # stubs the weight fetch itself AND the dispatch park - the bare per-step
    # floor (grid machinery + branches + prologue). (weights - all) =
    # what the stream truly costs; measured 2026-08-29: weights was
    # UNCHANGED window->manual fetch (3266 -> 3247), so the floor, not the
    # stream, is the prime suspect. Static Python gates - `ablate` is a
    # static kwarg, so stubbed stages never enter the trace and DMA
    # start/wait pairing holds trivially.
    do_masks = ablate not in ("masks", "weights", "all")
    do_gather = ablate not in ("gather", "weights", "all")
    do_ffn = ablate not in ("ffn", "weights", "all")
    do_combine = ablate not in ("combine", "weights", "all")
    do_weights = ablate != "all"
    do_park = ablate != "all"
    # routing/ag rows decompose the FIXED per-call cost (the 2.8ms
    # ablate=all floor is once-per-invocation, not per-step): routing
    # stubs the router matmul + topk (masks then read garbage tables);
    # ag stubs the barrier + remote sends + drains (peer rows stay
    # uninitialized). Timing-only, like every other row.
    do_routing = ablate != "routing"
    do_ag = ablate != "ag"

    block_id = pl.program_id(0)          # which expert block (grid step)
    num_blocks = pl.num_programs(0)
    num_tokens, hidden_size = tokens_full_vmem.shape         # T, D
    # static under shard_map - same axis env as lax.axis_index below
    num_devices = lax.axis_size(axis_name)                   # P
    num_local_tokens = tokens_local_vmem.shape[0]            # T/P (real)
    # per-device row block, padded to the sublane tile by the entry so
    # every dynamic row0 below is PROVABLY 16-aligned (E2003) - serving
    # calls with as few as 2 rows/device. Phantom rows are masked out
    # of dispatch membership; zeros flow through gather/combine.
    padded_rows = num_tokens // lax.axis_size(axis_name)
    inbound = num_devices - 1
    group_pos = lax.rem(block_id, bg)    # step's position within its group

    # ---- DISPATCH-AHEAD pipeline indices ----
    # Each step builds the NEXT step's dispatch (masks depend only on the
    # prologue's routing tables), so its VALU/XLU work can fill the FFN's
    # MXU-latency stalls. x is a parity double buffer; ohg is a ring.
    # 2*bg ring slots make writer and readers collision-free by
    # construction (bg+1 suffices: reuse at m+bg vs last read m+bg-1;
    # 2*bg makes rem cheap for power-of-two bg, the tuned cases - shrink
    # to bg+1 if VMEM gets tight).
    x_par = lax.rem(block_id, 2)         # x slot CONSUMED this step
    x_nxt = lax.rem(block_id + 1, 2)     # x slot BUILT this step
    ohg_ring = 2 * bg
    ohg_next_slot = lax.rem(block_id + 1, ohg_ring)
    # the LAST step builds a never-read dispatch for expert ids >= E:
    # expert_id is comparison-only, so out-of-range ids give all-false
    # masks - AND its park still runs the zero-operand [be*C, T] @ [T, D]
    # gather into the dead slot. That is one phantom dispatch+gather per
    # kernel invocation (1/num_blocks of the total; noise at E/be >> 1,
    # visible at tiny num_blocks). Accepted in exchange for keeping the
    # steady state branch-free; fence the tail park with
    # pl.when(block_id < num_blocks - 1) if that trade ever flips.
    first_expert = block_id * be
    next_first_expert = (block_id + 1) * be

    # ---- one function per pipeline action (v1's convention) ----

    def get_mesh_device_id(peer: int):
        # MESH device ids are coordinate tuples, one entry per mesh axis:
        # the target's rank on the TP axis, my OWN coordinate on every
        # other axis (v1's get_mesh_device_id, generalized).
        return tuple(
            jnp.int32(peer) if name == axis_name else lax.axis_index(name)
            for name in mesh_axis_names)

    def sync_barrier():
        # The GLOBAL barrier semaphore (requires collective_id in
        # CompilerParams): the one semaphore remote signals may target. No
        # peer VMEM may be written before its owner enters the kernel.
        # v1's sync_barrier shape: signal ALL peers including self, wait
        # for the full count - no conditionals.
        barrier_sem = pltpu.get_barrier_semaphore()
        for peer in range(num_devices):
            pl.semaphore_signal(barrier_sem,
                                device_id=get_mesh_device_id(peer),
                                device_id_type=pl.DeviceIdType.MESH)
        pl.semaphore_wait(barrier_sem, num_devices)

    # ---- blocked weight fetch (the measured-best structure): the whole
    # NEXT expert block streams as one contiguous slab per tensor,
    # started at the step top, and THIS block's slab is waited once at
    # the step top - so the F/A/B/C emission stays ONE basic block and
    # the dispatch-ahead mask work slides into the MXU's latency gaps.
    # (The per-tile start-next/wait-one chain was measured SLOWER here:
    # its waits inside the F loop are scheduler fences that exposed the
    # mask work - masks ablation delta 30us -> 213us. v1's loop is pure
    # FFN, so intra-loop waits cost him nothing; ours is not.)
    # bd1c stays a compute-only contraction chunk.

    def start_fetch_bw(block, bw_sem_id):
        """Start expert block `block` - w1 [be, D, 2I] and w2 [be, I, D]
        slabs, ONE contiguous descriptor each (max inner runs; measured
        2.2+ TB/s standalone). Callers fence out-of-range blocks (OOB
        DMA reads are fatal)."""
        pltpu.make_async_copy(
            src_ref=w1_hbm.at[pl.ds(block * be, be)],
            dst_ref=b_w1_x2_vmem.at[bw_sem_id],
            sem=local_sems.at[bw_sem_id, 1]).start()
        pltpu.make_async_copy(
            src_ref=w2_hbm.at[pl.ds(block * be, be)],
            dst_ref=b_w2_x2_vmem.at[bw_sem_id],
            sem=local_sems.at[bw_sem_id, 2]).start()

    def wait_fetch_bw(bw_sem_id):
        """Wait BOTH of this step's slabs, once, at the step top (dummy
        refs only size the waits)."""
        pltpu.make_async_copy(
            src_ref=w1_hbm.at[pl.ds(0, be)],
            dst_ref=b_w1_x2_vmem.at[bw_sem_id],
            sem=local_sems.at[bw_sem_id, 1]).wait()
        pltpu.make_async_copy(
            src_ref=w2_hbm.at[pl.ds(0, be)],
            dst_ref=b_w2_x2_vmem.at[bw_sem_id],
            sem=local_sems.at[bw_sem_id, 2]).wait()

    # PROLOGUE 0 (DMA): the once-per-kernel inputs, one start/wait pair
    # per tensor (v1's start_fetch_b_gating / wait_fetch_b_gating
    # shape). These used to be Pallas windows with a CONSTANT index
    # map, which Mosaic pipelines in `synchronous` mode - per-step
    # machinery for per-kernel constants. As plain HBM refs the
    # harness has nothing to do between steps.

    def start_fetch_b_tokens():
        pltpu.make_async_copy(
            src_ref=tokens_local_hbm, dst_ref=tokens_local_vmem,
            sem=local_sems.at[0, 3]).start()

    def wait_fetch_b_tokens():
        pltpu.make_async_copy(
            src_ref=tokens_local_hbm, dst_ref=tokens_local_vmem,
            sem=local_sems.at[0, 3]).wait()

    def start_fetch_b_gating():
        pltpu.make_async_copy(
            src_ref=route_in_hbm, dst_ref=router_w_vmem,
            sem=local_sems.at[0, 0]).start()

    def wait_fetch_b_gating():
        pltpu.make_async_copy(
            src_ref=route_in_hbm, dst_ref=router_w_vmem,
            sem=local_sems.at[0, 0]).wait()

    def compute_routing(row0):
        """PROLOGUE 1 (grid step 0): router logits -> top-k.

        Reads tokens_local [T/P, D] and router_w [E, D] (fused router:
        one MXU op gives logits [E, T/P]) - or, on the serving path, the
        upstream logits [T/P, E] via one f32 transpose. Writes this
        device's rows [row0 : row0+T/P] of topk_idx [T, k] i32 and
        topk_w [T, k] f32."""
        with jax.named_scope("moe_routing"):
            if router_fused:
                logits_et = lax.dot_general(
                    router_w_vmem[...],      # [E, D]
                    tokens_local_vmem[...],  # [T/P, D]
                    dimension_numbers=(((1,), (1,)), ((), ())),
                    preferred_element_type=jnp.float32,
                )                            # -> [E, T/P]
            else:
                # serving path: the router ran upstream, the second input
                # is its logits [T/P, E]; one f32 transpose to [E, T/P]
                logits_et = router_w_vmem[...].astype(jnp.float32).T
            topk_weights, topk_indices = _routing(
                logits_et=logits_et,
                top_k=top_k,
                renormalize_topk_logits=renormalize_topk_logits,
            )                                # -> [T/P, k] each
            topk_idx_vmem[pl.ds(row0, num_local_tokens), :] = topk_indices
            topk_w_vmem[pl.ds(row0, num_local_tokens), :] = topk_weights

    def start_ag(my_id, row0):
        """PROLOGUE 2 (grid step 0): start the VMEM-direct all-gather.

        Local: tokens_local [T/P, D] -> tokens_full[row0:row0+T/P],
        then 3 remote copies per peer (token shard, topk_idx rows,
        topk_w rows) straight into each peer's VMEM - no HBM staging;
        the raw gating never exists outside the kernel. Returns with
        the copies IN FLIGHT. Caller MUST have passed sync_barrier()
        first: no peer VMEM may be written before its owner enters the
        kernel (the barrier is hoisted into the prologue so it overlaps
        the input fetches)."""
        with jax.named_scope("moe_ag_local"):
            tokens_full_vmem[pl.ds(row0, num_local_tokens), :] = (
                tokens_local_vmem[...])

        if not do_ag:
            return
        with jax.named_scope("moe_ag_remote"):
            for peer in range(num_devices):
                @pl.when(peer != my_id)
                def _send(peer=peer):
                    rows = pl.ds(row0, num_local_tokens)
                    for src, dst, rsem in (
                        (tokens_local_vmem,
                         tokens_full_vmem.at[rows, :],
                         recv_sem_tokens),
                        (topk_idx_vmem.at[rows, :],
                         topk_idx_vmem.at[rows, :],
                         recv_sem_meta),
                        (topk_w_vmem.at[rows, :],
                         topk_w_vmem.at[rows, :],
                         recv_sem_meta),
                    ):
                        pltpu.make_async_remote_copy(
                            src_ref=src,
                            dst_ref=dst,
                            send_sem=send_sem,
                            recv_sem=rsem,
                            device_id=get_mesh_device_id(peer),
                            device_id_type=pl.DeviceIdType.MESH,
                        ).start()

    # A wait consumes semaphore credits equal to the dummy ref's byte
    # count (the ref's ADDRESS is meaningless) - so one wait shaped
    # "all P-1 inbound shards" drains a buffer in a single call.

    def wait_ag_meta():
        """PROLOGUE 3: settle ONLY the [T, k] top-k arrivals - the mask
        builder needs these small tables, not the token shards, which
        keep streaming underneath."""
        with jax.named_scope("moe_ag_drain_meta"):
            # ONLY the top-k arrivals: everything up to the gather matmul
            # depends on these and not on the (much larger) token shards,
            # which keep streaming underneath.
            inbound_rows = pl.ds(0, inbound * num_local_tokens)
            for buf in (topk_idx_vmem, topk_w_vmem):
                pltpu.make_async_copy(
                    buf.at[inbound_rows, :],
                    buf.at[inbound_rows, :],
                    recv_sem_meta).wait()

    def wait_ag_tokens():
        """PROLOGUE 5 (its own pl.when, as late as possible): settle the
        [T, D] token arrivals + send-side hygiene. The first reader of
        tokens_full is the gather matmul."""
        with jax.named_scope("moe_ag_drain_tokens"):
            inbound_rows = pl.ds(0, inbound * num_local_tokens)
            for sem in (recv_sem_tokens, send_sem):
                pltpu.make_async_copy(
                    tokens_full_vmem.at[inbound_rows, :],
                    tokens_full_vmem.at[inbound_rows, :],
                    sem).wait()
            # send-side hygiene: the top-k sends complete too
            for buf in (topk_idx_vmem, topk_w_vmem):
                pltpu.make_async_copy(
                    buf.at[inbound_rows, :],
                    buf.at[inbound_rows, :], send_sem).wait()

    def init_dispatch_tables():
        """PROLOGUE 4: idx_kt [k, T] = topk_idx.T - one transpose here
        beats one per expert per grid step. (acc is zeroed separately,
        BEFORE the meta wait: it needs no data, so it fills the ICI
        arrival latency instead of following it.)"""
        with jax.named_scope("moe_topk_transpose"):
            # the per-step mask builder reduces over k with T on LANES;
            # one transpose here beats one per expert per grid step.
            idx_kt_vmem[...] = topk_idx_vmem[...].T          # [k, T]

    def build_dispatch(first_expert):
        """STEP 1 fill form (VALU + XLU, unskewed): per-expert dispatch
        pieces for experts first_expert..first_expert+be-1. Used only at
        grid step 0, which has no predecessor to build its slots (there
        is nothing to overlap with yet). Returns
        [(OH [C, T], OHG_T [T, C])] * be in act dtype."""
        with jax.named_scope("moe_masks"):
            tables = (idx_kt_vmem[...],                      # [k, T]
                      topk_idx_vmem[...],                    # [T, k]
                      topk_w_vmem[...])                      # [T, k]
            pieces = []
            for local_e_id in range(be):
                membership = dispatch_phase_a_valu(
                    first_expert + local_e_id, tables)
                slots = dispatch_phase_b_xlu(membership)
                pieces.append(dispatch_phase_c_valu(membership, slots))
            return pieces

    def dispatch_phase_a_valu(expert_id, tables):
        """STEP 1a (VALU): membership of ONE next-step expert - table
        compares + reductions. tables = (idx_kt [k, T], topk_idx [T, k],
        topk_w [T, k]), loaded once per step by the caller. Phantom pad
        rows (row blocks padded to the sublane tile) are masked out of
        membership here - their topk entries are uninitialized VMEM.
        Returns (routed_mask [1, T], live_t [T, 1], gate_t [T, 1])."""
        topk_idx_kt, topk_idx_tk, topk_w_tk = tables
        with jax.named_scope("moe_masks"):
            routed_mask, live_t, gate_t = _mask_membership(
                topk_idx_kt=topk_idx_kt,
                topk_idx_tk=topk_idx_tk,
                topk_w_tk=topk_w_tk,
                expert_id=expert_id)
            if padded_rows != num_local_tokens:
                lane_pos = lax.broadcasted_iota(
                    jnp.int32, (1, num_tokens), 1)
                routed_mask = routed_mask * (
                    lax.rem(lane_pos, padded_rows)
                    < num_local_tokens).astype(routed_mask.dtype)
                sub_pos = lax.broadcasted_iota(
                    jnp.int32, (num_tokens, 1), 0)
                live_t = live_t & (
                    lax.rem(sub_pos, padded_rows) < num_local_tokens)
            return routed_mask, live_t, gate_t

    def dispatch_phase_b_xlu(membership):
        """STEP 1b (XLU): the lane prefix-sum + transpose for ONE
        next-step expert. Returns (slot [1, T], slot_t [T, 1])."""
        with jax.named_scope("moe_masks"):
            return _mask_slots(membership[0], num_tokens)

    def dispatch_phase_c_valu(membership, slots):
        """STEP 1c (VALU): the one-hot operator selects for ONE
        next-step expert. Returns (OH [C, T], OHG_T [T, C])."""
        routed_mask, live_t, gate_t = membership
        slot, slot_t = slots
        with jax.named_scope("moe_masks"):
            return _mask_operators(
                routed_mask=routed_mask, live_t=live_t, gate_t=gate_t,
                slot=slot, slot_t=slot_t, num_tokens=num_tokens,
                capacity=capacity, act_dtype=act_dtype)

    def zero_pieces():
        """Ablate stub: the piece structure with zero operators."""
        return [(jnp.zeros((capacity, num_tokens), act_dtype),
                 jnp.zeros((num_tokens, capacity), act_dtype))] * be

    def park_dispatch_mxu(pieces, x_slot, ohg_slot):
        """STEP 1d (MXU + vst): finish a step's dispatch from its pieces.
        gather: x[x_slot] [be*C, D] = concat(OH) [be*C, T] @ tokens_full
        [T, D] - ONE matmul, the stationary-operand fills amortise across
        the whole block. park: ohg[ohg_slot] [T, be*C] = concat(OHG_T)
        for that step's group combine. Dynamic LEADING slot indices are
        the v1 .at[sem_id] idiom; tiled minor dims stay static."""
        if do_gather:
            with jax.named_scope("moe_gather"):
                oh_block = jnp.concatenate(
                    [oh for oh, _ in pieces], axis=0)        # [be*C, T]
                x_vmem.at[x_slot][...] = jnp.dot(
                    oh_block, tokens_full_vmem[...],
                    preferred_element_type=jnp.float32,
                ).astype(act_dtype)                          # [be*C, D]
        with jax.named_scope("moe_ohg_park"):
            ohg_vmem.at[ohg_slot][...] = jnp.concatenate(
                [ohg_t for _, ohg_t in pieces], axis=1)      # [T, be*C]

    def expert_gmm1_mxu(local_e_id: int, x_step, w1_step):
        """STEP 4a (MXU): gmm1 for ONE expert - rows
        [local_e_id*C : +C] of x_step [be*C, D] @ w1_step [be, D, 2I]
        (this step's slab, dynamic parity slot bound by the caller).
        bd1c chunks the contraction. Returns gate_up [C, 2I] f32."""
        rows = slice(local_e_id * capacity,
                     (local_e_id + 1) * capacity)
        with jax.named_scope("moe_gmm1"):
            k_chunk = bd1c or hidden_size
            gate_up = jnp.zeros((capacity, w1_hbm.shape[-1]),
                                jnp.float32)                 # [C, 2I]
            for k0 in range(0, hidden_size, k_chunk):
                gate_up = gate_up + jnp.dot(
                    x_step[rows, k0:k0 + k_chunk],           # [C, k_chunk]
                    w1_step[local_e_id, k0:k0 + k_chunk, :], # [k_chunk, 2I]
                    preferred_element_type=jnp.float32)
            return gate_up

    def expert_act_eup(gate_up):
        """STEP 4b (EUP): SwiGLU - silu(gate) * up, exp2 + reciprocal on
        the transcendental unit. Own trace scope so its cost is visible
        (gates the F sub-skew decision). gate_up [C, 2I] f32 ->
        act_out [C, I] act dtype."""
        with jax.named_scope("moe_act"):
            inter_size = gate_up.shape[-1] // 2              # I
            return apply_act_fn(gate_up[:, :inter_size],
                                gate_up[:, inter_size:],
                                "silu").astype(act_dtype)

    def expert_gmm2_mxu(local_e_id: int, act_out, y_step, w2_step):
        """STEP 4c (MXU): gmm2 for ONE expert - act_out [C, Ipad]
        sliced to w2's UNPADDED rows @ w2_step [be, I, D] (this step's
        slab) -> rows [local_e_id*C : +C] of y_step [be*C, D]. Serving
        pads w13's per-shard intermediate to 128 but leaves w2 at the
        real I (30B: 96 vs 128); the pad columns of act_out are exact
        zeros (silu(0)*0), so the slice drops nothing. bd2c chunks the
        output."""
        rows = slice(local_e_id * capacity,
                     (local_e_id + 1) * capacity)
        i_w2 = w2_hbm.shape[1]           # unpadded per-shard I
        with jax.named_scope("moe_gmm2"):
            n_chunk = bd2c or hidden_size
            for n0 in range(0, hidden_size, n_chunk):
                y_step[rows, n0:n0 + n_chunk] = jnp.dot(
                    act_out[:, :i_w2],
                    w2_step[local_e_id, :, n0:n0 + n_chunk], # [I, n_chunk]
                    preferred_element_type=jnp.float32).astype(act_dtype)

    def combine_group():
        """STEP 5 (group end): acc [T, D] f32 += sum over the group's bg
        steps of OHG_T(p) [T, be*C] @ y(p) [be*C, D], with each step's
        OHG_T read from its ring slot (written one step ahead).

        Scatter-add as matmuls: OHG_T carries the gate weights, so
        this both routes rows home and applies the top-k weighting. Once
        per GROUP, with the bg partial products summed in REGISTERS
        (static p, unrolled - like gmm1's h) so the accumulator sees ONE
        read-modify-write per T-chunk per group instead of one per step.

        CHUNKED over T: a full-width `acc[...] = acc[...] + dot(...)`
        emits every load, then every add, then every store - three phases
        each far wider than the scheduler's window, so nothing can
        overlap. Chunking puts vld + valu + vst work in every window
        instead. bcT is the tunable; None = one full-width expression
        (the old, unpipelineable form)."""
        with jax.named_scope("moe_combine"):
            t_chunk = bcT or num_tokens
            first_of_group = block_id - (bg - 1)
            for t0 in range(0, num_tokens, t_chunk):
                rows = pl.ds(t0, t_chunk)
                partial_sum = jnp.zeros((t_chunk, hidden_size), jnp.float32)
                for p_idx in range(bg):
                    buf_slot = lax.rem(first_of_group + p_idx, ohg_ring)
                    partial_sum = partial_sum + jnp.dot(
                        ohg_vmem[buf_slot, t0:t0 + t_chunk, :],
                        y_vmem[p_idx],
                        preferred_element_type=jnp.float32)
                acc_vmem[rows, :] = acc_vmem[rows, :] + partial_sum

    def start_send_bo():
        """v1's start_send_bo: ONE DMA out_stage -> out HBM."""
        pltpu.make_async_copy(
            src_ref=out_stage_vmem, dst_ref=out_hbm,
            sem=local_sems.at[0, 4]).start()

    def wait_send_bo():
        pltpu.make_async_copy(
            src_ref=out_stage_vmem, dst_ref=out_hbm,
            sem=local_sems.at[0, 4]).wait()

    def emit_output():
        """EPILOGUE (last grid step): out_stage [T, D] = acc in act
        dtype, then ONE DMA out_stage -> out HBM, waited before exit.
        The wait is legitimately terminal: it is the kernel's last
        instruction, the kernel may not exit with the copy in flight,
        and out's only consumer (the caller's reduce-scatter) is
        outside. Nothing in-kernel ever reads out back - acc and every
        cross-step buffer stay resident in VMEM scratch."""
        out_stage_vmem[...] = acc_vmem[...].astype(out_stage_vmem.dtype)
        start_send_bo()
        wait_send_bo()

    # ---- driver ----

    # grid step 0: local routing -> AG. The token drain sits in its OWN
    # pl.when so it settles as LATE as possible - everything before the
    # gather matmul depends only on the (small, drained-early) top-k
    # tables while the token shards keep streaming underneath.
    @pl.when(block_id == 0)
    def _prologue():
        # DMA starts first (weight slab 12 MiB, inputs 4.5 MiB), then
        # the one prologue action that needs NO data - the global
        # barrier - runs under them; the input wait sits at first use
        # (the routing matmul reads both buffers). The barrier being
        # done also means start_ag's remote sends fire the moment
        # routing finishes. Block 0's weight slab keeps streaming under
        # routing + AG.
        if do_weights:
            start_fetch_bw(0, 0)   # block 0 streams under routing + AG
        start_fetch_b_tokens()
        start_fetch_b_gating()
        if padded_rows != num_local_tokens:
            # phantom pad rows must be REAL zeros: the gather multiplies
            # them by zeroed mask columns and 0 * garbage(NaN/inf) = NaN.
            # Static per-device slices; placed BEFORE the barrier so no
            # peer's remote write can race the zeroing.
            gap = padded_rows - num_local_tokens
            for peer in range(num_devices):
                pad0 = peer * padded_rows + num_local_tokens
                tokens_full_vmem[pl.ds(pad0, gap), :] = jnp.zeros(
                    (gap, hidden_size), tokens_full_vmem.dtype)
        if do_ag:
            with jax.named_scope("moe_ag_barrier"):
                sync_barrier()
        wait_fetch_b_tokens()
        wait_fetch_b_gating()
        my_id = lax.axis_index(axis_name)
        row0 = my_id * padded_rows
        if do_routing:
            compute_routing(row0)
        start_ag(my_id, row0)
        acc_vmem[...] = jnp.zeros_like(acc_vmem)   # data-free, pre-wait
        if do_ag:
            wait_ag_meta()
        init_dispatch_tables()

    @pl.when(block_id == 0)
    def _settle_tokens():
        if do_ag:
            wait_ag_tokens()

    # ---- FILL stage (grid step 0 only): no predecessor built x[0] and
    # ohg[0], so step 0 builds its own dispatch, unskewed. Runs AFTER
    # _settle_tokens: the gather is the first tokens_full reader.
    if do_park:
        @pl.when(block_id == 0)
        def _dispatch_first():
            park_dispatch_mxu(
                build_dispatch(0) if do_masks else zero_pieces(),
                x_slot=0, ohg_slot=0)

    # ---- blocked weight fetch: start the NEXT block's slabs, then wait
    # THIS block's, both at the step top - the F/A/B/C emission below
    # stays one basic block.
    if do_weights:
        @pl.when(block_id + 1 < num_blocks)
        def _prefetch_weights():
            start_fetch_bw(block_id + 1, x_nxt)

        wait_fetch_bw(x_par)

    # ---- steady state: F(k) interleaved with the skewed A/B/C dispatch
    # of step k+1. ONE basic block - no pl.when: branches are scheduler
    # barriers. Emission column j carries MXU (F) + VALU (A, C) + XLU (B)
    # work from up to four distinct (step, expert) indices, so the
    # in-order machine can issue mask work into the FFN's MXU stalls.
    # (F stays contiguous per expert here; the F sub-skew - gmm1(j) /
    # act(j-1) / gmm2(j-2) - is a separate change gated on the moe_act
    # segment's measured size.)
    if do_masks:
        with jax.named_scope("moe_masks"):
            tables = (idx_kt_vmem[...],                      # [k, T]
                      topk_idx_vmem[...],                    # [T, k]
                      topk_w_vmem[...])                      # [T, k]
    a_out, b_out, pieces = {}, {}, []
    for j in range(be + 2):
        if do_ffn and j < be:
            gate_up = expert_gmm1_mxu(j, x_vmem.at[x_par],
                                      b_w1_x2_vmem.at[x_par])  # F(k, e_j)
            act_out = expert_act_eup(gate_up)
            expert_gmm2_mxu(j, act_out, y_vmem.at[group_pos],
                            b_w2_x2_vmem.at[x_par])
        if do_masks:
            if j < be:
                a_out[j] = dispatch_phase_a_valu(            # A(k+1, e_j)
                    next_first_expert + j, tables)
            if 1 <= j <= be:
                b_out[j - 1] = dispatch_phase_b_xlu(         # B(k+1, e_j-1)
                    a_out[j - 1])
            if 2 <= j <= be + 1:
                pieces.append(dispatch_phase_c_valu(         # C(k+1, e_j-2)
                    a_out[j - 2], b_out[j - 2]))
    if do_park:
        park_dispatch_mxu(pieces if do_masks else zero_pieces(),
                          x_slot=x_nxt, ohg_slot=ohg_next_slot)

    if do_combine:
        @pl.when(group_pos == bg - 1)
        def _combine():
            combine_group()

    # ---- last grid step: emit ----
    @pl.when(block_id == num_blocks - 1)
    def _epilogue():
        emit_output()


@functools.partial(
    jax.jit,
    static_argnames=("router_fused", "mesh", "axis_name", "top_k",
                     "renormalize_topk_logits", "capacity", "be", "bg",
                     "bd1c", "bd2c", "bcT", "vmem_limit_bytes", "ablate",
                     "interpret"),
)
def fused_moe_decode_tp_fused(
    tokens_local: jax.Array,   # [T/P, D]
    route_in: jax.Array,       # router_fused: [E, D] router weight
                               # (replicated); else [T/P, E] upstream logits
    w1_local: jax.Array,       # [E, D, 2*I/P]
    w2_local: jax.Array,       # [E, I/P, D]
    *,
    router_fused: bool = True,
    mesh: jax.sharding.Mesh,
    axis_name: str,
    top_k: int,
    renormalize_topk_logits: bool = True,
    capacity: int = 32,
    be: int = 4,
    bg: int = 1,
    bd1c: int | None = None,
    bd2c: int | None = None,
    bcT: int | None = None,
    vmem_limit_bytes: int = 64 * 1024 * 1024,
    ablate: str = "none",
    interpret=False,
) -> jax.Array:  # [T/P, D] this device's output rows (post reduce-scatter)
    """v0.2: router-fused + topk-first, the all-gather fused INTO the
    kernel as VMEM-direct ICI remote copies (per peer: token shard +
    [T/P, k] top-k results - the raw gating never exists outside the
    kernel), overlapping the first weight-block prefetch. Weights stream
    via manual blocked DMA fetches (double-buffered, one contiguous
    descriptor per slab; see
    fetch_weights), not BlockSpec windows. Exit stays an external
    psum_scatter (in-kernel RS = the next step, gmm_fused_rs-style
    direct writes)."""
    if interpret is True:
        # Remote DMAs/semaphore signals need the TPU-simulating interpret
        # machine (jax/_src/pallas/mosaic/interpret/); plain interpret=True
        # routes to the discharge interpreter, which raises "Remote signal
        # not implemented" / cannot take mesh device ids.
        interpret = pltpu.InterpretParams(dma_execution_mode="on_wait")
    num_devices = mesh.shape[axis_name]                      # P
    num_local_tokens, hidden_size = tokens_local.shape       # T/P, D
    # pad each device's row block to the sublane tile: serving calls
    # with as few as 2 rows/device, and the kernel's dynamic row0
    # (my_id * padded_rows) must be provably 16-aligned (E2003). All
    # T-sized buffers are allocated padded; phantom rows are masked out
    # of dispatch and sliced off after the reduce-scatter. When T/P is
    # already aligned (the bench shapes) this is the identity.
    padded_rows = -(-num_local_tokens // 16) * 16
    num_tokens = padded_rows * num_devices                   # T (padded)
    num_experts, _, inter2 = w1_local.shape                  # E, _, 2*I/P
    # serving pads w13's per-shard I to 128 but not w2's (30B: w13 gives
    # 128/projection, w2 has 96 rows) - w2 buffers size from w2 itself
    i_w2 = w2_local.shape[1]
    assert i_w2 <= inter2 // 2, (w2_local.shape, inter2)
    assert num_experts % be == 0, (num_experts, be)
    assert (hidden_size % (bd1c or hidden_size) == 0
            and hidden_size % (bd2c or hidden_size) == 0), (
                hidden_size, bd1c, bd2c)
    assert num_tokens % (bcT or num_tokens) == 0, (num_tokens, bcT)
    if router_fused:
        assert route_in.shape == (num_experts, hidden_size), (
            route_in.shape, num_experts, hidden_size)        # [E, D]
    else:
        assert route_in.shape == (num_local_tokens, num_experts), (
            route_in.shape, num_local_tokens, num_experts)   # [T/P, E]
    num_blocks = num_experts // be
    # groups of bg grid steps share one dispatch/combine
    assert num_blocks % bg == 0, (num_blocks, bg)

    # Static VMEM budget: every scratch buffer is a compile-time
    # constant, so fail fast with an itemized sum instead of a backend
    # allocation error. NO Pallas windows remain (all IO is HBM refs +
    # manual DMA - synchronous-mode windows were the per-step floor);
    # weights stream through x2 buffers of whole expert-block slabs
    # (waited once per step, keeping the emission one basic block).
    act_bytes = jnp.dtype(tokens_local.dtype).itemsize
    weight_block = be * (hidden_size * inter2
                         + i_w2 * hidden_size) * act_bytes
    vmem_need = (
        2 * weight_block                     # w1+w2 slab x2 buffers
        + num_local_tokens * hidden_size * act_bytes   # tokens_local copy
        + route_in.size * route_in.dtype.itemsize  # router_w/gating copy
        + num_tokens * hidden_size * act_bytes         # out staging
        + num_tokens * hidden_size * act_bytes         # tokens_full
        + num_tokens * hidden_size * 4                 # acc (f32)
        + (2 + bg) * be * capacity * hidden_size * act_bytes  # x(2) + y
        + 2 * bg * num_tokens * be * capacity * act_bytes  # ohg ring
        + 3 * num_tokens * top_k * 4           # topk tables, both layouts
    )
    assert vmem_need <= vmem_limit_bytes, (
        f"static VMEM need {vmem_need / 2**20:.1f} MiB exceeds "
        f"{vmem_limit_bytes / 2**20:.0f} MiB "
        f"(weight block {2 * weight_block / 2**20:.1f}, be={be}, bg={bg} "
        f"with 2x x-parity and 2*bg ohg-ring slots; "
        f"try a smaller be, bg or capacity)")

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        grid=(num_blocks,),
        in_specs=[
            # EVERYTHING in HBM, all movement manual (v1's structure).
            # A constant-index-map window is pipelined by Mosaic in
            # `synchronous` mode - per-step harness machinery that
            # measured 2777us/kernel (ablate=all) for buffers that are
            # per-kernel constants. The weights stream via the blocked
            # fetch (see fetch_weights); tokens/route are fetched once
            # in the prologue; out is DMA'd once in the epilogue.
            pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
        scratch_shapes=[
            pltpu.VMEM((num_tokens, hidden_size),
                       tokens_local.dtype),                  # tokens_full
            pltpu.SemaphoreType.DMA,                         # send_sem
            pltpu.SemaphoreType.DMA,                         # recv_sem_tokens
            pltpu.SemaphoreType.DMA,                         # recv_sem_meta
            pltpu.VMEM((num_tokens, hidden_size), jnp.float32),      # acc
            pltpu.VMEM((num_tokens, top_k), jnp.int32),      # topk_idx
            pltpu.VMEM((num_tokens, top_k), jnp.float32),    # topk_w
            pltpu.VMEM((top_k, num_tokens), jnp.int32),      # idx_kt
            pltpu.VMEM((2, be * capacity, hidden_size),
                       tokens_local.dtype),                  # x, parity
            pltpu.VMEM((bg, be * capacity, hidden_size),
                       tokens_local.dtype),                  # y
            pltpu.VMEM((2 * bg, num_tokens, be * capacity),
                       tokens_local.dtype),                  # ohg ring
            pltpu.VMEM((2, be, hidden_size, inter2),
                       w1_local.dtype),                      # b_w1_x2_vmem
            pltpu.VMEM((2, be, i_w2, hidden_size),
                       w2_local.dtype),                      # b_w2_x2_vmem
            pltpu.SemaphoreType.DMA((2, 5)),                 # local_sems
            pltpu.VMEM(tokens_local.shape,
                       tokens_local.dtype),                  # tokens copy
            pltpu.VMEM(route_in.shape, route_in.dtype),      # route copy
            pltpu.VMEM((num_tokens, hidden_size),
                       tokens_local.dtype),                  # out staging
        ],
    )
    kernel = functools.partial(
        _decode_moe_kernel,
        axis_name=axis_name,
        mesh_axis_names=tuple(mesh.axis_names),
        num_experts=num_experts,
        be=be,
        bg=bg,
        capacity=capacity,
        top_k=top_k,
        renormalize_topk_logits=renormalize_topk_logits,
        act_dtype=tokens_local.dtype,
        router_fused=router_fused,
        bd1c=bd1c,
        bd2c=bd2c,
        bcT=bcT,
        ablate=ablate,
    )
    out = pl.pallas_call(
        kernel,
        grid_spec=grid_spec,
        out_shape=jax.ShapeDtypeStruct((num_tokens, hidden_size),
                                   tokens_local.dtype),
        compiler_params=pltpu.CompilerParams(
            # ablate=ag removes the barrier, and Mosaic rejects a
            # collective_id without one
            **({} if ablate == "ag" else {"collective_id": 13}),
            vmem_limit_bytes=vmem_limit_bytes,
        ),
        interpret=interpret,
    )(
      # pin every operand in HBM (v1's idiom): without the constraint
      # XLA may try to stage operands into VMEM
      pltpu.with_memory_space_constraint(tokens_local, pltpu.HBM),
      pltpu.with_memory_space_constraint(route_in, pltpu.HBM),
      pltpu.with_memory_space_constraint(w1_local, pltpu.HBM),
      pltpu.with_memory_space_constraint(w2_local, pltpu.HBM))
    out = lax.psum_scatter(out, axis_name, scatter_dimension=0, tiled=True)
    return out[:num_local_tokens]        # drop the phantom pad rows


def fused_moe_decode_tp_serving(
    hidden_states: jax.Array,  # [T, D] rows sharded over axis_name
    gating_output: jax.Array,  # [T, E] router logits, same sharding
    w1: jax.Array,             # [E, D, 2I] sharded on the last axis
    w2: jax.Array,             # [E, I, D] sharded on the middle axis
    *,
    mesh: jax.sharding.Mesh,
    axis_name: str,
    top_k: int,
    renormalize_topk_logits: bool,
    capacity: int,
    be: int = 4,
    bg: int = 1,
    interpret: bool = False,
) -> jax.Array:  # [T, D] rows sharded over axis_name
    """Serving entry, called from moe_apply inside the model's jit: the
    router already ran upstream, so the kernel gets its logits instead
    of the router weight. shard_map over the ONE mesh axis that carries
    both the token shards and the weight shards (TP-MoE under
    data-parallel attention).

    TRACE-time only: this Python stages the shard_map -> pallas_call ->
    psum_scatter chain into the compiled program; nothing here executes
    on the host at serving time.

    Keep renormalize_topk_logits=True where the model allows: it is also
    a perf choice - the kernel then pays O(k) transcendentals per token
    in the prologue instead of an O(E)-wide softmax."""

    # the fused entry's positional order IS the shard_map operand order,
    # so binding the static config is all it takes
    P = jax.sharding.PartitionSpec
    return jax.shard_map(
        functools.partial(
            fused_moe_decode_tp_fused,
            router_fused=False,
            mesh=mesh,
            axis_name=axis_name,
            top_k=top_k,
            renormalize_topk_logits=renormalize_topk_logits,
            capacity=capacity,
            be=be,
            bg=bg,
            interpret=interpret,
        ),
        mesh=mesh,
        in_specs=(P(axis_name, None),
                  P(axis_name, None),
                  P(None, None, axis_name),
                  P(None, axis_name, None)),
        out_specs=P(axis_name, None),
        check_vma=False,
    )(hidden_states, gating_output, w1, w2)
