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
    tokens_local_vmem,   # [T/P, D] this device's token shard
    router_w_vmem,       # [E, D] router weight, or [T/P, E] logits when
                         # not router_fused (upstream [out, in] layout)
    w1_hbm,              # HBM [E, D, 2I] - streamed via the manual ring
    w2_hbm,              # HBM [E, I, D]
    out_vmem,            # [T, D] union partials (caller reduce-scatters)
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
    w1_ring,             # VMEM [2, be, D, 2I] manual weight fetch ring,
                         # parity slots like x: slot k%2 read at step k
    w2_ring,             # VMEM [2, be, I, D]
    w_sems,              # DMA [2, 2] - [ring slot, tensor (w1/w2)]
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
    # The weight fetch ring is NEVER ablated (starts and waits must stay
    # paired), so ablate=weights = all compute stubbed with the ring still
    # streaming: it measures the manual ring's in-situ bandwidth.
    do_masks = ablate not in ("masks", "weights")
    do_gather = ablate not in ("gather", "weights")
    do_ffn = ablate not in ("ffn", "weights")
    do_combine = ablate not in ("combine", "weights")

    block_id = pl.program_id(0)          # which expert block (grid step)
    num_blocks = pl.num_programs(0)
    num_tokens, hidden_size = tokens_full_vmem.shape         # T, D
    # static under shard_map - same axis env as lax.axis_index below
    num_devices = lax.axis_size(axis_name)                   # P
    num_local_tokens = tokens_local_vmem.shape[0]            # T/P
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

    def fetch_weights(block, slot, *, wait=False):
        """STEP 0 (DMA): stream expert block `block` - w1 [be, D, 2I]
        and w2 [be, I, D] slabs, each as ONE contiguous descriptor into
        ring slot `slot` (v1's paired start/wait fetch shape).

        One flat descriptor is the point: the DMA access-size law prices
        a descriptor by its inner contiguous run, and a be-expert slab is
        a single maximal run (measured 2179 GB/s vs the window pipeline's
        strided copies at 494 - the 4.4x this ring exists for). Callers
        fence out-of-range prefetches (OOB DMA reads are fatal); the
        fence keeps start/wait counts balanced by construction."""
        for hbm, ring, s in ((w1_hbm, w1_ring, 0), (w2_hbm, w2_ring, 1)):
            cp = pltpu.make_async_copy(
                src_ref=hbm.at[pl.ds(block * be, be)],
                dst_ref=ring.at[slot],
                sem=w_sems.at[slot, s])
            if wait:
                cp.wait()
            else:
                cp.start()

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

        Local: tokens_local [T/P, D] -> tokens_full[row0:row0+T/P].
        Then a global barrier, then 3 remote copies per peer (token
        shard, topk_idx rows, topk_w rows) straight into each peer's
        VMEM - no HBM staging; the raw gating never exists outside the
        kernel. Returns with the copies IN FLIGHT."""
        with jax.named_scope("moe_ag_local"):
            tokens_full_vmem[pl.ds(row0, num_local_tokens), :] = (
                tokens_local_vmem[...])

        with jax.named_scope("moe_ag_barrier"):
            sync_barrier()

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
        """PROLOGUE 4: idx_kt [k, T] = topk_idx.T (one transpose here
        beats one per expert per grid step) and zero acc [T, D] f32."""
        with jax.named_scope("moe_topk_transpose"):
            # the per-step mask builder reduces over k with T on LANES;
            # one transpose here beats one per expert per grid step.
            idx_kt_vmem[...] = topk_idx_vmem[...].T          # [k, T]
        acc_vmem[...] = jnp.zeros_like(acc_vmem)

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
        topk_w [T, k]), loaded once per step by the caller. Returns
        (routed_mask [1, T], live_t [T, 1], gate_t [T, 1])."""
        topk_idx_kt, topk_idx_tk, topk_w_tk = tables
        with jax.named_scope("moe_masks"):
            return _mask_membership(
                topk_idx_kt=topk_idx_kt,
                topk_idx_tk=topk_idx_tk,
                topk_w_tk=topk_w_tk,
                expert_id=expert_id)

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
        (this step's ring slot, dynamic leading slot bound by the
        caller). Returns gate_up [C, 2I] f32. bd1c chunks the
        contraction."""
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
        """STEP 4c (MXU): gmm2 for ONE expert - act_out [C, I] @
        w2_step [be, I, D] (this step's ring slot) -> rows
        [local_e_id*C : +C] of y_step [be*C, D]. bd2c chunks the
        output."""
        rows = slice(local_e_id * capacity,
                     (local_e_id + 1) * capacity)
        with jax.named_scope("moe_gmm2"):
            n_chunk = bd2c or hidden_size
            for n0 in range(0, hidden_size, n_chunk):
                y_step[rows, n0:n0 + n_chunk] = jnp.dot(
                    act_out,
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
                    ring_slot = lax.rem(first_of_group + p_idx, ohg_ring)
                    partial_sum = partial_sum + jnp.dot(
                        ohg_vmem[ring_slot, t0:t0 + t_chunk, :],
                        y_vmem[p_idx],
                        preferred_element_type=jnp.float32)
                acc_vmem[rows, :] = acc_vmem[rows, :] + partial_sum

    def emit_output():
        """EPILOGUE (last grid step): out [T, D] = acc, in act dtype.
        Partial sums under TP - the caller reduce-scatters."""
        out_vmem[...] = acc_vmem[...].astype(out_vmem.dtype)

    # ---- driver ----

    # grid step 0: local routing -> AG. The token drain sits in its OWN
    # pl.when so it settles as LATE as possible - everything before the
    # gather matmul depends only on the (small, drained-early) top-k
    # tables while the token shards keep streaming underneath.
    @pl.when(block_id == 0)
    def _prologue():
        # block 0's weight slab streams UNDER the routing matmul and the
        # all-gather (the window pipeline serialized this first fill
        # ahead of the kernel body).
        fetch_weights(0, 0)
        my_id = lax.axis_index(axis_name)
        row0 = my_id * num_local_tokens
        compute_routing(row0)
        start_ag(my_id, row0)
        wait_ag_meta()
        init_dispatch_tables()

    @pl.when(block_id == 0)
    def _settle_tokens():
        wait_ag_tokens()

    # ---- FILL stage (grid step 0 only): no predecessor built x[0] and
    # ohg[0], so step 0 builds its own dispatch, unskewed. Runs AFTER
    # _settle_tokens: the gather is the first tokens_full reader.
    @pl.when(block_id == 0)
    def _dispatch_first():
        park_dispatch_mxu(
            build_dispatch(0) if do_masks else zero_pieces(),
            x_slot=0, ohg_slot=0)

    # ---- weight ring: start next block's fetch (probe order: start,
    # THEN wait this block's slab, just before its first reader). The
    # pl.when fence suppresses the one out-of-range prefetch at the last
    # step; the wait is unfenced - its slab was fetched a step earlier.
    # Both sit BEFORE the F/A/B/C emission block, so the steady-state
    # compute stays one basic block.
    @pl.when(block_id + 1 < num_blocks)
    def _prefetch_weights():
        fetch_weights(block_id + 1, x_nxt)

    fetch_weights(block_id, x_par, wait=True)

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
                                      w1_ring.at[x_par])     # F(k, e_j)
            act_out = expert_act_eup(gate_up)
            expert_gmm2_mxu(j, act_out, y_vmem.at[group_pos],
                            w2_ring.at[x_par])
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
    through a manual DMA ring (one contiguous descriptor per slab; see
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
    num_tokens = num_local_tokens * num_devices              # T
    num_experts, _, inter2 = w1_local.shape                  # E, _, 2*I/P
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

    # Static VMEM budget: every window and scratch buffer is a
    # compile-time constant, so fail fast with an itemized sum instead of
    # a backend allocation error. The weight rings hold 2 slots (same
    # bytes the window pipeline's double-buffer used to take); the
    # constant-index windows are single-buffered.
    act_bytes = jnp.dtype(tokens_local.dtype).itemsize
    weight_block = be * (hidden_size * inter2
                         + (inter2 // 2) * hidden_size) * act_bytes
    vmem_need = (
        2 * weight_block                     # w1+w2 manual fetch rings
        + num_local_tokens * hidden_size * act_bytes   # tokens_local window
        + route_in.size * route_in.dtype.itemsize  # router_w/gating window
        + num_tokens * hidden_size * act_bytes         # out window
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
            pl.BlockSpec(tokens_local.shape, lambda i: (0, 0)),
            pl.BlockSpec(route_in.shape, lambda i: (0, 0)),
            # weights stay in HBM; the kernel streams them itself (manual
            # fetch ring) - the window pipeline's generated copies ran at
            # 494 GB/s vs 2179 for one contiguous descriptor per slab
            pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
        ],
        out_specs=pl.BlockSpec((num_tokens, hidden_size), lambda i: (0, 0)),
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
                       w1_local.dtype),                      # w1 ring
            pltpu.VMEM((2, be, inter2 // 2, hidden_size),
                       w2_local.dtype),                      # w2 ring
            pltpu.SemaphoreType.DMA((2, 2)),                 # w_sems
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
            collective_id=13,
            vmem_limit_bytes=vmem_limit_bytes,
        ),
        interpret=interpret,
    )(tokens_local, route_in,
      # pin the streamed weights in HBM (v1's idiom): without the
      # constraint XLA may try to stage the operands into VMEM
      pltpu.with_memory_space_constraint(w1_local, pltpu.HBM),
      pltpu.with_memory_space_constraint(w2_local, pltpu.HBM))
    return lax.psum_scatter(out, axis_name, scatter_dimension=0, tiled=True)


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
