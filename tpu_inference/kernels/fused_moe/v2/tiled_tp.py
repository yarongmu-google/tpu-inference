"""No-drop TP MoE with token-tiled VMEM routing and accumulation.

The grid visits (token tile, expert). Expert occupancy controls runtime
row loops, not storage capacity. Empty experts skip their weight loads.
Only inputs, expert weights and completed output tiles reside in HBM.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, PartitionSpec as P


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _routing(logits: jax.Array, *, top_k: int,
             renormalize: bool) -> tuple[jax.Array, jax.Array]:
    """Return [k, local_tokens] gates/ids; highest-expert-id tie break."""
    scores = logits.astype(jnp.float32).T
    if not renormalize:
        scores = jax.nn.softmax(scores, axis=0)
    expert_ids = lax.broadcasted_iota(jnp.int32, scores.shape, 0)
    values, indices = [], []
    for k in range(top_k):
        value = jnp.max(scores, axis=0, keepdims=True)
        index = jnp.max(jnp.where(scores == value, expert_ids, 0),
                        axis=0, keepdims=True)
        values.append(value)
        indices.append(index)
        if k + 1 != top_k:
            scores = jnp.where(expert_ids == index, -jnp.inf, scores)
    gates = jnp.concatenate(values, axis=0)
    if renormalize:
        gates = jax.nn.softmax(gates, axis=0)
    return gates, jnp.concatenate(indices, axis=0)


def _slots(mask: jax.Array) -> jax.Array:
    """Exclusive prefix sum, with tokens on the vector lane dimension."""
    prefix = mask
    lanes = lax.broadcasted_iota(jnp.int32, mask.shape, 1)
    shift = 1
    while shift < mask.shape[1]:
        prefix = prefix + jnp.where(
            lanes >= shift, pltpu.roll(prefix, shift=shift, axis=1), 0)
        shift *= 2
    return prefix - mask


def _kernel(tensor_amax, x_hbm, logits_hbm, w1_hbm, w2_hbm,
            s1_hbm, s2_hbm, out_hbm, x_local, logits_local,
            tokens, ids, gates, scales, ids_kt, gates_kt, scales_row,
            acc, output_tile, w1, w2, s1, s2, act, down,
            slot_row, live_row, weight_row,
            count_vmem, count_smem, local_sems, send_sems, recv_sems, *,
            axis_name: str, mesh_axis_names: tuple[str, ...], width: int,
            real_local_tokens: int, top_k: int, renormalize: bool,
            act_scale: str, fp8: bool, bf16_rows: int,
            combine_token_rows: int, combine_hidden_cols: int):
    tile, expert = pl.program_id(0), pl.program_id(1)
    local_rows, hidden = x_local.shape
    first_rows = 2 * bf16_rows if fp8 else bf16_rows
    rank = lax.axis_index(axis_name)
    row0 = rank * local_rows
    inter = w2.shape[0]

    def peer_id(peer):
        return tuple(jnp.int32(peer) if name == axis_name else lax.axis_index(name)
                     for name in mesh_axis_names)

    @pl.when(expert == 0)
    def initialize():
        # Each tile reuses symmetric VMEM. Wait until every peer has finished
        # consuming the previous tile before allowing new remote writes.
        if width > 1:
            barrier = pltpu.get_barrier_semaphore()
            for peer in range(width):
                pl.semaphore_signal(barrier, device_id=peer_id(peer),
                                    device_id_type=pl.DeviceIdType.MESH)
            pl.semaphore_wait(barrier, width)
        copies = [
            pltpu.make_async_copy(
                x_hbm.at[pl.ds(tile * local_rows, local_rows), :],
                x_local, local_sems.at[0]),
            pltpu.make_async_copy(
                logits_hbm.at[pl.ds(tile * local_rows, local_rows), :],
                logits_local, local_sems.at[1]),
        ]
        for copy in copies:
            copy.start()
        acc[...] = jnp.zeros(acc.shape, jnp.float32)
        for copy in copies:
            copy.wait()
        local_gates, local_ids = _routing(
            logits_local[...], top_k=top_k, renormalize=renormalize)
        if real_local_tokens % local_rows:
            rows = lax.broadcasted_iota(jnp.int32, local_ids.shape, 1)
            valid = (tile * local_rows + rows) < real_local_tokens
            local_ids = jnp.where(valid, local_ids, -1)
            local_gates = jnp.where(valid, local_gates, 0.)
        # Rank is a leading DMA dimension; no narrow minor-axis remote slices.
        ids[rank, :, :] = local_ids
        gates[rank, :, :] = local_gates
        xf = x_local[...].astype(jnp.float32)
        if fp8:
            if act_scale == "tensor":
                amax = jnp.full((local_rows, 1), tensor_amax[0], jnp.float32)
            else:
                amax = jnp.max(jnp.abs(xf), axis=1, keepdims=True)
            inv = jnp.where(amax > 0, 448. / amax, 0.)
            tokens[pl.ds(row0, local_rows), :] = jnp.clip(
                xf * inv, -448., 448.).astype(jnp.float8_e4m3fn)
            scales[rank, :, :] = (amax / 448.).T
        else:
            tokens[pl.ds(row0, local_rows), :] = x_local[...]
            scales[rank, :, :] = jnp.ones((1, local_rows), jnp.float32)
        if width > 1:
            shards = (tokens.at[pl.ds(row0, local_rows), :],
                      ids.at[rank], gates.at[rank], scales.at[rank])
            for peer in range(width):
                @pl.when(peer != rank)
                def send(peer=peer):
                    for index, shard in enumerate(shards):
                        pltpu.make_async_remote_copy(
                            src_ref=shard, dst_ref=shard,
                            send_sem=send_sems.at[index],
                            recv_sem=recv_sems.at[index], device_id=peer_id(peer),
                            device_id_type=pl.DeviceIdType.MESH).start()
            # DMA waits consume byte credits. These dummy slices describe
            # the aggregate traffic, not the addresses that were transferred.
            traffic = (tokens.at[pl.ds(0, (width - 1) * local_rows), :],
                       ids.at[pl.ds(0, width - 1)],
                       gates.at[pl.ds(0, width - 1)],
                       scales.at[pl.ds(0, width - 1)])
            for index, received in enumerate(traffic):
                for sem in (recv_sems.at[index], send_sems.at[index]):
                    pltpu.make_async_copy(received, received, sem).wait()
        # Assemble the lane axis once per tile, retaining [k, tokens].
        ids_kt[...] = jnp.concatenate([ids[peer, :, :] for peer in range(width)], axis=1)
        gates_kt[...] = jnp.concatenate([gates[peer, :, :] for peer in range(width)], axis=1)
        scales_row[...] = jnp.concatenate([scales[peer, :, :] for peer in range(width)], axis=1)

    hit = ids_kt[...] == expert
    mask = jnp.max(hit.astype(jnp.int32), axis=0, keepdims=True)
    gate_row = jnp.sum(jnp.where(hit, gates_kt[...], 0.), axis=0, keepdims=True)
    # The count is produced by the vector unit. Transfer it to on-chip SMEM
    # for scalar control flow; no expert rows or intermediates spill to HBM.
    count_vmem[...] = jnp.broadcast_to(jnp.sum(mask), count_vmem.shape)
    count_copy = pltpu.make_async_copy(count_vmem, count_smem, local_sems.at[0])
    count_copy.start()
    count_copy.wait()
    count = count_smem[0, 0]

    @pl.when(count > 0)
    def active_expert():
        # Skip BEFORE issuing any weight or scale DMA for an empty expert.
        copies = [pltpu.make_async_copy(w1_hbm.at[expert], w1, local_sems.at[0]),
                  pltpu.make_async_copy(w2_hbm.at[expert], w2, local_sems.at[1])]
        if fp8:
            copies += [pltpu.make_async_copy(s1_hbm.at[expert], s1, local_sems.at[2]),
                       pltpu.make_async_copy(s2_hbm.at[expert], s2, local_sems.at[3])]
        for copy in copies:
            copy.start()
        slot = _slots(mask)
        slot_row[...] = slot
        live_row[...] = mask
        weight_row[...] = gate_row
        for copy in copies:
            copy.wait()

        def row_step(block, unused):
            first = block * first_rows
            row_ids = first + lax.broadcasted_iota(
                jnp.int32, (first_rows, tokens.shape[0]), 0)
            selected = (slot == row_ids) & (mask != 0)
            # Both selects consume f32; convert values, not a shared Boolean
            # predicate, to the narrower gather operand dtype.
            onehot = jnp.where(selected, jnp.float32(1), jnp.float32(0))
            operand_dtype = jnp.float8_e4m3fn if fp8 else jnp.bfloat16
            with jax.named_scope("moe_gather"):
                gathered = jnp.dot(onehot.astype(operand_dtype), tokens[...],
                                   preferred_element_type=jnp.float32).astype(operand_dtype)
            with jax.named_scope("moe_gmm1"):
                gate_up = jnp.dot(gathered, w1[...],
                                  preferred_element_type=jnp.float32)
                if fp8:
                    sx = jnp.sum(jnp.where(selected, scales_row[...], 0.),
                                 axis=1, keepdims=True)
                    gate_up = gate_up * sx * s1[...]
            half = gate_up.shape[1] // 2
            with jax.named_scope("moe_act"):
                act[...] = (jax.nn.silu(gate_up[:, :half]) * gate_up[:, half:])[
                    :, :inter].astype(jnp.bfloat16)
            live = jnp.minimum(first_rows, count - first)

            def down_step(part, unused):
                with jax.named_scope("moe_gmm2"):
                    result = jnp.dot(act[pl.ds(part * bf16_rows, bf16_rows), :],
                                     w2[...], preferred_element_type=jnp.float32)
                    if fp8:
                        result = result * s2[...]
                    down[...] = result.astype(jnp.bfloat16)
                # Construct [rows, tokens] weights, never [tokens, 1] masks.
                # Contract rows directly. The implicit LHS transpose is a
                # matrix-boundary cost to inspect on the installed backend.
                total_rows = tokens.shape[0]
                ct = (min(total_rows, combine_token_rows)
                      if total_rows % combine_token_rows == 0 else total_rows)
                cd = min(hidden, combine_hidden_cols)

                def token_step(token_block, unused):
                    ts = pl.ds(token_block * ct, ct)
                    rows = first + part * bf16_rows + lax.broadcasted_iota(
                        jnp.int32, (bf16_rows, ct), 0)
                    chosen = ((slot_row[:, ts] == rows) & (live_row[:, ts] != 0))
                    combine = jnp.where(chosen, weight_row[:, ts], 0.).astype(jnp.bfloat16)

                    def hidden_step(hidden_block, unused):
                        ds = pl.ds(hidden_block * cd, cd)
                        with jax.named_scope("moe_combine"):
                            update = lax.dot_general(
                                combine, down[:, ds],
                                dimension_numbers=(((0,), (0,)), ((), ())),
                                preferred_element_type=jnp.float32)
                            acc[ts, ds] += update
                        return unused

                    return lax.fori_loop(0, hidden // cd, hidden_step, unused)

                if total_rows == ct:
                    return token_step(0, unused)
                return lax.fori_loop(0, total_rows // ct, token_step, unused)

            lax.fori_loop(0, (live + bf16_rows - 1) // bf16_rows,
                          down_step, unused)
            return unused

        lax.fori_loop(0, (count + first_rows - 1) // first_rows,
                      row_step, jnp.int32(0))

    @pl.when(expert == w1_hbm.shape[0] - 1)
    def finish():
        output_tile[...] = acc[...].astype(jnp.bfloat16)
        padded_local = x_hbm.shape[0]
        # Preserve rank-major order across tiles for the final reduce-scatter.
        for peer in range(width):
            copy = pltpu.make_async_copy(
                output_tile.at[pl.ds(peer * local_rows, local_rows), :],
                out_hbm.at[pl.ds(peer * padded_local + tile * local_rows,
                                  local_rows), :], local_sems.at[0])
            copy.start()
            copy.wait()


def _local_moe(x: jax.Array, gating: jax.Array, w1: jax.Array,
               w2: jax.Array, s1: jax.Array, s2: jax.Array, *,
               axis_name: str, mesh_axis_names: tuple[str, ...], width: int,
               top_k: int, renormalize: bool, act_scale: str,
               token_tile_size: int, bf16_rows: int, combine_token_rows: int,
               combine_hidden_cols: int, interpret: bool) -> jax.Array:
    fp8 = w1.dtype == jnp.float8_e4m3fn
    alignment = 32 if fp8 else 16
    real_local_tokens, hidden = x.shape
    local_rows = min(token_tile_size // width,
                     _align_up(real_local_tokens, alignment))
    padded_local = _align_up(real_local_tokens, local_rows)
    total_rows = width * local_rows
    num_tiles = padded_local // local_rows
    first_rows = 2 * bf16_rows if fp8 else bf16_rows
    inter2, inter = w1.shape[2], w2.shape[1]
    tensor_amax = (lax.pmax(jnp.max(jnp.abs(x.astype(jnp.float32))), axis_name)
                   if fp8 and act_scale == "tensor" else jnp.float32(0)).reshape(1)
    # Only pad local input tails for aligned DMA; no HBM routing/packing buffers.
    x = jnp.pad(x, ((0, padded_local - real_local_tokens), (0, 0)))
    gating = jnp.pad(gating, ((0, padded_local - real_local_tokens), (0, 0)))
    grid = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1, grid=(num_tiles, w1.shape[0]),
        in_specs=[pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)] * 6,
        out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
        scratch_shapes=(
            pltpu.VMEM((local_rows, hidden), x.dtype),
            pltpu.VMEM((local_rows, w1.shape[0]), gating.dtype),
            pltpu.VMEM((total_rows, hidden), w1.dtype),
            pltpu.VMEM((width, top_k, local_rows), jnp.int32),
            pltpu.VMEM((width, top_k, local_rows), jnp.float32),
            pltpu.VMEM((width, 1, local_rows), jnp.float32),
            pltpu.VMEM((top_k, total_rows), jnp.int32),
            pltpu.VMEM((top_k, total_rows), jnp.float32),
            pltpu.VMEM((1, total_rows), jnp.float32),
            pltpu.VMEM((total_rows, hidden), jnp.float32),
            pltpu.VMEM((total_rows, hidden), jnp.bfloat16),
            pltpu.VMEM((hidden, inter2), w1.dtype),
            pltpu.VMEM((inter, hidden), w2.dtype),
            pltpu.VMEM((1, inter2), jnp.float32),
            pltpu.VMEM((1, hidden), jnp.float32),
            pltpu.VMEM((first_rows, inter), jnp.bfloat16),
            pltpu.VMEM((bf16_rows, hidden), jnp.bfloat16),
            pltpu.VMEM((1, total_rows), jnp.int32),
            pltpu.VMEM((1, total_rows), jnp.int32),
            pltpu.VMEM((1, total_rows), jnp.float32),
            pltpu.VMEM((1, 128), jnp.int32),
            pltpu.SMEM((1, 128), jnp.int32),
            pltpu.SemaphoreType.DMA((4,)),
            pltpu.SemaphoreType.DMA((4,)),
            pltpu.SemaphoreType.DMA((4,)),
        ))
    partial = pl.pallas_call(
        functools.partial(_kernel, axis_name=axis_name,
                          mesh_axis_names=mesh_axis_names, width=width,
                          real_local_tokens=real_local_tokens, top_k=top_k,
                          renormalize=renormalize, act_scale=act_scale,
                          fp8=fp8, bf16_rows=bf16_rows,
                          combine_token_rows=combine_token_rows,
                          combine_hidden_cols=combine_hidden_cols),
        grid_spec=grid,
        out_shape=jax.ShapeDtypeStruct((width * padded_local, hidden), jnp.bfloat16),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
            collective_id=0, vmem_limit_bytes=64 * 1024 * 1024),
        interpret=(pltpu.InterpretParams(dma_execution_mode="on_wait")
                   if interpret else False),
    )(tensor_amax, x, gating, w1, w2, s1, s2)
    return lax.psum_scatter(partial, axis_name, scatter_dimension=0,
                            tiled=True)[:real_local_tokens]


def fused_moe_tp_tiled_serving(
    hidden_states: jax.Array, gating_output: jax.Array,
    w1: jax.Array, w2: jax.Array,
    w1_scale: jax.Array | None = None,
    w2_scale: jax.Array | None = None, *, mesh: Mesh, axis_name: str,
    top_k: int, renormalize_topk_logits: bool,
    act_scale: str = "token", token_tile_size: int = 1024,
    bf16_rows: int = 32, combine_token_rows: int = 128,
    combine_hidden_cols: int = 128, interpret: bool = False,
) -> jax.Array:
    """TP expert weights, DP token rows; accepts the existing serving layout.

    w13 is already arranged as [gate_rank | up_rank] within each TP shard.
    No weight transpose/requantization is performed in the execution path.
    token_tile_size bounds global rows in VMEM. GMM1 uses twice bf16_rows
    with FP8 weights; GMM2 always uses bf16_rows. Both sizes are static
    tuning choices; expert occupancy only changes device-side loop counts.
    """
    if hidden_states.dtype != jnp.bfloat16:
        raise ValueError("tiled TP MoE requires BF16 input activations")
    fp8 = w1.dtype == jnp.float8_e4m3fn
    if w1.dtype != w2.dtype or w1.dtype not in (jnp.bfloat16, jnp.float8_e4m3fn):
        raise ValueError("tiled TP MoE takes matching BF16 or e4m3 expert weights")
    if act_scale not in ("token", "tensor"):
        raise ValueError("act_scale must be token or tensor")
    t, d = hidden_states.shape
    e, dw, inter2 = w1.shape
    width = mesh.shape[axis_name]
    alignment = 32 if fp8 else 16
    if token_tile_size < width * alignment or token_tile_size % (width * alignment):
        raise ValueError("token_tile_size must align to mesh width times DMA row alignment")
    if bf16_rows < 16 or bf16_rows % 16:
        raise ValueError("bf16_rows must be a positive multiple of 16")
    if combine_token_rows < 128 or combine_token_rows % 128:
        raise ValueError("combine_token_rows must be a positive multiple of 128")
    if (combine_hidden_cols < 128 or combine_hidden_cols % 128
            or d % min(d, combine_hidden_cols)):
        raise ValueError("combine_hidden_cols must align to 128 and divide hidden")
    if t < 1 or t % width or gating_output.shape != (t, e) or dw != d:
        raise ValueError("token, router, weight or mesh shapes do not match")
    if any(mesh.shape[a] > 1 for a in mesh.axis_names if a != axis_name):
        raise ValueError("tiled TP MoE only supports sharding on its TP axis")
    if w2.shape[0] != e or w2.shape[2] != d or not 1 <= top_k <= e:
        raise ValueError("expert shapes or top_k do not match")
    if d % 128 or inter2 % (2 * width * 128) or w2.shape[1] % (width * 128):
        raise ValueError("local hidden and projection dimensions must align to 128")
    if w2.shape[1] > inter2 // 2:
        raise ValueError("w2 contraction cannot exceed the gate/up projection")
    if fp8:
        if (w1_scale is None or w2_scale is None
                or w1_scale.shape != (e, 1, 1, inter2)
                or w2_scale.shape != (e, 1, 1, d)):
            raise ValueError("FP8 weights require per-channel serving scales")
        s1 = w1_scale.reshape(e, 1, inter2).astype(jnp.float32)
        s2 = w2_scale.reshape(e, 1, d).astype(jnp.float32)
    else:
        if w1_scale is not None or w2_scale is not None:
            raise ValueError("BF16 weights must not carry quantization scales")
        s1 = jnp.ones((e, 1, inter2), jnp.float32)
        s2 = jnp.ones((e, 1, d), jnp.float32)
    fn = functools.partial(
        _local_moe, axis_name=axis_name, mesh_axis_names=tuple(mesh.axis_names),
        width=width, top_k=top_k, token_tile_size=token_tile_size,
        bf16_rows=bf16_rows, combine_token_rows=combine_token_rows,
        combine_hidden_cols=combine_hidden_cols,
        renormalize=renormalize_topk_logits, act_scale=act_scale,
        interpret=interpret)
    return jax.shard_map(
        fn, mesh=mesh,
        in_specs=(P(axis_name, None), P(axis_name, None),
                  P(None, None, axis_name), P(None, axis_name, None),
                  P(None, None, axis_name), P(None, None, None)),
        out_specs=P(axis_name, None), check_vma=False,
    )(hidden_states, gating_output, w1, w2, s1, s2)
