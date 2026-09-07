"""Standalone, synthetic Pallas layout and movement probes.

These are diagnostic kernels, not decoder implementations. Interpret mode
checks arithmetic only; actual TPU compilation decides physical legality.
No serving-package imports or checkpoint dependencies are required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


@dataclass(frozen=True)
class Config:
    rows: int = 16
    head_row_block: int = 0
    heads: int = 3
    width: int = 128
    experts: int = 32
    top: int = 4
    hit_block: int = 8
    feature_block: int = 128
    query_block: int = 8
    key_block: int = 16
    keys: int = 31
    queries: int = 7
    query_offset: int = 24
    steps: int = 2
    dtype: str = "float32"
    pattern: str = "random"
    model_dim: int = 128
    latent_dim: int = 128
    output_block: int = 128
    router_block: int = 32
    chunk: int = 32
    core_reserve_bytes: int = 16384
    auxiliary_reserve_bytes: int = 4096
    seed: int = 17
    pipeline_depth: int = 2

    def validate(self) -> None:
        for name, value in vars(self).items():
            if name not in ("dtype", "pattern") and (
                not isinstance(value, int) or isinstance(value, bool)
            ):
                raise ValueError(f"{name} must be an integer")
        positive = (
            self.rows,
            self.heads,
            self.width,
            self.experts,
            self.top,
            self.hit_block,
            self.feature_block,
            self.query_block,
            self.key_block,
            self.keys,
            self.queries,
            self.steps,
            self.model_dim,
            self.latent_dim,
            self.output_block,
            self.router_block,
            self.chunk,
        )
        if min(positive) <= 0 or self.top > self.experts:
            raise ValueError("positive dimensions and top <= experts required")
        if self.dtype not in ("float32", "bfloat16"):
            raise ValueError("this batch implements f32/bf16 arithmetic only")
        if self.feature_block % 128 or self.width % self.feature_block:
            raise ValueError("feature_block must divide width and span full lanes")
        if self.core_reserve_bytes < 0 or self.auxiliary_reserve_bytes < 0:
            raise ValueError("reserves must be nonnegative")
        if self.head_row_block < 0:
            raise ValueError("head_row_block must be nonnegative; zero selects native")
        if self.pattern not in ("random", "contiguous", "skew", "ties"):
            raise ValueError("unknown input pattern")
        if not 1 <= self.pipeline_depth <= 64:
            raise ValueError("pipeline_depth must be in 1..64")

    @property
    def element_type(self) -> Any:
        return getattr(jnp, self.dtype)

    @property
    def native_rows(self) -> int:
        return 16 if self.dtype == "bfloat16" else 8

    @property
    def padded_rows(self) -> int:
        return round_up(self.rows, self.native_rows)


@dataclass
class Case:
    name: str
    function: Callable[..., Any]
    inputs: tuple[Any, ...]
    expected: Any
    metadata: dict[str, Any]
    rtol: float = 2e-5
    atol: float = 2e-5


def round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def call(
    kernel: Callable[..., None],
    *,
    outputs: Any,
    interpret: bool,
    name: str,
    grid: tuple[int, ...] = (),
    inputs: Any = pl.no_block_spec,
    output_specs: Any = pl.no_block_spec,
    scratch: tuple[Any, ...] = (),
    scalar_inputs: int = 0,
    sequential: bool = False,
) -> Callable[..., Any]:
    params = pltpu.CompilerParams(
        dimension_semantics=("arbitrary",) * len(grid) if sequential else None
    )
    return pl.pallas_call(
        kernel,
        out_shape=outputs,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=scalar_inputs,
            grid=grid,
            in_specs=inputs,
            out_specs=output_specs,
            scratch_shapes=scratch,
        ),
        interpret=interpret,
        name=name,
        compiler_params=params,
    )


def data(config: Config, shape: tuple[int, ...], *, scale: float = 0.2) -> Any:
    rng = np.random.default_rng(config.seed)
    return jnp.asarray(rng.standard_normal(shape) * scale, config.element_type)


def head_view(config: Config, *, variant: str, interpret: bool) -> Case:
    """Flat, grouped-row and naive head representations share one consumer."""
    tile = config.head_row_block or config.native_rows
    rows, heads, channels = (
        round_up(config.rows, tile),
        config.heads,
        128,
    )
    x = data(config, (rows, heads * channels))

    def kernel(x_ref: Any, y_ref: Any) -> None:
        head = pl.program_id(1)
        y_ref[...] = x_ref[...] * 2 + head.astype(config.element_type)

    if variant == "flat":
        in_spec = pl.BlockSpec((tile, channels), lambda r, h: (r, h))
    elif variant == "grouped":
        in_spec = pl.BlockSpec((None, None, tile, channels), lambda r, h: (r, h, 0, 0))
    elif variant == "naive":
        in_spec = pl.BlockSpec((tile, None, channels), lambda r, h: (r, h, 0))
    else:
        raise ValueError(variant)

    def prepare(value: Any) -> Any:
        if variant == "grouped":
            return value.reshape(rows // tile, tile, heads, channels).transpose(
                0, 2, 1, 3
            )
        if variant == "naive":
            return value.reshape(rows, heads, channels)
        return value

    inner = call(
        kernel,
        outputs=jax.ShapeDtypeStruct((heads, rows, channels), config.element_type),
        interpret=interpret,
        name=f"head_{variant}",
        grid=(rows // tile, heads),
        inputs=(in_spec,),
        output_specs=pl.BlockSpec((None, tile, channels), lambda r, h: (h, r, 0)),
    )

    def function(value: Any) -> Any:
        return inner(prepare(value))

    expected = (
        np.asarray(x, dtype=np.float32)
        .reshape(rows, heads, channels)
        .transpose(1, 0, 2)
    )
    expected = np.asarray(
        jnp.asarray(expected * 2 + np.arange(heads)[:, None, None], config.element_type)
    )
    return Case(
        f"head_{variant}",
        function,
        (x,),
        expected,
        {
            "boundary": [rows, heads * channels],
            "row_tile": tile,
            "view_claim": "assigned physical layouts must be inspected",
            "consumer": "elementwise head-dependent transform, not output GEMM",
        },
        rtol=0,
        atol=0,
    )


def state_orientation(config: Config, *, variant: str, interpret: bool) -> Case:
    """One recurrence with the same nonsymmetric state in K,V or V,K order."""
    heads, size = config.heads, 128
    vectors = data(config, (heads, 4, size), scale=0.03).astype(jnp.float32)
    vectors = vectors.at[:, 3, :].set(-jnp.abs(vectors[:, 3, :]))
    source = data(config, (heads, size, size), scale=0.02).astype(jnp.float32)
    beta = jnp.linspace(0.1, 0.9, heads, dtype=jnp.float32)
    if variant not in ("kv", "vk"):
        raise ValueError(variant)
    state = source if variant == "kv" else source.swapaxes(-1, -2)

    def kernel(
        beta_ref: Any, vectors_ref: Any, state_ref: Any, new_ref: Any, out_ref: Any
    ) -> None:
        q, k, v, g = (vectors_ref[i, :] for i in range(4))
        s = state_ref[...]
        if variant == "kv":
            s = s * jnp.exp(g)[:, None]
            delta = beta_ref[pl.program_id(0)] * (v - jnp.sum(k[:, None] * s, axis=0))
            s = s + k[:, None] * delta[None, :]
            out = jnp.sum(q[:, None] * s, axis=0)
        else:
            s = s * jnp.exp(g)[None, :]
            delta = beta_ref[pl.program_id(0)] * (v - jnp.sum(k[None, :] * s, axis=1))
            s = s + delta[:, None] * k[None, :]
            out = jnp.sum(q[None, :] * s, axis=1)
        new_ref[...] = s
        out_ref[...] = out[None, :]

    state_spec = pl.BlockSpec((None, size, size), lambda h, b: (h, 0, 0))
    fn = call(
        kernel,
        outputs=(
            jax.ShapeDtypeStruct(state.shape, jnp.float32),
            jax.ShapeDtypeStruct((heads, 1, size), jnp.float32),
        ),
        interpret=interpret,
        name=f"state_{variant}",
        grid=(heads,),
        scalar_inputs=1,
        inputs=(pl.BlockSpec((None, 4, size), lambda h, b: (h, 0, 0)), state_spec),
        output_specs=(
            state_spec,
            pl.BlockSpec((None, 1, size), lambda h, b: (h, 0, 0)),
        ),
    )
    q, k, v, g = np.asarray(vectors).transpose(1, 0, 2)
    s = np.asarray(source) * np.exp(g)[:, :, None]
    delta = np.asarray(beta)[:, None] * (v - np.einsum("hk,hkv->hv", k, s))
    s = s + k[:, :, None] * delta[:, None, :]
    out = np.einsum("hk,hkv->hv", q, s)[:, None, :]
    return Case(
        f"state_{variant}",
        fn,
        (beta, vectors, state),
        (s if variant == "kv" else s.swapaxes(-1, -2), out),
        {
            "state_shape": list(state.shape),
            "state_dtype": "float32",
            "scope": "one-head state tile, no conv/norm/projection benchmark",
        },
    )


def beta_smem(config: Config, *, interpret: bool) -> Case:
    rows, heads = config.rows, config.heads
    logits = data(config, (config.steps, rows, heads)).astype(jnp.float32)

    def kernel(x_ref: Any, out_ref: Any, staging: Any, beta: Any, sem: Any) -> None:
        staging[...] = jax.nn.sigmoid(x_ref[...])
        pltpu.async_copy(staging, beta, sem).wait()

        def head_body(h: Any, _: None) -> None:
            def row_body(r: Any, acc: Any) -> Any:
                return acc + beta[r, h] * (r + 1).astype(jnp.float32)

            acc = jax.lax.fori_loop(0, rows, row_body, jnp.zeros((8, 128), jnp.float32))
            out_ref[h, :, :] = acc

        jax.lax.fori_loop(0, heads, head_body, None)

    fn = call(
        kernel,
        outputs=jax.ShapeDtypeStruct((config.steps, heads, 8, 128), jnp.float32),
        interpret=interpret,
        name="beta_smem",
        grid=(config.steps,),
        sequential=True,
        inputs=(pl.BlockSpec((None, rows, heads), lambda step: (step, 0, 0)),),
        output_specs=pl.BlockSpec((None, heads, 8, 128), lambda step: (step, 0, 0, 0)),
        scratch=(
            pltpu.VMEM((rows, heads), jnp.float32),
            pltpu.SMEM((rows, heads), jnp.float32),
            pltpu.SemaphoreType.DMA,
        ),
    )
    values = 1 / (1 + np.exp(-np.asarray(logits)))
    expected = np.sum(
        values * np.arange(1, rows + 1)[None, :, None], axis=1, dtype=np.float32
    )
    expected = np.broadcast_to(
        expected[:, :, None, None], (config.steps, heads, 8, 128)
    )
    return Case(
        "beta_smem",
        fn,
        (logits,),
        expected,
        {
            "retained_smem_bytes": 4 * rows * heads,
            "producer_vmem_logical_bytes": 4 * rows * heads,
            "grid_steps": config.steps,
            "cross_call_persistence": False,
        },
    )


def router(config: Config, *, variant: str, interpret: bool) -> Case:
    rows, experts, top = config.padded_rows, config.experts, config.top
    logits = data(config, (rows, experts), scale=0.8).astype(jnp.float32)
    bias = jnp.linspace(-0.05, 0.05, experts, dtype=jnp.float32)
    if config.pattern == "ties":
        logits, bias = jnp.zeros_like(logits), jnp.zeros_like(bias)
    if variant not in ("re", "er"):
        raise ValueError(variant)

    def kernel(logit_ref: Any, bias_ref: Any, ids_ref: Any, weight_ref: Any) -> None:
        scores = jax.nn.sigmoid(logit_ref[...])
        axis = 1 if variant == "re" else 0
        expert_ids = jnp.arange(experts, dtype=jnp.int32)
        expert_ids = expert_ids[None, :] if variant == "re" else expert_ids[:, None]
        correction = (
            bias_ref[...][None, :] if variant == "re" else bias_ref[...][:, None]
        )
        choice = scores + correction
        chosen, weights = [], []
        for _ in range(top):
            maximum = jnp.max(choice, axis=axis, keepdims=True)
            index = jnp.min(
                jnp.where(choice == maximum, expert_ids, experts), axis=axis
            )
            expanded = index[:, None] if variant == "re" else index[None, :]
            selected = expert_ids == expanded
            weights.append(jnp.sum(jnp.where(selected, scores, 0), axis=axis))
            chosen.append(index)
            choice = jnp.where(selected, -jnp.inf, choice)
        weights_array = jnp.stack(weights, axis=0)
        weights_array /= jnp.sum(weights_array, axis=0, keepdims=True) + 1e-20
        valid = jnp.arange(rows)[None, :] < config.rows
        ids_ref[...] = jnp.where(valid, jnp.stack(chosen, axis=0), -1)
        weight_ref[...] = jnp.where(valid, weights_array, 0)

    inner = call(
        kernel,
        outputs=(
            jax.ShapeDtypeStruct((top, rows), jnp.int32),
            jax.ShapeDtypeStruct((top, rows), jnp.float32),
        ),
        interpret=interpret,
        name=f"router_{variant}",
    )

    def function(x: Any, correction: Any) -> Any:
        return inner(x if variant == "re" else x.T, correction)

    scores = 1 / (1 + np.exp(-np.asarray(logits)))
    ids = np.argsort(-(scores + np.asarray(bias)), axis=1, kind="stable")[:, :top]
    weights = np.take_along_axis(scores, ids, axis=1)
    weights /= np.sum(weights, axis=1, keepdims=True) + 1e-20
    ids[config.rows :] = -1
    weights[config.rows :] = 0
    return Case(
        f"router_{variant}",
        function,
        (logits, bias),
        (ids.T, weights.T),
        {
            "top": top,
            "tie_policy": "lowest expert id; model tie compatibility remains a gate",
            "input_scope": "precomputed logits; router GEMM and SMEM list building excluded",
            "orientation": variant,
            "pattern": config.pattern,
        },
    )


def gather_combine(config: Config, *, variant: str, interpret: bool) -> Case:
    """Bounded hit batches; no overflow dropping or overlapping program writes."""
    if variant not in ("aligned", "dma", "row"):
        raise ValueError(variant)
    rows, width, tile = config.padded_rows, config.width, config.native_rows
    block = config.feature_block
    hit_capacity = round_up(config.hit_block, tile)
    hits = config.rows * config.top
    padded_hits = round_up(hits, config.hit_block)
    rng = np.random.default_rng(config.seed)
    tokens = np.tile(np.arange(config.rows, dtype=np.int32), config.top)
    if config.pattern == "random":
        rng.shuffle(tokens)
    elif config.pattern == "skew":
        tokens.sort()  # repeated token visits stress serialized combine
    weights_np = rng.uniform(0.01, 0.2, hits).astype(np.float32)
    ids = jnp.asarray(np.pad(tokens, (0, padded_hits - hits)), jnp.int32)
    weights = jnp.asarray(np.pad(weights_np, (0, padded_hits - hits)))
    x = data(config, (rows, width))

    def kernel(
        ids_ref: Any,
        weights_ref: Any,
        x_ref: Any,
        y_ref: Any,
        gathered: Any,
        row_stage: Any,
        output_stage: Any,
        native: Any,
        sem: Any,
    ) -> None:
        y_ref[...] = jnp.zeros((rows, block), jnp.float32)

        def batch_body(batch: Any, _: None) -> None:
            gathered[...] = jnp.zeros(gathered.shape, config.element_type)

            def fetch(mi: Any, _: None) -> None:
                hit = batch * config.hit_block + mi

                @pl.when(hit < hits)
                def valid() -> None:
                    row = ids_ref[hit]
                    if variant == "dma":
                        pltpu.async_copy(
                            x_ref.at[pl.ds(row, 1), :],
                            gathered.at[pl.ds(mi, 1), :],
                            sem,
                        ).wait()
                    elif variant == "row":
                        gathered[mi, :, :] = x_ref[row, :, :]
                    else:
                        base = row // tile * tile
                        parent = x_ref[pl.ds(base, tile), :]
                        selected = jnp.sum(
                            jnp.where(
                                jnp.arange(tile)[:, None] == row % tile,
                                parent.astype(jnp.float32),
                                0,
                            ),
                            axis=0,
                        ).astype(config.element_type)
                        dest = mi // tile * tile
                        prior = gathered[pl.ds(dest, tile), :]
                        gathered[pl.ds(dest, tile), :] = jnp.where(
                            jnp.arange(tile)[:, None] == mi % tile,
                            selected[None, :],
                            prior,
                        )

            with jax.named_scope("hit_gather"):
                jax.lax.fori_loop(0, config.hit_block, fetch, None)
            # Explicit matrix-native handoff, including row-layout conversion.
            with jax.named_scope("native_handoff_and_rounding"):
                native[...] = gathered[...].reshape(hit_capacity, block)
                native[...] = (native[...].astype(jnp.float32) * 1.125).astype(
                    config.element_type
                )

            def combine(mi: Any, _: None) -> None:
                hit = batch * config.hit_block + mi

                @pl.when(hit < hits)
                def valid() -> None:
                    row, weight = ids_ref[hit], weights_ref[hit]
                    if variant == "dma":
                        pltpu.async_copy(
                            native.at[pl.ds(mi, 1), :], row_stage, sem
                        ).wait()
                        pltpu.async_copy(
                            y_ref.at[pl.ds(row, 1), :], output_stage, sem
                        ).wait()
                        output_stage[...] += row_stage[...].astype(jnp.float32) * weight
                        pltpu.async_copy(
                            output_stage, y_ref.at[pl.ds(row, 1), :], sem
                        ).wait()
                    else:
                        src_base = mi // tile * tile
                        parent = native[pl.ds(src_base, tile), :].astype(jnp.float32)
                        value = jnp.sum(
                            jnp.where(
                                jnp.arange(tile)[:, None] == mi % tile, parent, 0
                            ),
                            axis=0,
                        )
                        dest = row // 8 * 8
                        prior = y_ref[pl.ds(dest, 8), :]
                        y_ref[pl.ds(dest, 8), :] = prior + jnp.where(
                            jnp.arange(8)[:, None] == row % 8,
                            value[None, :] * weight,
                            0,
                        )

            with jax.named_scope("hit_combine"):
                jax.lax.fori_loop(0, config.hit_block, combine, None)

        jax.lax.fori_loop(0, padded_hits // config.hit_block, batch_body, None)

    input_shape = (rows, 1, block) if variant == "row" else (rows, block)
    index_map = (
        (lambda col, ids, w: (0, 0, col))
        if variant == "row"
        else (lambda col, ids, w: (0, col))
    )
    gather_shape = (
        (hit_capacity, 1, block) if variant == "row" else (hit_capacity, block)
    )
    inner = call(
        kernel,
        outputs=jax.ShapeDtypeStruct((rows, width), jnp.float32),
        interpret=interpret,
        name=f"gather_{variant}",
        grid=(width // block,),
        scalar_inputs=2,
        inputs=(pl.BlockSpec(input_shape, index_map),),
        output_specs=pl.BlockSpec((rows, block), lambda col, ids, w: (0, col)),
        scratch=(
            pltpu.VMEM(gather_shape, config.element_type),
            pltpu.VMEM((1, block), config.element_type),
            pltpu.VMEM((1, block), jnp.float32),
            pltpu.VMEM((hit_capacity, block), config.element_type),
            pltpu.SemaphoreType.DMA,
        ),
    )

    def function(token_ids: Any, mix: Any, value: Any) -> Any:
        return inner(
            token_ids, mix, value.reshape(rows, 1, width) if variant == "row" else value
        )

    transformed = np.asarray(
        (x.astype(jnp.float32) * 1.125).astype(config.element_type), dtype=np.float32
    )
    expected = np.zeros((rows, width), np.float32)
    for token, weight in zip(tokens, weights_np, strict=True):
        expected[token] += transformed[token] * weight
    return Case(
        f"gather_{variant}",
        function,
        (ids, weights, x),
        expected,
        {
            "hits": hits,
            "logical_hit_block": config.hit_block,
            "allocated_hit_rows": hit_capacity,
            "variant": variant,
            "scope": "prebuilt hit list, rounded synthetic expert transform, no expert GEMM",
            "cross_expert_repeated_tokens": True,
            "no_dropped_hits": True,
        },
    )


def phase_bytes(config: Config) -> tuple[int, list[int]]:
    r, d, z = config.padded_rows, config.model_dim, config.latent_dim
    persistent = round_up(4 * r * d, 512)
    auxiliary = config.auxiliary_reserve_bytes
    # Reserve an explicit core workspace; C-dependent minimum prevents a free large C.
    core = max(
        config.core_reserve_bytes,
        4 * (2 * 128**2 + 8 * config.chunk * 128 + 2 * config.chunk**2),
    )
    totals = [
        8 * r * d + auxiliary,
        (4 * d + 21760 + 8 * config.output_block) * r + core + auxiliary,
        (4 * d + 10 * z + 8 * config.output_block + 64 + 4 * config.router_block) * r
        - 4 * max(0, r - config.hit_block) * z
        + 4 * auxiliary,
    ]
    return persistent, [round_up(max(512, total - persistent), 512) for total in totals]


def phase_residency(config: Config, *, variant: str, interpret: bool) -> Case:
    if variant not in ("scoped", "colive"):
        raise ValueError(variant)
    persistent_bytes, working_bytes = phase_bytes(config)
    block_rows = 8
    # Full native f32 chunks: round reservations up, never under-allocate.
    shapes = [(round_up(size // 512, block_rows), 128) for size in working_bytes]
    persistent_shape = (round_up(persistent_bytes // 512, block_rows), 128)
    seed = data(config, (block_rows, 128), scale=0.1).astype(jnp.float32)

    def kernel(seed_ref: Any, out_ref: Any, persistent: Any) -> None:
        seed_value = seed_ref[...]

        def fill(ref: Any, value: Any) -> None:
            def body(i: Any, _: None) -> None:
                ref[pl.ds(i * block_rows, block_rows), :] = (
                    value + (i % 7).astype(jnp.float32) * 0.001
                )

            jax.lax.fori_loop(0, ref.shape[0] // block_rows, body, None)

        def consume(ref: Any) -> Any:
            def body(i: Any, acc: Any) -> Any:
                index = ref.shape[0] // block_rows - 1 - i
                return acc + ref[pl.ds(index * block_rows, block_rows), :]

            return jax.lax.fori_loop(
                0, ref.shape[0] // block_rows, body, jnp.zeros_like(seed_value)
            )

        fill(persistent, seed_value)
        if variant == "scoped":
            acc = seed_value
            for phase_index, shape in enumerate(shapes):

                def phase(ref: Any) -> Any:
                    fill(ref, acc)
                    return consume(ref) / (shape[0] // block_rows)

                with jax.named_scope(f"allocation_phase_{phase_index}"):
                    acc = pl.run_scoped(phase, pltpu.VMEM(shape, jnp.float32))
        else:

            def phases(*refs: Any) -> Any:
                # Store all buffers before consuming; then apply the same chained
                # mean offsets as scoped. No premature last-use for these refs.
                for ref in refs:
                    fill(ref, seed_value)
                acc = seed_value
                for ref in refs:
                    acc = acc + (
                        consume(ref) / (ref.shape[0] // block_rows) - seed_value
                    )
                return acc

            acc = pl.run_scoped(
                phases, *(pltpu.VMEM(shape, jnp.float32) for shape in shapes)
            )
        out_ref[...] = acc + consume(persistent) / (persistent.shape[0] // block_rows)

    fn = call(
        kernel,
        outputs=jax.ShapeDtypeStruct(seed.shape, jnp.float32),
        interpret=interpret,
        name=f"phase_{variant}",
        scratch=(pltpu.VMEM(persistent_shape, jnp.float32),),
    )
    expected = np.asarray(seed) * 2
    for shape in [*shapes, persistent_shape]:
        count = shape[0] // block_rows
        expected += np.mean(np.arange(count) % 7) * 0.001
    logical = persistent_shape[0] * 512 + (max if variant == "scoped" else sum)(
        shape[0] * 512 for shape in shapes
    )
    return Case(
        f"phase_{variant}",
        fn,
        (seed,),
        expected,
        {
            "named_scratch_bytes": logical,
            "persistent_shape": persistent_shape,
            "phase_shapes": shapes,
            "variant": variant,
            "warning": "allocation skeleton, not full arithmetic or a layer latency proxy; inspect DCE and actual allocation",
        },
        rtol=2e-4,
        atol=2e-4,
    )


def mla_layout(config: Config, *, variant: str, interpret: bool) -> Case:
    if variant not in ("expanded", "split", "expanded_pipeline", "split_pipeline"):
        raise ValueError(variant)
    pipelined = variant.endswith("_pipeline")
    expanded = variant.startswith("expanded")
    depth = config.pipeline_depth if pipelined else 1
    qb, kb, size, value_size = config.query_block, config.key_block, 192, 128
    qrows, krows = round_up(config.queries, qb), round_up(config.keys, kb)
    q = data(config, (qrows, size))
    k = data(config, (krows, size), scale=0.17)
    v = data(config, (krows, value_size), scale=0.13)

    def kernel(q_ref: Any, *refs: Any) -> None:
        if expanded:
            k_ref, v_ref, out_ref, kbuf, vbuf, sem = refs
        else:
            kp_ref, kr_ref, v_ref, out_ref, kpbuf, krbuf, vbuf, sem = refs
        query = q_ref[...]
        query_positions = pl.program_id(0) * qb + jnp.arange(qb) + config.query_offset

        def descriptors(index: Any) -> tuple[Any, ...]:
            slot = index % depth
            if expanded:
                refs = ((k_ref, kbuf), (v_ref, vbuf))
            else:
                refs = ((kp_ref, kpbuf), (kr_ref, krbuf), (v_ref, vbuf))
            return tuple(
                pltpu.make_async_copy(
                    src.at[pl.ds(index * kb, kb), :], dst.at[slot], sem.at[slot, i]
                )
                for i, (src, dst) in enumerate(refs)
            )

        def start(index: Any) -> None:
            for descriptor in descriptors(index):
                descriptor.start()

        if pipelined:
            for p in range(depth - 1):
                if p < krows // kb:
                    start(p)

        def body(index: Any, carry: tuple[Any, Any, Any]) -> tuple[Any, Any, Any]:
            maximum, normalizer, acc = carry
            if pipelined:
                future = index + depth - 1

                @pl.when(future < krows // kb)
                def prefetch() -> None:
                    start(future)

                copies = descriptors(index)
                for descriptor in copies[:-1]:
                    descriptor.wait()
                slot = index % depth
                if expanded:
                    scores = jnp.dot(
                        query, kbuf[slot, ...].T, preferred_element_type=jnp.float32
                    )
                else:
                    scores = jnp.dot(
                        query[:, :128],
                        kpbuf[slot, ...].T,
                        preferred_element_type=jnp.float32,
                    )
                    scores += jnp.dot(
                        query[:, 128:],
                        krbuf[slot, ...].T,
                        preferred_element_type=jnp.float32,
                    )
            elif expanded:
                pltpu.async_copy(k_ref.at[pl.ds(index * kb, kb), :], kbuf, sem).wait()
                scores = jnp.dot(query, kbuf[...].T, preferred_element_type=jnp.float32)
            else:
                pltpu.async_copy(kp_ref.at[pl.ds(index * kb, kb), :], kpbuf, sem).wait()
                pltpu.async_copy(kr_ref.at[pl.ds(index * kb, kb), :], krbuf, sem).wait()
                scores = jnp.dot(
                    query[:, :128], kpbuf[...].T, preferred_element_type=jnp.float32
                )
                scores += jnp.dot(
                    query[:, 128:], krbuf[...].T, preferred_element_type=jnp.float32
                )
            if not pipelined:
                pltpu.async_copy(v_ref.at[pl.ds(index * kb, kb), :], vbuf, sem).wait()
            key_positions = index * kb + jnp.arange(kb)
            valid = (key_positions[None, :] <= query_positions[:, None]) & (
                key_positions[None, :] < config.keys
            )
            scores = jnp.where(valid, scores / np.sqrt(size), -jnp.inf)
            updated_max = jnp.maximum(maximum, jnp.max(scores, axis=1))
            safe_max = jnp.where(jnp.isfinite(updated_max), updated_max, 0)
            alpha = jnp.where(normalizer > 0, jnp.exp(maximum - safe_max), 0)
            p = jnp.where(valid, jnp.exp(scores - safe_max[:, None]), 0)
            if pipelined:
                copies[-1].wait()  # V readiness is needed by PV, not QK/softmax.
                values = vbuf[index % depth, ...]
            else:
                values = vbuf[...]
            # Explicit operand cast: the independent oracle uses this same
            # blocked rounding schedule. This is not a bf16 model golden.
            acc = acc * alpha[:, None] + jnp.dot(
                p.astype(config.element_type),
                values,
                preferred_element_type=jnp.float32,
            )
            normalizer = normalizer * alpha + jnp.sum(p, axis=1)
            return updated_max, normalizer, acc

        _, denom, acc = jax.lax.fori_loop(
            0,
            krows // kb,
            body,
            (
                jnp.full((qb,), -jnp.inf, jnp.float32),
                jnp.zeros((qb,), jnp.float32),
                jnp.zeros((qb, value_size), jnp.float32),
            ),
        )
        valid_query = pl.program_id(0) * qb + jnp.arange(qb) < config.queries
        out_ref[...] = jnp.where(
            valid_query[:, None], acc / jnp.maximum(denom[:, None], 1e-20), 0
        )

    qspec = pl.BlockSpec((qb, size), lambda i: (i, 0))
    any_spec = pl.BlockSpec(memory_space=pl.ANY)
    prefix = (depth,) if pipelined else ()
    if expanded:
        in_specs = (qspec, any_spec, any_spec)
        scratch = (pltpu.VMEM((*prefix, kb, size), config.element_type),)
        inputs = (q, k, v)
    else:
        in_specs = (qspec, any_spec, any_spec, any_spec)
        scratch = (
            pltpu.VMEM((*prefix, kb, 128), config.element_type),
            pltpu.VMEM((*prefix, kb, 64), config.element_type),
        )
        # Preformat synthetic cache outside the timed invocation. A real loader
        # would need its own source-compatible cache producer/append path.
        inputs = (q, jnp.array(k[:, :128]), jnp.array(k[:, 128:]), v)
    fn = call(
        kernel,
        outputs=jax.ShapeDtypeStruct((qrows, value_size), jnp.float32),
        interpret=interpret,
        name=f"mla_{variant}",
        grid=(qrows // qb,),
        inputs=in_specs,
        output_specs=pl.BlockSpec((qb, value_size), lambda i: (i, 0)),
        scratch=(
            *scratch,
            pltpu.VMEM((*prefix, kb, value_size), config.element_type),
            pltpu.SemaphoreType.DMA((depth, 2 if expanded else 3))
            if pipelined
            else pltpu.SemaphoreType.DMA,
        ),
    )
    qn, kn, vn = (np.asarray(x, dtype=np.float32) for x in (q, k, v))
    maximum = np.full(qrows, -np.inf, np.float32)
    denom = np.zeros(qrows, np.float32)
    acc = np.zeros((qrows, value_size), np.float32)
    for start in range(0, krows, kb):
        scores = qn @ kn[start : start + kb].T / np.sqrt(size)
        valid = (
            np.arange(start, start + kb)[None, :]
            <= (np.arange(qrows) + config.query_offset)[:, None]
        ) & (np.arange(start, start + kb)[None, :] < config.keys)
        scores = np.where(valid, scores, -np.inf)
        updated = np.maximum(maximum, scores.max(axis=1))
        safe = np.where(np.isfinite(updated), updated, 0)
        alpha = np.where(denom > 0, np.exp(maximum - safe), 0)
        p = np.where(valid, np.exp(scores - safe[:, None]), 0).astype(np.float32)
        operand = np.asarray(jnp.asarray(p, config.element_type), dtype=np.float32)
        acc = acc * alpha[:, None] + operand @ vn[start : start + kb]
        denom = denom * alpha + p.sum(axis=1)
        maximum = updated
    expected = acc / np.maximum(denom[:, None], 1e-20)
    expected[config.queries :] = 0
    return Case(
        f"mla_{variant}",
        fn,
        inputs,
        expected,
        {
            "cache_variant": variant,
            "query_block": qb,
            "key_block": kb,
            "keys": config.keys,
            "query_offset": config.query_offset,
            "scope": "single head, expanded values; no absorption, cache append or projections",
            "numerics": "explicit blocked probability operand rounding; not full-model fidelity",
        },
        rtol=2e-2 if config.dtype == "bfloat16" else 1e-4,
        atol=2e-3 if config.dtype == "bfloat16" else 1e-5,
    )


NAMES = (
    "head_flat",
    "head_grouped",
    "head_naive",
    "state_kv",
    "state_vk",
    "beta_smem",
    "router_re",
    "router_er",
    "gather_aligned",
    "gather_dma",
    "gather_row",
    "phase_scoped",
    "phase_colive",
    "mla_expanded",
    "mla_split",
)


def make_case(name: str, *, config: Config, interpret: bool) -> Case:
    config.validate()
    family, variant = name.split("_", 1)
    builders = {
        "head": head_view,
        "state": state_orientation,
        "router": router,
        "gather": gather_combine,
        "phase": phase_residency,
        "mla": mla_layout,
    }
    if name == "beta_smem":
        return beta_smem(config, interpret=interpret)
    if name not in NAMES:
        raise ValueError(f"unknown case: {name}")
    return builders[family](config, variant=variant, interpret=interpret)
