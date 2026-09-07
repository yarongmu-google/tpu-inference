"""Synthetic pipeline experiments; no serving or checkpoint dependencies.

Source ordering exposes opportunities, not a guaranteed instruction schedule.
The paired arms require TPU dumps and device traces before overlap is claimed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from layer_probe_kernels import Case, call, round_up


@dataclass(frozen=True)
class PipelineConfig:
    jobs: int = 5
    block_rows: int = 16
    columns: int = 128
    depth: int = 3
    unroll: int = 2
    rows: int = 32
    hits: int = 4
    dtype: str = "float32"
    direction: str = "mixed"
    pattern: str = "random"
    seed: int = 19

    def validate(self) -> None:
        for name, value in vars(self).items():
            if name not in ("dtype", "direction", "pattern"):
                if not isinstance(value, int) or isinstance(value, bool):
                    raise ValueError(f"{name} must be an integer")
        if not 1 <= self.jobs <= 8192 or not 1 <= self.depth <= 64:
            raise ValueError("jobs/depth outside experiment bounds")
        if not 1 <= self.unroll <= 8 or not 1 <= self.hits <= 16:
            raise ValueError("unroll/hits outside experiment bounds")
        if min(self.rows, self.block_rows) < 1 or self.columns % 128:
            raise ValueError("positive rows and full-lane columns required")
        if not 128 <= self.columns <= 1024 or self.block_rows > 2048:
            raise ValueError("panel outside experiment bounds")
        if self.dtype not in ("float32", "bfloat16"):
            raise ValueError("unsupported experiment dtype")
        if self.direction not in ("read", "write", "mixed", "local"):
            raise ValueError("unknown transfer direction")
        if self.pattern not in ("random", "contiguous", "skew"):
            raise ValueError("unknown hit pattern")

    @property
    def element_type(self) -> Any:
        return getattr(jnp, self.dtype)

    @property
    def native_rows(self) -> int:
        return 16 if self.dtype == "bfloat16" else 8


def sample(c: PipelineConfig, shape: tuple[int, ...]) -> Any:
    value = np.random.default_rng(c.seed).normal(0, 0.1, shape).astype(np.float32)
    return jnp.asarray(value, c.element_type)


def ring_loop(*, jobs: int, depth: int, unroll: int, start: Any, consume: Any) -> None:
    """Read-only input ring: prime D-1, start future before waiting current."""
    for p in range(depth - 1):

        @pl.when(p < jobs)
        def prime() -> None:
            start(p)

    def group_body(group: Any, carry: Any) -> Any:
        for u in range(unroll):
            j = group * unroll + u

            @pl.when(j < jobs)
            def step() -> None:
                future = j + depth - 1

                @pl.when(future < jobs)
                def prefetch() -> None:
                    start(future)

                consume(j)

        return carry

    jax.lax.fori_loop(0, (jobs + unroll - 1) // unroll, group_body, None)


def stream(c: PipelineConfig, *, variant: str, interpret: bool) -> Case:
    n, r, w = c.jobs, c.block_rows, c.columns
    depth = 1 if variant == "serial" else c.depth
    dtype = c.element_type
    any_spec = pl.BlockSpec(memory_space=pl.ANY)
    source_hbm = c.direction in ("read", "mixed")
    output_hbm = c.direction in ("write", "mixed")
    x = sample(c, (n * r if source_hbm else r, w))
    sampled_rows = min(r, c.native_rows)
    output_shape = (n * r, w) if output_hbm else (n, sampled_rows, w)
    output_dtype = dtype if output_hbm else jnp.float32

    def kernel(
        x_ref: Any, y_ref: Any, slots: Any, sems: Any, out_slots: Any, out_sems: Any
    ) -> None:
        def descriptor(j: Any) -> Any:
            source = x_ref.at[pl.ds(j * r, r), :] if source_hbm else x_ref
            if c.direction == "write":
                return pltpu.make_async_copy(
                    source, y_ref.at[pl.ds(j * r, r), :], sems.at[j % depth]
                )
            return pltpu.make_async_copy(
                source, slots.at[j % depth], sems.at[j % depth]
            )

        def start(j: Any) -> None:
            descriptor(j).start()

        def output_descriptor(j: Any) -> Any:
            return pltpu.make_async_copy(
                out_slots.at[j % 2], y_ref.at[pl.ds(j * r, r), :], out_sems.at[j % 2]
            )

        def consume(j: Any) -> None:
            with jax.named_scope("input_ready"):
                descriptor(j).wait()
            if c.direction == "write":
                return
            if c.direction == "mixed":
                if variant != "serial":

                    @pl.when(j >= 2)
                    def release_output() -> None:
                        output_descriptor(j - 2).wait()

                with jax.named_scope("native_consumer"):
                    out_slots[j % 2, ...] = slots[j % depth, ...]
                output_descriptor(j).start()
                if variant == "serial":
                    output_descriptor(j).wait()
            else:
                # Sample the first native plane. The explicit descriptor still
                # requests the entire panel; verify retained DMA sizes in dumps.
                with jax.named_scope("sample_consumer"):
                    y_ref[j, ...] = (
                        slots[j % depth, :sampled_rows, :].astype(jnp.float32) + j
                    )

        ring_loop(jobs=n, depth=depth, unroll=c.unroll, start=start, consume=consume)
        if c.direction == "mixed" and variant != "serial":
            for last in range(max(0, n - 2), n):
                output_descriptor(last).wait()

    if variant == "emitted":
        if c.direction != "mixed":
            raise ValueError("emitted comparator uses mixed copy traffic")

        def emitted(x_ref: Any, y_ref: Any) -> None:
            def body(a: Any, b: Any) -> None:
                b[...] = a[...]

            pltpu.emit_pipeline(
                body,
                grid=(n,),
                in_specs=(
                    pl.BlockSpec(
                        (r, w),
                        lambda j: (j, 0),
                        pipeline_mode=pl.Buffered(buffer_count=depth),
                    ),
                ),
                out_specs=pl.BlockSpec(
                    (r, w), lambda j: (j, 0), pipeline_mode=pl.Buffered(buffer_count=2)
                ),
            )(x_ref, y_ref)

        fn = call(
            emitted,
            outputs=jax.ShapeDtypeStruct(output_shape, output_dtype),
            inputs=(any_spec,),
            output_specs=any_spec,
            interpret=interpret,
            name="stream_emitted",
        )
    else:
        fn = call(
            kernel,
            outputs=jax.ShapeDtypeStruct(output_shape, output_dtype),
            inputs=(any_spec if source_hbm else pl.no_block_spec,),
            output_specs=any_spec if output_hbm else pl.no_block_spec,
            scratch=(
                pltpu.VMEM((depth, r, w), dtype),
                pltpu.SemaphoreType.DMA((depth,)),
                pltpu.VMEM((2, r, w), dtype),
                pltpu.SemaphoreType.DMA((2,)),
            ),
            interpret=interpret,
            name=f"stream_{variant}_{c.direction}",
        )
    xn = np.asarray(x)
    if output_hbm:
        expected = xn if source_hbm else np.tile(xn, (n, 1))
    else:
        expected = np.stack(
            [
                (
                    xn[j * r : j * r + sampled_rows]
                    if source_hbm
                    else xn[:sampled_rows]
                ).astype(np.float32)
                + j
                for j in range(n)
            ]
        )
    return Case(
        f"stream_{variant}",
        fn,
        (x,),
        expected,
        {
            "direction": c.direction,
            "requested_bytes_per_dma": r * w * (2 if c.dtype == "bfloat16" else 4),
            "input_depth": depth,
            "output_depth": 2 if c.direction == "mixed" else 0,
            "chart": "prime D-1; launch j+D-1; wait j; native consumer; deferred output reuse/drain",
            "timing_scope": "whole-call stream plus consumer and boundary traffic; not isolated DMA latency",
            "read_local_consumer": "first native plane sampled; verify full descriptor extent survives",
        },
        rtol=0,
        atol=0,
    )


def dma_add(c: PipelineConfig, *, interpret: bool) -> Case:
    x = sample(c, (c.native_rows, c.columns))

    def kernel(a: Any, y: Any, tmp: Any, sem: Any) -> None:
        tmp[...] = a[...]
        pltpu.async_copy(a, tmp, sem, add=True).wait()
        y[...] = tmp[...]

    fn = call(
        kernel,
        outputs=jax.ShapeDtypeStruct(x.shape, x.dtype),
        interpret=interpret,
        name="dma_add_capability",
        scratch=(pltpu.VMEM(x.shape, x.dtype), pltpu.SemaphoreType.DMA),
    )
    return Case(
        "dma_add",
        fn,
        (x,),
        np.asarray(x * 2),
        {
            "capability": "DMA accumulate; rejection is not a performance pass",
            "chart": "initialize nonzero VMEM; DMA add; wait; return destination",
        },
        rtol=0,
        atol=0,
    )


def broadcast(c: PipelineConfig, *, variant: str, interpret: bool) -> Case:
    x = sample(c, (8, 128)).astype(jnp.float32)
    q = sample(c, (1, 128)).astype(jnp.float32)
    is_smem = variant.startswith("smem")

    def body(q_ref: Any, x_ref: Any, y_ref: Any, stage: Any, sem: Any) -> None:
        if variant == "smem_vector":
            values = q_ref[...]
        elif variant == "smem_stage":
            pltpu.async_copy(q_ref, stage, sem).wait()
            values = stage[...]
        elif variant == "smem_scalar":
            values = jnp.zeros((1, 128), jnp.float32)
            for lane in range(128):
                values += jnp.where(jnp.arange(128)[None, :] == lane, q_ref[0, lane], 0)
        else:
            values = q_ref[...]
        if variant == "v_rows":
            expanded = jnp.broadcast_to(values[0, :8, None], (8, 128))
        else:
            expanded = jnp.broadcast_to(values, (8, 128))
        y_ref[...] = x_ref[...] * expanded

    fn = call(
        body,
        outputs=jax.ShapeDtypeStruct(x.shape, jnp.float32),
        scalar_inputs=int(is_smem),
        inputs=(pl.no_block_spec,) if is_smem else pl.no_block_spec,
        scratch=(pltpu.VMEM((1, 128), jnp.float32), pltpu.SemaphoreType.DMA),
        interpret=interpret,
        name=f"broadcast_{variant}",
    )
    qn = np.asarray(q)
    expected = np.asarray(x) * (qn[0, :8, None] if variant == "v_rows" else qn)
    return Case(
        f"broadcast_{variant}",
        fn,
        (q, x),
        expected,
        {
            "chart": "SMEM scalar/staged or VMEM vector load; explicit [1,128] or [8,1] broadcast; native multiply",
            "scope": "f32 recurrence operand preparation; allocation fit does not establish vector access",
        },
        rtol=0,
        atol=0,
    )


def compute(c: PipelineConfig, *, variant: str, interpret: bool) -> Case:
    u = c.unroll if variant == "interleaved" else 1
    n = round_up(c.jobs, u)
    r = round_up(c.block_rows, 8)
    x = sample(c, (n, r, 128))
    w = sample(c, (128, 128))

    def kernel(a: Any, b: Any, out: Any) -> None:
        def body(group: Any, carry: Any) -> Any:
            mm, exp = {}, {}
            # At each tick emit new MXU work, older EUP work, and still older
            # reduction/normalization work. Arrays are SSA values, not futures.
            for tick in range(u + 2):
                if tick < u:
                    with jax.named_scope("matrix_producer"):
                        mm[tick] = jnp.dot(
                            a[group * u + tick, ...],
                            b[...],
                            preferred_element_type=jnp.float32,
                        )
                if 0 <= tick - 1 < u:
                    with jax.named_scope("unary_consumer"):
                        exp[tick - 1] = jnp.exp(mm[tick - 1] * 0.01)
                if 0 <= tick - 2 < u:
                    with jax.named_scope("reduce_normalize"):
                        e = exp[tick - 2]
                        out[group * u + tick - 2, ...] = e / jnp.sum(
                            e, axis=1, keepdims=True
                        )
            return carry

        jax.lax.fori_loop(0, n // u, body, None)

    fn = call(
        kernel,
        outputs=jax.ShapeDtypeStruct(x.shape, jnp.float32),
        interpret=interpret,
        name=f"compute_{variant}",
    )
    mm = np.asarray(x, np.float32) @ np.asarray(w, np.float32)
    e = np.exp(mm * np.float32(0.01))
    expected = e / e.sum(axis=-1, keepdims=True)
    return Case(
        f"compute_{variant}",
        fn,
        (x, w),
        expected,
        {
            "valid_jobs": c.jobs,
            "executed_jobs": n,
            "unroll": u,
            "chart": "MXU j -> VALU/EUP j -> XLU/VALU j; stagger distinct jobs within a bounded window",
            "scope": "synthetic compute-to-compute chain, not a decoder; spills/overlap require backend evidence",
        },
        rtol=3e-3,
        atol=2e-5,
    )


def state_writeback(c: PipelineConfig, *, overlap: bool, interpret: bool) -> Case:
    n = c.jobs
    x = sample(c, (n, 128, 128)).astype(jnp.float32)
    vectors = sample(c, (n, 4, 128)).astype(jnp.float32)
    beta = jnp.linspace(0.1, 0.9, n, dtype=jnp.float32)
    any_spec = pl.BlockSpec(memory_space=pl.ANY)

    def kernel(
        beta_ref: Any,
        source: Any,
        vec: Any,
        dest: Any,
        out: Any,
        state: Any,
        read_sem: Any,
        write_sem: Any,
    ) -> None:
        def body(j: Any, carry: Any) -> Any:
            pltpu.async_copy(source.at[j], state, read_sem).wait()
            q, k, v, g = (vec[j, i, :] for i in range(4))
            s = state[...] * jnp.exp(-jnp.abs(g))[None, :]
            delta = beta_ref[j] * (v - jnp.sum(s * k[None, :], axis=1))
            state[...] = s + delta[:, None] * k[None, :]
            wb = pltpu.async_copy(state, dest.at[j], write_sem)
            if not overlap:
                wb.wait()
            with jax.named_scope("state_read_only_consumer"):
                out[j, 0, :] = jnp.sum(state[...] * q[None, :], axis=1)
            if overlap:
                wb.wait()  # Before the next iteration overwrites the source.
            return carry

        jax.lax.fori_loop(0, n, body, None)

    fn = call(
        kernel,
        outputs=(
            jax.ShapeDtypeStruct(x.shape, jnp.float32),
            jax.ShapeDtypeStruct((n, 1, 128), jnp.float32),
        ),
        scalar_inputs=1,
        inputs=(any_spec, pl.no_block_spec),
        output_specs=(any_spec, pl.no_block_spec),
        scratch=(
            pltpu.VMEM((128, 128), jnp.float32),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
        ),
        interpret=interpret,
        name=f"state_writeback_{overlap}",
    )
    vn = np.asarray(vectors)
    q, k, v, g = (vn[:, i] for i in range(4))
    s = np.asarray(x) * np.exp(-np.abs(g))[:, None, :]
    delta = np.asarray(beta)[:, None] * (v - (s * k[:, None, :]).sum(axis=2))
    s = s + delta[:, :, None] * k[:, None, :]
    o = (s * q[:, None, :]).sum(axis=2)[:, None, :]
    return Case(
        "state_overlap" if overlap else "state_serial",
        fn,
        (beta, x, vectors),
        (s, o),
        {
            "chart": "read state; V-first recurrence; vst state; start writeback; read-only output reduction; drain before reuse",
            "scope": "writeback/read-only overlap only; input DMA is serialized to isolate the comparison",
        },
    )


def gather(c: PipelineConfig, *, variant: str, interpret: bool) -> Case:
    n, m, w = c.jobs, c.hits, c.columns
    depth = c.depth if variant.endswith("pipeline") else 1
    parent_arm = variant.startswith("parent")
    tau = c.native_rows
    rows = round_up(c.rows, max(8, tau))
    x = sample(c, (rows, w))
    rng = np.random.default_rng(c.seed)
    ids = rng.integers(0, c.rows, n * m, dtype=np.int32)
    if c.pattern == "skew":
        ids[:] = 0
    elif c.pattern == "contiguous":
        ids = np.arange(n * m, dtype=np.int32) % c.rows
    weights = rng.uniform(0.01, 0.2, n * m).astype(np.float32)
    shape = (depth, m, tau, w) if parent_arm else (depth, m, w)

    def kernel(
        ids_ref: Any,
        weights_ref: Any,
        src: Any,
        out: Any,
        buffers: Any,
        sems: Any,
        native: Any,
    ) -> None:
        out[...] = jnp.zeros((rows, w), jnp.float32)

        def desc(j: Any, h: int) -> Any:
            token = ids_ref[j * m + h]
            if parent_arm:
                source = src.at[pl.ds(token // tau * tau, tau), :]
                dest = buffers.at[j % depth, h]
            else:
                source = src.at[pl.ds(token, 1), :]
                dest = buffers.at[j % depth, pl.ds(h, 1), :]
            return pltpu.make_async_copy(source, dest, sems.at[j % depth, h])

        def start(j: Any) -> None:
            for h in range(m):
                desc(j, h).start()
                if variant.endswith("serial"):
                    desc(j, h).wait()

        def consume(j: Any) -> None:
            if not variant.endswith("serial"):
                for h in range(m):
                    desc(j, h).wait()
            with jax.named_scope("gather_select_pack"):
                values = []
                for h in range(m):
                    if parent_arm:
                        token = ids_ref[j * m + h]
                        values.append(
                            jnp.sum(
                                jnp.where(
                                    jnp.arange(tau)[:, None] == token % tau,
                                    buffers[j % depth, h, ...].astype(jnp.float32),
                                    0,
                                ),
                                axis=0,
                            ).astype(c.element_type)
                        )
                    else:
                        values.append(buffers[j % depth, h, :])
                native[...] = jnp.stack(values)
            with jax.named_scope("rounded_synthetic_expert"):
                native[...] = (native[...].astype(jnp.float32) * 1.125).astype(
                    c.element_type
                )
            with jax.named_scope("owned_combine"):
                for h in range(m):
                    token = ids_ref[j * m + h]
                    base = token // 8 * 8
                    old = out[pl.ds(base, 8), :]
                    out[pl.ds(base, 8), :] = old + jnp.where(
                        jnp.arange(8)[:, None] == token % 8,
                        native[h, :].astype(jnp.float32) * weights_ref[j * m + h],
                        0,
                    )

        ring_loop(jobs=n, depth=depth, unroll=c.unroll, start=start, consume=consume)

    fn = call(
        kernel,
        outputs=jax.ShapeDtypeStruct((rows, w), jnp.float32),
        scalar_inputs=2,
        inputs=(pl.no_block_spec,),
        scratch=(
            pltpu.VMEM(shape, c.element_type),
            pltpu.SemaphoreType.DMA((depth, m)),
            pltpu.VMEM((m, w), c.element_type),
        ),
        interpret=interpret,
        name=f"gather_{variant}",
    )
    transformed = np.asarray(
        (x.astype(jnp.float32) * 1.125).astype(c.element_type), np.float32
    )
    expected = np.zeros((rows, w), np.float32)
    for token, weight in zip(ids, weights, strict=True):
        expected[token] += transformed[token] * weight
    return Case(
        f"dispatch_{variant}",
        fn,
        (jnp.asarray(ids), jnp.asarray(weights), x),
        expected,
        {
            "chart": "queue independent hit reads; exact waits; aligned-parent select/pack or direct rows; source cast; one-owner VMEM combine",
            "pending_descriptors_upper_bound": depth * m,
            "scope": "synthetic rounded transform, not expert GEMM; direct bf16 row-DMA legality unproven",
            "ownership": "one output tile writer; hit order preserved, repeated tokens never race",
        },
    )


NAMES = (
    "stream_serial",
    "stream_manual",
    "stream_emitted",
    "dma_add",
    "broadcast_vmem",
    "broadcast_v_rows",
    "broadcast_smem_scalar",
    "broadcast_smem_stage",
    "broadcast_smem_vector",
    "compute_serial",
    "compute_interleaved",
    "state_serial",
    "state_overlap",
    "dispatch_row_serial",
    "dispatch_row_pipeline",
    "dispatch_parent_pipeline",
    "mla_expanded_pipeline",
    "mla_split_pipeline",
)


def blocked_chart(case: Case, c: PipelineConfig) -> str:
    rows = {
        "stream": [
            (
                "start future copy",
                f"[{c.block_rows},{c.columns}] {c.dtype}",
                "HBM/VMEM -> owned slot; exact extents, no transpose",
            ),
            (
                "wait current",
                "same source/destination slice as start",
                "per-slot completion; no group/global barrier",
            ),
            (
                "consume",
                "full panel (mixed) or first native plane (read/local)",
                "VMEM -> VReg; copy or sampled checksum",
            ),
            (
                "retire",
                "output panel or checksum plane",
                "separate output slots; drain before reuse",
            ),
        ],
        "broadcast": [
            (
                "load",
                "q[1,128] f32; x[8,128] f32",
                "SMEM scalar/staged or VMEM vector path",
            ),
            (
                "replicate",
                "[1,128] -> [8,128], or [8,1] -> [8,128]",
                "explicit broadcast; inspect sublane versus lane movement",
            ),
            (
                "multiply/store",
                "[8,128] -> [8,128] f32",
                "native VALU; no HBM intermediate",
            ),
        ],
        "compute": [
            (
                "dot",
                f"[{round_up(c.block_rows, 8)},128] @ [128,128]",
                "VMEM/VReg -> MXU -> f32 result",
            ),
            (
                "scale/exp",
                "matrix-shaped f32 result",
                "VALU/EUP; retain bounded SSA values across ticks",
            ),
            (
                "sum/divide/store",
                "[...,128] -> [...,1] -> [...,128]",
                "XLU/result path/VALU; older job interleaved with newer dot",
            ),
        ],
        "state": [
            (
                "load state",
                "[128,128] f32 V-first",
                "HBM -> VMEM; input serialization held constant",
            ),
            (
                "decay/predict/update",
                "q/k/v/g[128], beta scalar, S[128,128]",
                "broadcast, multiply, complete-K reduce, rank-one update",
            ),
            (
                "start writeback",
                "updated [128,128] f32",
                "vst -> DMA; immutable until completion",
            ),
            (
                "output/drain",
                "reduce_K(q*S) -> [1,128]",
                "read-only local consumer; wait before source reuse",
            ),
        ],
        "dispatch": [
            (
                "gather",
                f"{c.hits} hits; row[1,{c.columns}] or parent[{c.native_rows},{c.columns}]",
                "queue independent DMAs into per-hit slots; bf16 row legality is a question",
            ),
            (
                "select/pack",
                f"[{c.hits},{c.columns}] {c.dtype}",
                "parent mask/reduce or row load; stack into matrix-native scratch",
            ),
            (
                "transform/cast",
                f"[{c.hits},{c.columns}]",
                "synthetic multiply1.125, source dtype rounding; no expert GEMM",
            ),
            (
                "combine",
                f"owned f32[8,{c.columns}] parent",
                "aligned load, masked weighted add, store; ordered one-owner RMW",
            ),
        ],
        "mla": [
            (
                "prefetch K/V",
                f"D={c.depth}; K[128,192] or Kp[128,128]+Kr[128,64]; V[128,128]",
                "distinct semaphores; guarded D-1 prologue; future launch before current wait",
            ),
            (
                "QK/softmax",
                "Q[16,192] -> scores[16,128] -> p[16,128]",
                "MXU -> mask/max -> EUP; preserve online maximum and denominator",
            ),
            (
                "PV/update",
                "rounded p[16,128] @ V[128,128] -> f32[16,128]",
                "late V wait; carry recurrence; buffer reuse after last read",
            ),
            (
                "normalize/store",
                "acc[16,128], denominator[16]",
                "complete key loop; causal/tail masks; no cache ABI rewrite",
            ),
        ],
        "dma": [
            (
                "initialize/add/wait/store",
                "native input/output panel",
                "nonzero initial destination; explicit capability probe",
            )
        ],
    }[case.name.split("_", 1)[0]]
    return (
        "\n## Blocked operations\n\n| Operation | Shape | Storage/dependency |\n|---|---|---|\n"
        + "\n".join(f"| {op} | {shape} | {where} |" for op, shape, where in rows)
        + "\n\nAssigned layouts, instruction counts and exact cycle overlaps remain backend evidence.\n"
    )


def make_case(name: str, *, config: PipelineConfig, interpret: bool) -> Case:
    config.validate()
    if name not in NAMES:
        raise ValueError(f"unknown pipeline case {name}")
    if name.startswith("stream_"):
        return stream(config, variant=name.removeprefix("stream_"), interpret=interpret)
    if name.startswith("broadcast_"):
        return broadcast(
            config, variant=name.removeprefix("broadcast_"), interpret=interpret
        )
    if name.startswith("compute_"):
        return compute(
            config, variant=name.removeprefix("compute_"), interpret=interpret
        )
    if name.startswith("dispatch_"):
        return gather(
            config, variant=name.removeprefix("dispatch_"), interpret=interpret
        )
    if name.startswith("state_"):
        return state_writeback(
            config, overlap=name == "state_overlap", interpret=interpret
        )
    if name.startswith("mla_"):
        from layer_probe_kernels import Config, mla_layout

        c = Config(
            dtype=config.dtype,
            query_block=16,
            key_block=128,
            queries=17,
            keys=config.jobs * 128 - 1,
            query_offset=max(0, config.jobs * 128 - 18),
            pipeline_depth=config.depth,
        )
        return mla_layout(c, variant=name.removeprefix("mla_"), interpret=interpret)
    return dma_add(config, interpret=interpret)
