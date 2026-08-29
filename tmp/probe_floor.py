"""Per-step floor probe: the decode kernel runs 2777us (21.7us/step x
128) with EVERY stage stubbed (ablate=all). What does an empty grid
step cost, and is it the window machinery or the grid itself?

Four kernels, all with an empty body (one tiny epilogue write so the
module is not trivially dead):

  win128 / win64    the decode kernel's constant-window structure
                    (tokens [64,4096] bf16, route [512,4096] f32, out
                    [512,4096] bf16 windows; acc-sized f32 scratch),
                    128 vs 64 grid steps
  hbm128 / hbm64    identical, but every input pinned in HBM (no
                    windows at all); out is a tiny (8,128) window

Readout: win ~20us/step -> the floor is Pallas per-step machinery
around constant windows - evict them to manual refs like the weights.
win ~= hbm ~= 0 -> the floor lives in OUR kernel's body/structure -
read the ablate=all dump. Time scaling ~2x from 64->128 confirms
per-step (not per-call) cost.

    python tmp/probe_floor.py          # on the TPU VM
"""

import functools
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

T, D, E = 512, 4096, 512             # decode-shape constants
TL = 64                              # tokens_local rows (T/P)


def _kernel(tok, route, o_ref, acc, blk_scr, *, out_windowed: bool):
    blk = pl.program_id(0)
    nblk = pl.num_programs(0)

    @pl.when(blk == nblk - 1)
    def _epilogue():
        if out_windowed:
            o_ref[...] = acc[...].astype(o_ref.dtype)
        else:
            o_ref[...] = acc[0:8, 0:128].astype(o_ref.dtype)


def build(steps: int, windowed: bool):
    if windowed:
        in_specs = [
            pl.BlockSpec((TL, D), lambda i: (0, 0)),
            pl.BlockSpec((E, D), lambda i: (0, 0)),
        ]
        out_specs = pl.BlockSpec((T, D), lambda i: (0, 0))
        out_shape = jax.ShapeDtypeStruct((T, D), jnp.bfloat16)
    else:
        in_specs = [pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                    pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)]
        out_specs = pl.BlockSpec((8, 128), lambda i: (0, 0))
        out_shape = jax.ShapeDtypeStruct((8, 128), jnp.bfloat16)
    return pl.pallas_call(
        functools.partial(_kernel, out_windowed=windowed),
        grid=(steps,),
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=out_shape,
        scratch_shapes=[
            pltpu.VMEM((T, D), jnp.float32),                 # acc-sized
            pltpu.VMEM((4, D, 256), jnp.bfloat16),           # slab-sized
        ],
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=100 * 2**20),
    )


def main() -> None:
    rng = np.random.default_rng(0)
    tok = jax.device_put(jnp.asarray(
        rng.standard_normal((TL, D)), jnp.bfloat16), jax.devices()[0])
    route = jax.device_put(jnp.asarray(
        rng.standard_normal((E, D)), jnp.float32), jax.devices()[0])
    for name, steps, windowed in (("win128", 128, True), ("win64", 64, True),
                                  ("hbm128", 128, False),
                                  ("hbm64", 64, False)):
        fn = jax.jit(build(steps, windowed))
        try:
            jax.block_until_ready(fn(tok, route))
        except Exception as ex:
            print(f"{name:>8}: failed ({type(ex).__name__}: "
                  f"{str(ex).splitlines()[0][:90]})")
            continue
        ts = []
        for _ in range(10):
            t0 = time.perf_counter()
            jax.block_until_ready(fn(tok, route))
            ts.append(time.perf_counter() - t0)
        t = min(ts)
        print(f"{name:>8}: {t * 1e6:8.1f} us total   "
              f"{t * 1e6 / steps:6.2f} us/step   "
              f"(median {statistics.median(ts) * 1e6:.1f} us)")


if __name__ == "__main__":
    main()
