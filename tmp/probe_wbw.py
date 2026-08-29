"""Weight-stream bandwidth probe: WHY does the window pipeline run at
~494 GB/s when the DMA spec says one flow should sustain 3207?

Times three manual-DMA strategies against the known window number
(ablate=weights: 3.27ms for 1.61GB = 494 GB/s):

  manual1   ONE descriptor per block [be, D, 2I] -> (16,128) scratch ring
  manual4   FOUR per-expert descriptors per block, separate semaphores
            (candidate thread parallelism: 6 eligible HBM->VMEM threads)
  manual8   be=8 blocks, 8 descriptors (size x parallelism scaling)

Readout: manual1 ~3000 GB/s -> the WINDOW descriptor is derated (layout
reorder, small inner runs) - fix = manual fetch ring in the kernel.
manual1 ~500 but manual4 ~4x -> per-thread cap - fix = split the fetch
across parallel DMAs. All ~500 -> the target number itself is wrong for
this part; rethink the floor.

    python tmp/probe_wbw.py            # on the TPU VM
"""

import functools
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

E, D, I2 = 512, 4096, 256            # the w1 local shard shape


def _kernel(w_hbm, o_ref, w_ring, sems, *, be: int, nsplit: int):
    blk = pl.program_id(0)
    nblk = pl.num_programs(0)
    slot = jax.lax.rem(blk, 2)
    nxt = jax.lax.rem(blk + 1, 2)
    esz = be // nsplit               # experts per descriptor

    def fetch(block, ring_slot):
        e0 = block * be
        for s in range(nsplit):
            pltpu.make_async_copy(
                src_ref=w_hbm.at[pl.ds(e0 + s * esz, esz)],
                dst_ref=w_ring.at[ring_slot, pl.ds(s * esz, esz)],
                sem=sems.at[ring_slot, s],
            ).start()

    def wait(ring_slot):
        for s in range(nsplit):
            pltpu.make_async_copy(
                src_ref=w_hbm.at[pl.ds(0, esz)],
                dst_ref=w_ring.at[ring_slot, pl.ds(0, esz)],
                sem=sems.at[ring_slot, s],
            ).wait()

    @pl.when(blk == 0)
    def _prologue():
        fetch(0, 0)

    @pl.when(blk + 1 < nblk)
    def _prefetch():
        fetch(blk + 1, nxt)

    wait(slot)

    @pl.when(blk == nblk - 1)
    def _epilogue():
        o_ref[...] = w_ring[slot, 0, 0:8, 0:128].astype(jnp.float32)


def build(be: int, nsplit: int):
    nblk = E // be
    return pl.pallas_call(
        functools.partial(_kernel, be=be, nsplit=nsplit),
        grid=(nblk,),
        in_specs=[pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)],
        out_specs=pl.BlockSpec((8, 128), lambda i: (0, 0)),
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        scratch_shapes=[
            pltpu.VMEM((2, be, D, I2), jnp.bfloat16),        # ring
            pltpu.SemaphoreType.DMA((2, nsplit)),
        ],
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=100 * 2**20),
    )


def main() -> None:
    w = jnp.asarray(
        np.random.default_rng(0).standard_normal((E, D, I2)), jnp.bfloat16)
    w = jax.device_put(w, jax.devices()[0])
    total_bytes = w.size * 2
    print(f"array: {total_bytes / 2**30:.2f} GiB  ({E}x{D}x{I2} bf16)")
    for name, be, nsplit in (("manual1", 4, 1), ("manual2", 4, 2),
                             ("manual4", 4, 4), ("manual8x1", 8, 1),
                             ("manual8x8", 8, 8)):
        fn = jax.jit(build(be, nsplit))
        try:
            jax.block_until_ready(fn(w))
        except Exception as ex:
            print(f"{name:>10}: failed ({type(ex).__name__}: "
                  f"{str(ex).splitlines()[0][:90]})")
            continue
        ts = []
        for _ in range(10):
            t0 = time.perf_counter()
            jax.block_until_ready(fn(w))
            ts.append(time.perf_counter() - t0)
        t = min(ts)
        print(f"{name:>10}: {t * 1e6:8.1f} us   "
              f"{total_bytes / t / 1e9:7.1f} GB/s   "
              f"(median {statistics.median(ts) * 1e6:.1f} us)")


if __name__ == "__main__":
    main()
