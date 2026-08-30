"""Weight-stream bandwidth probe: WHY does the window pipeline run at
~494 GB/s when the DMA spec says one flow should sustain 3207?

Times three manual-DMA strategies against the known window number
(ablate=weights: 3.27ms for 1.61GB = 494 GB/s):

  manual1   ONE descriptor per block [be, D, 2I] -> (16,128) scratch double buffer
  manual4   FOUR per-expert descriptors per block, separate semaphores
            (candidate thread parallelism: 6 eligible HBM->VMEM threads)
  manual8   be=8 blocks, 8 descriptors (size x parallelism scaling)

Readout: manual1 ~3000 GB/s -> the WINDOW descriptor is derated (layout
reorder, small inner runs) - fix = manual blocked fetch in the kernel.
manual1 ~500 but manual4 ~4x -> per-thread cap - fix = split the fetch
across parallel DMAs. All ~500 -> the target number itself is wrong for
this part; rethink the floor.

    python tmp/probe_wbw.py            # on the TPU VM

2026-08-30: run_pair now also runs the E4M3 pair (768 MiB) after the
bf16 pair - the fp8-kernel floor constant. Expect ~equal GB/s, ~half
the wall time (byte-denominated DMA law).
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
IP = I2 // 2                         # w2 local shard is [E, IP, D]


def _kernel(w_hbm, o_ref, w_bufs, sems, *, be: int, nsplit: int):
    blk = pl.program_id(0)
    nblk = pl.num_programs(0)
    slot = jax.lax.rem(blk, 2)
    nxt = jax.lax.rem(blk + 1, 2)
    esz = be // nsplit               # experts per descriptor

    def fetch(block, buf_slot):
        e0 = block * be
        for s in range(nsplit):
            pltpu.make_async_copy(
                src_ref=w_hbm.at[pl.ds(e0 + s * esz, esz)],
                dst_ref=w_bufs.at[buf_slot, pl.ds(s * esz, esz)],
                sem=sems.at[buf_slot, s],
            ).start()

    def wait(buf_slot):
        for s in range(nsplit):
            pltpu.make_async_copy(
                src_ref=w_hbm.at[pl.ds(0, esz)],
                dst_ref=w_bufs.at[buf_slot, pl.ds(0, esz)],
                sem=sems.at[buf_slot, s],
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
        o_ref[...] = w_bufs[slot, 0, 0:8, 0:128].astype(jnp.float32)


def build(be: int, nsplit: int):
    nblk = E // be
    return pl.pallas_call(
        functools.partial(_kernel, be=be, nsplit=nsplit),
        grid=(nblk,),
        in_specs=[pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)],
        out_specs=pl.BlockSpec((8, 128), lambda i: (0, 0)),
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        scratch_shapes=[
            pltpu.VMEM((2, be, D, I2), jnp.bfloat16),        # double buffer
            pltpu.SemaphoreType.DMA((2, nsplit)),
        ],
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=100 * 2**20),
    )


def _kernel_pair(w1_hbm, w2_hbm, o_ref, w1_bufs, w2_bufs, sems, *, be: int):
    """The production fetch_weights structure, nothing else: per block,
    w1 [be, D, 2I] + w2 [be, I, D] slabs, one contiguous descriptor
    each, blocked loading through a 2-slot double buffer, prefetch-1."""
    blk = pl.program_id(0)
    nblk = pl.num_programs(0)
    slot = jax.lax.rem(blk, 2)
    nxt = jax.lax.rem(blk + 1, 2)

    def fetch(block, buf_slot, *, wait=False):
        for hbm, bufs, s in ((w1_hbm, w1_bufs, 0), (w2_hbm, w2_bufs, 1)):
            cp = pltpu.make_async_copy(
                src_ref=hbm.at[pl.ds(block * be, be)],
                dst_ref=bufs.at[buf_slot],
                sem=sems.at[buf_slot, s])
            if wait:
                cp.wait()
            else:
                cp.start()

    @pl.when(blk == 0)
    def _prologue():
        fetch(0, 0)

    @pl.when(blk + 1 < nblk)
    def _prefetch():
        fetch(blk + 1, nxt)

    fetch(blk, slot, wait=True)

    @pl.when(blk == nblk - 1)
    def _epilogue():
        o_ref[...] = (w1_bufs[slot, 0, 0:8, 0:128].astype(jnp.float32)
                      + w2_bufs[slot, 0, 0:8, 0:128].astype(jnp.float32))


def build_pair(be: int, dtype=jnp.bfloat16):
    return pl.pallas_call(
        functools.partial(_kernel_pair, be=be),
        grid=(E // be,),
        in_specs=[pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                  pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)],
        out_specs=pl.BlockSpec((8, 128), lambda i: (0, 0)),
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        scratch_shapes=[
            pltpu.VMEM((2, be, D, I2), dtype),               # w1 bufs
            pltpu.VMEM((2, be, IP, D), dtype),               # w2 bufs
            pltpu.SemaphoreType.DMA((2, 2)),
        ],
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=100 * 2**20),
    )


def _timed(fn, *args, iters: int = 10):
    jax.block_until_ready(fn(*args))
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append(time.perf_counter() - t0)
    return min(ts), statistics.median(ts)


def run_pair(w1_np: np.ndarray, w2_np: np.ndarray) -> None:
    """Stream the ACTUAL MoE weight pair (1.5 GiB, both tensors) through
    the production blocked-fetch structure - first on one device, then on ALL
    devices at once. The pure-stream time compares 1:1 against the
    kernel's (ablate=weights - ablate=all): no shape or byte-count
    normalization. If per-device bandwidth collapses only in the
    concurrent row, full-machine concurrency (shared HBM or fabric) is
    the derate, not the kernel."""
    per_dev_bytes = w1_np.nbytes + w2_np.nbytes
    print(f"weight pair: {per_dev_bytes / 2**30:.2f} GiB per device "
          f"(w1 {E}x{D}x{I2} + w2 {E}x{IP}x{D} {w1_np.dtype})")

    dev0 = jax.devices()[0]
    fn1 = jax.jit(build_pair(be=4, dtype=w1_np.dtype))
    try:
        t, med = _timed(fn1, jax.device_put(jnp.asarray(w1_np), dev0),
                        jax.device_put(jnp.asarray(w2_np), dev0))
        print(f" wpair_1dev: {t * 1e6:8.1f} us   "
              f"{per_dev_bytes / t / 1e9:7.1f} GB/s   "
              f"(median {med * 1e6:.1f} us)")
    except Exception as ex:
        print(f" wpair_1dev: failed ({type(ex).__name__}: "
              f"{str(ex).splitlines()[0][:90]})")

    p = len(jax.devices())
    if p < 2:
        print(f" wpair_{p}dev: skipped (1 device)")
        return
    mesh = jax.sharding.Mesh(np.array(jax.devices()), ("x",))
    P = jax.sharding.PartitionSpec
    w1_all = jax.make_array_from_callback(
        (p * E, D, I2), jax.sharding.NamedSharding(mesh, P("x", None, None)),
        lambda idx: w1_np)   # every device: its own full local shard
    w2_all = jax.make_array_from_callback(
        (p * E, IP, D), jax.sharding.NamedSharding(mesh, P("x", None, None)),
        lambda idx: w2_np)
    fn8 = jax.jit(jax.shard_map(
        build_pair(be=4, dtype=w1_np.dtype), mesh=mesh,
        in_specs=(P("x", None, None), P("x", None, None)),
        out_specs=P("x", None), check_vma=False))
    try:
        t, med = _timed(fn8, w1_all, w2_all)
        print(f" wpair_{p}dev: {t * 1e6:8.1f} us   "
              f"{per_dev_bytes / t / 1e9:7.1f} GB/s per device   "
              f"({per_dev_bytes * p / t / 1e9:.0f} aggregate, "
              f"median {med * 1e6:.1f} us)")
    except Exception as ex:
        print(f" wpair_{p}dev: failed ({type(ex).__name__}: "
              f"{str(ex).splitlines()[0][:90]})")


def main() -> None:
    rng = np.random.default_rng(0)
    w_np = np.asarray(rng.standard_normal((E, D, I2)), jnp.bfloat16)
    w2_np = np.asarray(rng.standard_normal((E, IP, D)), jnp.bfloat16)
    w = jax.device_put(jnp.asarray(w_np), jax.devices()[0])
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
    run_pair(w_np, w2_np)

    # fp8 pair: the fp8-kernel floor constant (qwen35_fp8 proposal sec
    # 8.1). The DMA access-size law is byte-denominated (512 B unit box
    # is [packing, 128] ELEMENTS), so the fp8 pair (768 MiB) should run
    # at the SAME GB/s as bf16 = ~half the time; a bandwidth deviation
    # here IS a finding and re-prices the whole fp8 plan.
    w1_f8 = np.asarray(w_np, jnp.float8_e4m3fn)   # values irrelevant
    w2_f8 = np.asarray(w2_np, jnp.float8_e4m3fn)
    run_pair(w1_f8, w2_f8)


if __name__ == "__main__":
    main()
