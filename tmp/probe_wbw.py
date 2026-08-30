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


def build_pair(be: int, dtype=jnp.bfloat16, num_experts: int = E):
    return pl.pallas_call(
        functools.partial(_kernel_pair, be=be),
        grid=(num_experts // be,),
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


def _kernel_null(x_ref, o_ref, sem):
    """Empty grid steps: no DMA, no compute - wall(null) isolates the
    CALL ENVELOPE (jit dispatch + shard_map + launch + sync), and
    null(128) - null(1) isolates the per-step grid floor."""
    del sem
    blk = pl.program_id(0)

    @pl.when(blk == pl.num_programs(0) - 1)
    def _emit():
        o_ref[...] = jnp.zeros_like(o_ref) + x_ref[0, 0]


def build_null(steps: int):
    return pl.pallas_call(
        _kernel_null,
        grid=(steps,),
        in_specs=[pl.BlockSpec((8, 128), lambda i: (0, 0))],
        out_specs=pl.BlockSpec((8, 128), lambda i: (0, 0)),
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        scratch_shapes=[pltpu.SemaphoreType.DMA],
    )


def run_null() -> None:
    """Round-3 discriminator for the ENVELOPE hypothesis (H1): the
    round-1/2 walls fit wall = bytes/BW + E with BW ~= 3.3-3.5 TB/s
    (near spec) and E_solo ~= 166 us / E_8dev ~= 363 us ~= the bench
    harness envelope (354 us, floor_decomp 2026-08-29). H1 PREDICTS:
    null_1dev(1) ~= 150-180, null_8dev(1) ~= 330-400, and steps=128
    adds < 60 us. If null_8dev is instead ~= 0, the fixed cost lives
    device-side and the fp8 stream floor really is ~600 us.

    ANSWERED (2026-08-30): null_1dev 132.5/141.0 us (steps 1/128) -
    clean solo envelope + a 67-226 ns/step floor (negligible; explains
    the flat be sweep). null_8dev came back 706.9/735.6 - INVALID as
    built: the original xs = jnp.zeros() landed unsharded on device 0,
    so every call paid an input reshard (fixed below for posterity).
    H1 was instead CONFIRMED by the E-sweep: linear with slope
    3149 GB/s/dev and intercept 366 us (max residual 7.9 us) - see
    the E-sweep comment. Device-side fp8 stream ~256 us => the fp8
    kernel is MXU-BOUND (357 > 256)."""
    x = jnp.zeros((8, 128), jnp.float32)
    for steps in (1, 128):
        fn = jax.jit(build_null(steps))
        t, med = _timed(fn, x)
        print(f" null_1dev[steps={steps}]: {t * 1e6:8.1f} us   "
              f"(median {med * 1e6:.1f} us)")
    p = len(jax.devices())
    if p < 2:
        return
    mesh = jax.sharding.Mesh(np.array(jax.devices()), ("x",))
    P = jax.sharding.PartitionSpec
    # sharded at creation - an unsharded jnp.zeros lands on device 0
    # and every call then pays an input reshard that swamps the
    # envelope being measured (the round-3 null_8dev bug)
    xs = jax.device_put(
        np.zeros((8 * p, 128), np.float32),
        jax.sharding.NamedSharding(mesh, P("x", None)))
    for steps in (1, 128):
        fn = jax.jit(jax.shard_map(
            build_null(steps), mesh=mesh,
            in_specs=(P("x", None),), out_specs=P("x", None),
            check_vma=False))
        t, med = _timed(fn, xs)
        print(f" null_{p}dev[steps={steps}]: {t * 1e6:8.1f} us   "
              f"(median {med * 1e6:.1f} us)")


def _timed(fn, *args, iters: int = 10):
    jax.block_until_ready(fn(*args))
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append(time.perf_counter() - t0)
    return min(ts), statistics.median(ts)


def run_pair(w1_np: np.ndarray, w2_np: np.ndarray, be: int = 4) -> None:
    """Stream the ACTUAL MoE weight pair (1.5 GiB, both tensors) through
    the production blocked-fetch structure - first on one device, then on ALL
    devices at once. The pure-stream time compares 1:1 against the
    kernel's (ablate=weights - ablate=all): no shape or byte-count
    normalization. If per-device bandwidth collapses only in the
    concurrent row, full-machine concurrency (shared HBM or fabric) is
    the derate, not the kernel."""
    per_dev_bytes = w1_np.nbytes + w2_np.nbytes
    print(f"weight pair: {per_dev_bytes / 2**30:.2f} GiB per device "
          f"(w1 {w1_np.shape[0]}x{D}x{I2} + w2 {w2_np.shape[0]}x{IP}x{D} "
          f"{w1_np.dtype})")

    dev0 = jax.devices()[0]
    nexp = w1_np.shape[0]
    fn1 = jax.jit(build_pair(be=be, dtype=w1_np.dtype, num_experts=nexp))
    try:
        t, med = _timed(fn1, jax.device_put(jnp.asarray(w1_np), dev0),
                        jax.device_put(jnp.asarray(w2_np), dev0))
        print(f" wpair_1dev[be={be}]: {t * 1e6:8.1f} us   "
              f"{per_dev_bytes / t / 1e9:7.1f} GB/s   "
              f"(median {med * 1e6:.1f} us)")
    except Exception as ex:
        print(f" wpair_1dev[be={be}]: failed ({type(ex).__name__}: "
              f"{str(ex).splitlines()[0][:90]})")

    p = len(jax.devices())
    if p < 2:
        print(f" wpair_{p}dev: skipped (1 device)")
        return
    mesh = jax.sharding.Mesh(np.array(jax.devices()), ("x",))
    P = jax.sharding.PartitionSpec
    w1_all = jax.make_array_from_callback(
        (p * nexp, D, I2),
        jax.sharding.NamedSharding(mesh, P("x", None, None)),
        lambda idx: w1_np)   # every device: its own full local shard
    w2_all = jax.make_array_from_callback(
        (p * nexp, IP, D),
        jax.sharding.NamedSharding(mesh, P("x", None, None)),
        lambda idx: w2_np)
    fn8 = jax.jit(jax.shard_map(
        build_pair(be=be, dtype=w1_np.dtype, num_experts=nexp), mesh=mesh,
        in_specs=(P("x", None, None), P("x", None, None)),
        out_specs=P("x", None), check_vma=False))
    try:
        t, med = _timed(fn8, w1_all, w2_all)
        print(f" wpair_{p}dev[be={be}]: {t * 1e6:8.1f} us   "
              f"{per_dev_bytes / t / 1e9:7.1f} GB/s per device   "
              f"({per_dev_bytes * p / t / 1e9:.0f} aggregate, "
              f"median {med * 1e6:.1f} us)")
    except Exception as ex:
        print(f" wpair_{p}dev[be={be}]: failed ({type(ex).__name__}: "
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
    # ROUND 1 ANSWERED (2026-08-30, be=4): byte-rate parity FAILED -
    # fp8 8dev 592-600us, not ~424. ROUND 2 ANSWERED (be sweep): FLAT
    # (fp8 579-631 at be=4/8/16; bf16 837-841 at be=8) - the fixed
    # cost is NOT per-step. H1 (call ENVELOPE): all 8 wall points fit
    # wall = bytes/BW + E with BW ~= 3.49 TB/s solo / 3.34 TB/s 8dev
    # and E ~= 166 solo / 363 8dev - and 363 matches the bench harness
    # envelope (354us, floor_decomp). If H1 holds, the TRUE device
    # stream is bf16 ~482 / fp8 ~241us, and the fp8 kernel is
    # MXU-BOUND (357us > 241) - run_null() is the discriminator.
    w1_f8 = np.asarray(w_np, jnp.float8_e4m3fn)   # values irrelevant
    w2_f8 = np.asarray(w2_np, jnp.float8_e4m3fn)
    for be in (4, 8, 16):
        run_pair(w1_f8, w2_f8, be=be)
    run_pair(w_np, w2_np, be=8)

    # ROUND 3: (a) null kernel - measures the envelope directly, see
    # run_null docstring for the H1 predictions; (b) fp8 byte sweep
    # over expert count (lesson 14: scaling separates rate from fixed
    # cost, catches nonlinearity) - H1 predicts wall(e) =
    # e/512 * 241us + 363: e=64 -> ~393, 128 -> ~423, 256 -> ~483.
    # Nonlinearity here = a NEW hidden cost (the bf16 XLA-copy
    # pattern); linear-with-intercept-363 = H1 confirmed twice over.
    # ANSWERED (2026-08-30): 395.4 / 427.3 / 501.6 (+ 618.5 at 512) -
    # LINEAR: lstsq slope = 3149 GB/s/dev (~= 3207 CMN spec),
    # intercept 366 us (~= the 354 us harness envelope), max residual
    # 7.9 us; solo fit 3323 GB/s + 147 us. H1 CONFIRMED: the historic
    # "1.83-1.93 TB/s under load" was envelope-conflated wall; the
    # device-side stream is fp8 ~256 / bf16 ~511 us => the fp8 kernel
    # is MXU-BOUND (357 > 256) and the bf16 kernel's 848 device ~=
    # MXU busy ~715 + floors, not the stream. See qwen35_fp8
    # proposal.md sec 0d for the re-based constants.
    run_null()
    for e_cnt in (64, 128, 256):
        run_pair(w1_f8[:e_cnt], w2_f8[:e_cnt], be=4)


if __name__ == "__main__":
    main()
