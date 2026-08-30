"""Combine microbench: as-coded (bg K=128 dots) vs K-FUSED (one K=512
dot) - the discriminating experiment for steps0_3_fp8.md Finding 1
before committing the y/ohg buffer restructure (lesson B.8).

The claim under test: per K-N tile the MXU pays the FULL M=512 LHS
stream (256 cyc bf16), so 4 half-deep (K=128) dots cost ~2x one
256-deep-tiled K=512 dot at identical MACs:
    as-coded  4 dots x 16 tiles x 256 cyc = 16384 cyc/group
    K-fused   32 tiles x 256 cyc          =  8192 cyc/group
At 128 grid steps (= 4 dispatches' worth of 32 groups) on 2 MXUs:
expect ~0.95 ms vs ~0.48 ms wall, ratio ~2x. Ratio well below 2x =
Mosaic already assembles 256-deep GMR fills across the split dots (or
the model is wrong) - either way the restructure is re-decided.

Both variants mirror the production combine_group: bcT row-chunking,
f32 accumulator in VMEM, register partial sums in the as-coded form.
Inputs live in VMEM scratch (loaded once at step 0) so NO DMA rides
the measurement - this is a pure MXU/VALU clock.

    python tmp/probe_combine_kfuse.py      # on the TPU VM
"""

import functools
import statistics
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

T, D = 512, 4096
BG, BEC = 4, 128          # bg dots of K=be*C=128; fused K = 512
K = BG * BEC
BCT = 256                 # production-range T chunk
STEPS = 128               # ~4 dispatches' worth of groups


def _kernel(ohg_hbm, y_hbm, o_ref, ohg_vmem, y_vmem, acc_vmem, sem,
            *, mode: str):
    blk = pl.program_id(0)
    kernel_form = mode == "kfused_kernel"

    @pl.when(blk == 0)
    def _load():
        if kernel_form:
            # populate the slot-major buffers: every ring slot gets a
            # BEC-column / BEC-row slice (slots p and p+BG identical -
            # only the read pattern matters)
            for p in range(2 * BG):
                pltpu.make_async_copy(
                    src_ref=ohg_hbm.at[:, pl.ds((p % BG) * BEC, BEC)],
                    dst_ref=ohg_vmem.at[p],
                    sem=sem.at[0]).start()
                pltpu.make_async_copy(
                    src_ref=ohg_hbm.at[:, pl.ds((p % BG) * BEC, BEC)],
                    dst_ref=ohg_vmem.at[p],
                    sem=sem.at[0]).wait()
            for p in range(BG):
                pltpu.make_async_copy(
                    src_ref=y_hbm.at[pl.ds(p * BEC, BEC), :],
                    dst_ref=y_vmem.at[p],
                    sem=sem.at[1]).start()
                pltpu.make_async_copy(
                    src_ref=y_hbm.at[pl.ds(p * BEC, BEC), :],
                    dst_ref=y_vmem.at[p],
                    sem=sem.at[1]).wait()
        else:
            for src, dst, s in ((ohg_hbm, ohg_vmem, 0),
                                (y_hbm, y_vmem, 1)):
                pltpu.make_async_copy(src_ref=src, dst_ref=dst,
                                      sem=sem.at[s]).start()
            for src, dst, s in ((ohg_hbm, ohg_vmem, 0),
                                (y_hbm, y_vmem, 1)):
                pltpu.make_async_copy(src_ref=src, dst_ref=dst,
                                      sem=sem.at[s]).wait()
        acc_vmem[...] = jnp.zeros_like(acc_vmem)

    if kernel_form:
        # THE KERNEL'S EXACT FORM (decode_kernel combine, fp8 path):
        # slot-major buffers, dynamic ring-slot reads concatenated
        # into one K=BG*BEC dot. Measures whether Mosaic materializes
        # the concatenated operands (review finding 3) or feeds the
        # dot from the slot reads at flat-buffer speed.
        y_wide = jnp.concatenate(
            [y_vmem[p] for p in range(BG)], axis=0)
        for t0 in range(0, T, BCT):
            rows = pl.ds(t0, BCT)
            ohg_wide = jnp.concatenate(
                [ohg_vmem[lax.rem(blk + p, 2 * BG), t0:t0 + BCT, :]
                 for p in range(BG)], axis=1)
            acc_vmem[rows, :] = acc_vmem[rows, :] + jnp.dot(
                ohg_wide, y_wide, preferred_element_type=jnp.float32)
    else:
        for t0 in range(0, T, BCT):
            rows = pl.ds(t0, BCT)
            if mode == "kfused_flat":
                # ONE K=512 dot per row chunk; MRB store-add owns the
                # cross-tile accumulation.
                partial = jnp.dot(ohg_vmem[rows, :], y_vmem[...],
                                  preferred_element_type=jnp.float32)
            else:
                # bf16 combine_group: bg K=128 dots, register sums
                partial = jnp.zeros((BCT, D), jnp.float32)
                for p in range(BG):
                    partial = partial + jnp.dot(
                        ohg_vmem[rows, pl.ds(p * BEC, BEC)],
                        y_vmem[pl.ds(p * BEC, BEC), :],
                        preferred_element_type=jnp.float32)
            acc_vmem[rows, :] = acc_vmem[rows, :] + partial

    @pl.when(blk == pl.num_programs(0) - 1)
    def _emit():
        o_ref[...] = acc_vmem[0:8, 0:128]


def build(mode: str):
    if mode == "kfused_kernel":
        bufs = [pltpu.VMEM((2 * BG, T, BEC), jnp.bfloat16),  # ohg ring
                pltpu.VMEM((BG, BEC, D), jnp.bfloat16)]      # y slots
    else:
        bufs = [pltpu.VMEM((T, K), jnp.bfloat16),   # ohg (group slice)
                pltpu.VMEM((K, D), jnp.bfloat16)]   # y group-flat
    return pl.pallas_call(
        functools.partial(_kernel, mode=mode),
        grid=(STEPS,),
        in_specs=[pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                  pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)],
        out_specs=pl.BlockSpec((8, 128), lambda i: (0, 0)),
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        scratch_shapes=bufs + [
            pltpu.VMEM((T, D), jnp.float32),       # acc
            pltpu.SemaphoreType.DMA((2,)),
        ],
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=100 * 2**20),
        name=f"combine_{mode}",
    )


def main() -> None:
    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    print(f"probe_combine_kfuse: commit {rev}, jax {jax.__version__}, "
          f"device {jax.devices()[0].device_kind}; T={T} D={D} "
          f"K={K} ({BG}x{BEC}) bcT={BCT} steps={STEPS}")
    rng = np.random.default_rng(0)
    ohg = jnp.asarray(rng.standard_normal((T, K)), jnp.bfloat16)
    y = jnp.asarray(rng.standard_normal((K, D)), jnp.bfloat16)
    results = {}
    # ANSWERED (2026-08-30, ascoded/kfused_flat): 734.9 vs 473.0 us =
    # 1.55x; kfused_flat 3.70 us/group vs the per-tile model's 3.72.
    # kfused_kernel is the NEW variant (review finding 3): the
    # decode_kernel's exact slot-major + concatenated-read form.
    # kfused_kernel ~= kfused_flat => Mosaic feeds the dot from the
    # slot reads (concat is free) and the kernel stands; materially
    # slower => restructure the kernel's ohg/y buffers to the flat
    # layouts (kernel_plan sec 2's original design).
    for mode in ("ascoded", "kfused_flat", "kfused_kernel"):
        fn = jax.jit(build(mode))
        try:
            out0 = jax.block_until_ready(fn(ohg, y))
        except Exception as ex:
            print(f"  {mode:>13}: failed ({type(ex).__name__}: "
                  f"{str(ex).splitlines()[0][:90]})")
            continue
        ts = []
        for _ in range(10):
            t0 = time.perf_counter()
            jax.block_until_ready(fn(ohg, y))
            ts.append(time.perf_counter() - t0)
        t = min(ts)
        results[mode] = (t, out0)
        print(f"  {mode:>13}: {t * 1e6:8.1f} us  "
              f"(median {statistics.median(ts) * 1e6:.1f} us)")
    if "ascoded" in results and "kfused_flat" in results:
        ratio = results["ascoded"][0] / results["kfused_flat"][0]
        diff = float(jnp.max(jnp.abs(
            results["ascoded"][1] - results["kfused_flat"][1])))
        print(f"  ascoded / kfused_flat = {ratio:.2f}x "
              f"(model ~2.0); max|diff| = {diff:.3e}")
    if "kfused_kernel" in results and "kfused_flat" in results:
        kratio = results["kfused_kernel"][0] / results["kfused_flat"][0]
        print(f"  kfused_kernel / kfused_flat = {kratio:.2f}x "
              f"(~1.0 = concat free, keep kernel as is; >1.2 = "
              f"restructure to flat buffers)")


if __name__ == "__main__":
    main()
