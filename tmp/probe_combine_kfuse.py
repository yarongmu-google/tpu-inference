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
            *, fused: bool):
    blk = pl.program_id(0)

    @pl.when(blk == 0)
    def _load():
        for src, dst, s in ((ohg_hbm, ohg_vmem, 0), (y_hbm, y_vmem, 1)):
            pltpu.make_async_copy(src_ref=src, dst_ref=dst,
                                  sem=sem.at[s]).start()
        for src, dst, s in ((ohg_hbm, ohg_vmem, 0), (y_hbm, y_vmem, 1)):
            pltpu.make_async_copy(src_ref=src, dst_ref=dst,
                                  sem=sem.at[s]).wait()
        acc_vmem[...] = jnp.zeros_like(acc_vmem)

    for t0 in range(0, T, BCT):
        rows = pl.ds(t0, BCT)
        if fused:
            # ONE K=512 dot per row chunk; MRB store-add owns the
            # cross-tile accumulation.
            partial = jnp.dot(ohg_vmem[rows, :], y_vmem[...],
                              preferred_element_type=jnp.float32)
        else:
            # production combine_group: bg K=128 dots, register sums
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


def build(fused: bool):
    return pl.pallas_call(
        functools.partial(_kernel, fused=fused),
        grid=(STEPS,),
        in_specs=[pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                  pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)],
        out_specs=pl.BlockSpec((8, 128), lambda i: (0, 0)),
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        scratch_shapes=[
            pltpu.VMEM((T, K), jnp.bfloat16),      # ohg (group slice)
            pltpu.VMEM((K, D), jnp.bfloat16),      # y group-flat
            pltpu.VMEM((T, D), jnp.float32),       # acc
            pltpu.SemaphoreType.DMA((2,)),
        ],
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=100 * 2**20),
        name="combine_kfused" if fused else "combine_ascoded",
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
    for fused in (False, True):
        label = "K-fused " if fused else "as-coded"
        fn = jax.jit(build(fused))
        out0 = jax.block_until_ready(fn(ohg, y))
        ts = []
        for _ in range(10):
            t0 = time.perf_counter()
            jax.block_until_ready(fn(ohg, y))
            ts.append(time.perf_counter() - t0)
        t = min(ts)
        results[fused] = (t, out0)
        print(f"  {label}: {t * 1e6:8.1f} us  "
              f"(median {statistics.median(ts) * 1e6:.1f} us)")
    ratio = results[False][0] / results[True][0]
    # numeric sanity: same sums, different bracketing -> tolerance
    diff = float(jnp.max(jnp.abs(results[False][1] - results[True][1])))
    print(f"  ratio as-coded / K-fused = {ratio:.2f}x "
          f"(model predicts ~2.0); max|diff| on probe tile = {diff:.3e}")


if __name__ == "__main__":
    main()
