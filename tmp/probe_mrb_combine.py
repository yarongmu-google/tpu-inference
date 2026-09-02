# MRB-accumulation probe: the v2 combine's split-K contraction as
# (A) chunked dot_general - the current kernel's structure, each
#     chunk's result drained and re-added (the measured vadd.f32
#     chain), vs
# (B) manual MXU control - pltpu.matmul_push_rhs / matmul_acc_lhs /
#     matmul_pop: stage each 256x256 RHS chunk, accumulate f32
#     in-place in the matrix result buffer across the WHOLE K, pop
#     once, and
# (C) B split across both MXUs (two accumulators, one final add).
#
# Shape = one 256-col output strip of the combine at the T=1024
# serving winner: M=512 rows, K=32768 (E x C slots), N=256. Verdicts:
#   - B/C vs A wall time: the drain+vadd tax, measured directly;
#   - B vs A allclose (f32 accumulation both; ordering differs);
#   - and the COEXISTENCE question: variant D runs a dot_general in
#     the same kernel right after the manual pop - does Mosaic's own
#     MXU allocation collide with manual acc_addr use?
#
# Local dev gate: --export lowers for the TPU target on CPU (the
# same trick as the kernel's lowering tests) - trace/lowering bugs
# surface without hardware. Run on the VM (serving env) for numbers:
#   python tmp/probe_mrb_combine.py
#   python tmp/probe_mrb_combine.py --export   # local, no TPU

import argparse
import functools
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

M, K, N = 512, 32768, 256
GROUP = 512          # variant A's chunk (bg*be*C at the T=1024 winner)
NATIVE = 256         # push_rhs granule


def kernel_dot(lhs_ref, rhs_ref, out_ref):
    acc = jnp.zeros((M, N), jnp.float32)
    for g in range(K // GROUP):
        acc += jax.lax.dot_general(
            lhs_ref[:, g * GROUP:(g + 1) * GROUP],
            rhs_ref[g * GROUP:(g + 1) * GROUP, :],
            dimension_numbers=(((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32)
    out_ref[...] = acc


def kernel_mrb(lhs_ref, rhs_ref, out_ref, *, mxus: int):
    per = K // NATIVE // mxus
    for m in range(mxus):
        for i in range(per):
            kc = m * per + i
            pltpu.matmul_push_rhs(
                rhs_ref[kc * NATIVE:(kc + 1) * NATIVE, :],
                staging_register=kc % 2, mxu_index=m)
            pltpu.matmul_acc_lhs(
                0, lhs_ref[:, kc * NATIVE:(kc + 1) * NATIVE],
                mxu_index=m, load_staged_rhs=kc % 2)
    acc = pltpu.matmul_pop(0, (M, N), jnp.float32, 0)
    if mxus == 2:
        acc = acc + pltpu.matmul_pop(0, (M, N), jnp.float32, 1)
    out_ref[...] = acc


def kernel_coexist(lhs_ref, rhs_ref, out_ref):
    """Manual MRB walk followed by a Mosaic-scheduled dot_general in
    the SAME kernel - the collision smoke for the real integration."""
    kernel_mrb(lhs_ref, rhs_ref, out_ref, mxus=1)
    tail = jax.lax.dot_general(
        lhs_ref[:, :GROUP], rhs_ref[:GROUP, :],
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32)
    out_ref[...] = out_ref[...] + tail


def build(body, **kw):
    return pl.pallas_call(
        functools.partial(body, **kw) if kw else body,
        out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32),
        compiler_params=getattr(pltpu, "CompilerParams",
                                getattr(pltpu, "TPUCompilerParams", None))(
            vmem_limit_bytes=100 * 2**20),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--export", action="store_true",
                   help="lower for TPU on CPU; no execution")
    p.add_argument("--iters", type=int, default=30)
    a = p.parse_args()

    rng = np.random.default_rng(0)
    lhs = jnp.asarray(rng.standard_normal((M, K)), jnp.float8_e4m3fn)
    rhs = jnp.asarray(rng.standard_normal((K, N)) * 0.05,
                      jnp.float8_e4m3fn)

    variants = {
        "A_dot_chunked": build(kernel_dot),
        "B_mrb_1mxu": build(kernel_mrb, mxus=1),
        "C_mrb_2mxu": build(kernel_mrb, mxus=2),
        "D_coexist": build(kernel_coexist),
    }

    if a.export:
        for name, fn in variants.items():
            exported = jax.export.export(
                jax.jit(fn), platforms=["tpu"])(lhs, rhs)
            print(f"{name}: lowering OK "
                  f"({len(exported.mlir_module_serialized)} bytes)")
        return

    outs = {}
    for name, fn in variants.items():
        jfn = jax.jit(fn)
        outs[name] = np.asarray(jax.block_until_ready(jfn(lhs, rhs)))
        times = []
        for _ in range(a.iters):
            t0 = time.perf_counter()
            jax.block_until_ready(jfn(lhs, rhs))
            times.append(time.perf_counter() - t0)
        print(f"{name}: min {min(times)*1e6:8.1f} us  "
              f"median {sorted(times)[len(times)//2]*1e6:8.1f} us")
    ref = outs["A_dot_chunked"]
    for name in ("B_mrb_1mxu", "C_mrb_2mxu"):
        d = np.max(np.abs(outs[name] - ref))
        print(f"max|{name} - A| = {d:.3e}")


if __name__ == "__main__":
    main()
