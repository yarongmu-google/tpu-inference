"""fp8 matmul LOWERING probes - the Mosaic dump is the result, not the
timing. (qwen35_fp8 proposal sec 8.2a; kernel_plan.md verify items.)

Each case is a tiny named pallas kernel at a v2-fp8 kernel shape. Run
under a Mosaic dump (tmp/run_fp8_probes.sh sets LIBTPU_INIT_ARGS) and
histogram the per-kernel dumps. What decides what:

  gmm1_w8a8    dot(e4m3 [32,4096], e4m3 [4096,256]) -> f32
               EXPECT vmatpush.e4m3 + fp8-mode vmatmul.
  gmm1_w8a16   dot(bf16 [32,4096], e4m3 [4096,256]) -> f32
               The ISA forbids mixed bf16 x f8 at the DATAPATH but the
               MSR->GMR latch converts fp8->bf16 losslessly for free
               (isa:9259-9267). EXPECT vmatpush.e4m3 + bf16-mode
               vmatmul and NO vcvt storm. A mass of vcvt.* (or a
               convert materializing a bf16 weight buffer) means
               Mosaic does NOT use the free latch -> the w8a16 arm
               gets a real dequant cost and the proposal's "LHS dtype
               irrelevant under fill" claim needs re-pricing.
  gmm2_w8a16   dot(bf16 [32,128], e4m3 [128,4096]) -> f32
               K=128: also shows whether the K<=128 case lowers to
               vmatpush.diag (curiosity, no perf claim).
  gather_fp8   dot(e4m3 [128,512], e4m3 [512,4096]) -> f32
               The dispatch gather shape; fp8-mode expected.
  scale_mult   gmm1_w8a8 + acc * s_row[1,256] * s_col[32,1]
               The per-channel scale application; watch for how the
               [C,1] lane-broadcast lowers (XLU bcast vs copies).

Also prints wall time per case (sanity only - shapes are tiny) and
checks outputs are finite. Provenance printed per lesson D.24.

    python tmp/probe_fp8_lowering.py       # on the TPU VM
"""

import functools
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

F8 = jnp.float8_e4m3fn


def _dot_kernel(a_ref, b_ref, o_ref):
    o_ref[...] = jnp.dot(a_ref[...], b_ref[...],
                         preferred_element_type=jnp.float32)


def _scale_kernel(a_ref, b_ref, sr_ref, sc_ref, o_ref):
    acc = jnp.dot(a_ref[...], b_ref[...],
                  preferred_element_type=jnp.float32)
    acc = acc * jnp.broadcast_to(sr_ref[...], acc.shape)   # [1,N] rows
    acc = acc * jnp.broadcast_to(sc_ref[...], acc.shape)   # [M,1] lanes
    o_ref[...] = acc


def _build(name, kernel, in_shapes_dtypes, out_shape):
    return pl.pallas_call(
        kernel,
        in_specs=[pl.BlockSpec(memory_space=pltpu.MemorySpace.VMEM)
                  for _ in in_shapes_dtypes],
        out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.VMEM),
        out_shape=jax.ShapeDtypeStruct(out_shape, jnp.float32),
        name=name,
    )


CASES = [
    ("gmm1_w8a8", _dot_kernel,
     [((32, 4096), F8), ((4096, 256), F8)], (32, 256)),
    ("gmm1_w8a16", _dot_kernel,
     [((32, 4096), jnp.bfloat16), ((4096, 256), F8)], (32, 256)),
    ("gmm2_w8a16", _dot_kernel,
     [((32, 128), jnp.bfloat16), ((128, 4096), F8)], (32, 4096)),
    ("gather_fp8", _dot_kernel,
     [((128, 512), F8), ((512, 4096), F8)], (128, 4096)),
    ("scale_mult", _scale_kernel,
     [((32, 4096), F8), ((4096, 256), F8),
      ((1, 256), jnp.float32), ((32, 1), jnp.float32)], (32, 256)),
]


def main() -> None:
    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    print(f"probe_fp8_lowering: commit {rev}, jax {jax.__version__}, "
          f"device {jax.devices()[0].device_kind}")
    rng = np.random.default_rng(0)
    for name, kernel, ins, out_shape in CASES:
        args = [jnp.asarray(rng.standard_normal(s), d) for s, d in ins]
        try:
            fn = jax.jit(_build(name, kernel, ins, out_shape))
            out = jax.block_until_ready(fn(*args))
            t0 = time.perf_counter()
            for _ in range(10):
                out = fn(*args)
            jax.block_until_ready(out)
            t = (time.perf_counter() - t0) / 10
            finite = bool(jnp.isfinite(out).all())
            print(f"{name:>12}: ok  {t * 1e6:7.1f} us/call  "
                  f"finite={finite}")
        except Exception as ex:
            # A COMPILE error on gmm*_w8a16 is itself the answer:
            # no free-latch path -> w8a16 needs an explicit dequant.
            print(f"{name:>12}: FAILED ({type(ex).__name__}: "
                  f"{str(ex).splitlines()[0][:100]})")


if __name__ == "__main__":
    main()
