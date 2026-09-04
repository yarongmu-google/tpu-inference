# fp4 (e2m1) -> fp8 (e4m3) unpack-rate probe.
#
# Measures how fast packed 4-bit float weights can be converted to
# e4m3 on-device, which bounds any fp4-stored / fp8-compute matmul
# pipeline. Three questions:
#   1. dtype support: do jax/ml_dtypes float4 types exist and lower
#      on this TPU at all?
#   2. rate: elements/sec for large-buffer conversion, vs the HBM
#      stream rate of the packed bytes (if convert >= stream rate,
#      unpack can hide under the weight fetch; if slower, it is the
#      bound).
#   3. codegen: with a mosaic dump, does the conversion lower to a
#      native convert op or to an emulation sequence?
#
# Run (serving env):
#   python tmp/probe_fp4_unpack.py 2>&1 | tee tmp/fp4_unpack_probe.log
# Optional codegen dump:
#   rm -rf tmp/mosaic_fp4 && mkdir -p tmp/mosaic_fp4
#   LIBTPU_INIT_ARGS=--xla_mosaic_dump_to=tmp/mosaic_fp4 \
#     python tmp/probe_fp4_unpack.py --pallas-only
# Then: git add tmp/ && git commit -m "fp4 unpack probe." && git push

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np


def bench(fn, *args, iters=30):
    fn_j = jax.jit(fn)
    out = jax.block_until_ready(fn_j(*args))
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(fn_j(*args))
        ts.append(time.perf_counter() - t0)
    return min(ts), out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--elems", type=int, default=1 << 28,
                   help="elements to convert (default 268M)")
    p.add_argument("--pallas-only", action="store_true")
    a = p.parse_args()

    print("jax", jax.__version__, "devices", jax.devices()[:1])

    # 1. dtype support inventory
    have = {}
    for name in ("float4_e2m1fn", "float8_e4m3fn", "int4", "uint4"):
        have[name] = hasattr(jnp, name)
        print(f"jnp.{name}: {'YES' if have[name] else 'NO'}")
    if not have.get("float4_e2m1fn"):
        try:
            import ml_dtypes
            have["float4_e2m1fn"] = hasattr(ml_dtypes, "float4_e2m1fn")
            print(f"ml_dtypes.float4_e2m1fn: {have['float4_e2m1fn']}")
        except ImportError:
            print("ml_dtypes not importable")

    n = a.elems
    rng = np.random.default_rng(0)

    results = []

    # 2a. native path, if the dtype exists end to end
    if have.get("float4_e2m1fn"):
        try:
            f4 = jnp.asarray(
                rng.standard_normal(n).astype(np.float32)
            ).astype(jnp.float4_e2m1fn)
            f4 = jax.block_until_ready(f4)
            t, _ = bench(lambda x: x.astype(jnp.float8_e4m3fn), f4)
            results.append(("native e2m1->e4m3 astype", t))
        except Exception as e:
            print("native path failed:", type(e).__name__, str(e)[:200])

    # 2b. emulation path: packed uint8 (2 nibbles) -> two e4m3 outputs
    # via integer bit surgery. e2m1 bits s|ee|m -> e4m3 bits s|eeee|mmm:
    # sign << 4, exponent rebias (+4 on the 2-bit field), mantissa << 2.
    packed = jnp.asarray(rng.integers(0, 256, n // 2, dtype=np.uint8))

    def emulate(pk):
        lo = pk & 0xF
        hi = pk >> 4

        def cvt(nib):
            sign = (nib & 0x8) << 4
            mag = nib & 0x7
            # subnormal (exp==0): value = m * 0.5 -> handled by the
            # same affine bit map here for RATE purposes only (this
            # probe measures throughput, not bit-exactness).
            out = sign | jnp.where(mag > 0, ((mag + 0x18) << 2) & 0x7F, 0)
            return out.astype(jnp.uint8)

        return cvt(lo), cvt(hi)

    t, _ = bench(emulate, packed)
    results.append(("uint8 nibble bit-surgery x2", t))

    # 2c. reference stream rate: plain copy of the packed bytes
    t, _ = bench(lambda x: x + jnp.uint8(0), packed)
    results.append(("packed-byte stream (add 0)", t))

    print(f"\n{'variant':38} {'ms':>9} {'Gelem/s':>9} {'GB/s(out)':>10}")
    for name, t in results:
        print(f"{name:38} {t*1e3:9.2f} {n/t/1e9:9.1f} {n/t/1e9:10.1f}")
    print("\nverdict rule: unpack Gelem/s must exceed the HBM packed "
          "stream rate (~1.9 TB/s = ~3800 Gelem/s at 4 bits) for the "
          "convert to hide under the weight fetch; the gap, if any, "
          "is the pipeline-depth requirement.")


if __name__ == "__main__":
    main()
