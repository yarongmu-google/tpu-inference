# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Decode-regime fused MoE benchmark: v1 (EP, in-kernel a2a) vs the v2
decode kernel (TP, router-fused, in-kernel VMEM-direct all-gather).

Both variants are timed in the SAME serving context (DP attention):
each device starts with T/P tokens and ends with its own T/P output rows.
v1 achieves this with expert-sharded weights and an in-kernel a2a; v2 with
I-sharded weights, an in-kernel token all-gather, and a token-axis
reduce-scatter out.

Run on a single-host TPU VM (P chips):

    python -m tpu_inference.kernels.fused_moe.v2.bench_decode
    python -m tpu_inference.kernels.fused_moe.v2.bench_decode --tune
    python -m tpu_inference.kernels.fused_moe.v2.bench_decode \
        --variants=v1,v02 --tokens=512 --iters=30
    python -m tpu_inference.kernels.fused_moe.v2.bench_decode \
        --profile-dir=/tmp/moe_xprof --variants=v02

--tune sweeps the kernel parameters itself (staged coordinate descent:
be x capacity, then bd1c, then bd2c), measures each config,
and prints the winner as ready-to-paste flags. Capacity candidates start
at the ACTUAL max expert load of the routing (overflow rows are silently
dropped - an accuracy bug, not a speed tradeoff - so the tuner never goes
below it).

With --profile-dir, a trace is captured around a few iterations; the v2
kernels' jax.named_scope stages (moe_routing, moe_ag_*, moe_gather,
moe_gmm1, moe_act, moe_gmm2, moe_combine) lower to device trace markers
and show up as sub-kernel spans in the xprof trace viewer (capture must
include the device trace at level 10).

Default shapes are Qwen3.5-MoE decode at 8 chips: T=512 global tokens
(64/chip), D=4096, E=512, I=1024, k=10, bf16 weights. NOTE: gating is
random (near-uniform expert load); the v2 kernel drops rows beyond
--capacity per expert, so keep capacity comfortably above T*k/E.
"""

import argparse
import statistics
import time
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tpu_inference.kernels.fused_moe.v1.kernel import fused_ep_moe
from tpu_inference.kernels.fused_moe.v2.decode_kernel import \
    fused_moe_decode_tp_fused


def _snake_sorted_devices() -> list[jax.Device]:
    """Ring-ordered devices (same ordering as the v1 kernel tests) so that
    mesh neighbors are ICI neighbors. Non-TPU devices (no coords) keep
    their default order."""
    devices = jax.devices()
    if not hasattr(devices[0], "coords"):
        return devices
    return sorted(
        devices,
        key=lambda x: (
            x.coords[0],
            (-1 if x.coords[0] % 2 else 1) * x.coords[1],
        ),
    )


def _time(fn: Callable[[], jax.Array], iters: int,
          warmup: int) -> tuple[float, float]:
    """Returns (min_us, median_us) of end-to-end wall time per call."""
    for _ in range(warmup):
        jax.block_until_ready(fn())
    times_s: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        times_s.append(time.perf_counter() - t0)
    return min(times_s) * 1e6, statistics.median(times_s) * 1e6


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=512,
                        help="global token count T (decode batch)")
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--experts", type=int, default=512)
    parser.add_argument("--intermediate", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--capacity", type=int, default=32,
                        help="v2: max rows per expert (overflow drops)")
    parser.add_argument("--be", type=int, default=4,
                        help="v2: experts per grid step (weight-block size)")
    parser.add_argument("--bd1c", type=int, default=0,
                        help="v2: gmm1 D-contraction sub-tile (0 = whole D)")
    parser.add_argument("--bd2c", type=int, default=0,
                        help="v2: gmm2 D-output sub-tile (0 = whole D)")
    parser.add_argument("--gather", type=str, default="dma",
                        choices=["dma", "take"],
                        help="v2 gather impl: per-row DMAs (A) or "
                        "tpu.dynamic_gather via take_along_axis (B)")
    parser.add_argument("--tune", action="store_true",
                        help="sweep v2 kernel params, print the winner "
                        "(ignores the one-off param flags above)")
    parser.add_argument("--variants", type=str, default="v1,v02",
                        help="comma list from {v1, v02}")
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--profile-dir", type=str, default="",
                        help="capture an xprof trace of 3 iters per variant")
    parser.add_argument("--hbm-gbps", type=float, default=0.0,
                        help="if set, also print the weight-stream floor "
                        "(weight bytes/device divided by this bandwidth)")
    parser.add_argument("--interpret", action="store_true",
                        help="CPU dry-run of the bench plumbing (v2 only, "
                        "timings meaningless); needs XLA_FLAGS="
                        "--xla_force_host_platform_device_count=8")
    args = parser.parse_args()
    if args.interpret:
        args.variants = ",".join(v for v in args.variants.split(",")
                                 if v != "v1")

    t, d, e, i, k = (args.tokens, args.hidden, args.experts,
                     args.intermediate, args.top_k)
    devices = _snake_sorted_devices()
    p = len(devices)
    assert t % p == 0 and i % p == 0 and e % p == 0, (t, i, e, p)

    dtype = jnp.bfloat16
    rng = np.random.default_rng(0)
    tokens = jnp.asarray(rng.standard_normal((t, d)), dtype)
    # global gate|up on the 2I axis as [E, D, 2, I]: slicing the LAST axis
    # keeps each device's gate|up locally concatenated (v2's weight
    # contract); transposing to [E, 2, D, I] gives v1's.
    w1 = jnp.asarray(rng.standard_normal((e, d, 2, i)) * 0.02, dtype)
    w2 = jnp.asarray(rng.standard_normal((e, i, d)) * 0.02, dtype)
    # router weight in the upstream [E, D] layout; v1/v0.1 consume the
    # derived logits, v0.2 computes them in-kernel from (tokens, router_w).
    router_w = jnp.asarray(rng.standard_normal((e, d)) * 0.1, dtype)
    gating = jnp.asarray(
        np.asarray(tokens, np.float32) @ np.asarray(router_w, np.float32).T,
        jnp.float32)

    mesh_v1 = Mesh(np.array(devices).reshape(1, p), ("data", "model"))
    mesh_v2 = Mesh(np.array(devices), ("x",))

    per_dev_weight_bytes = (w1.nbytes + w2.nbytes) // p
    print(f"devices={p}  T={t} ({t // p}/chip)  D={d}  E={e}  I={i}  k={k}  "
          f"dtype={dtype.__name__}")
    print(f"weights/device = {per_dev_weight_bytes / 2**20:.0f} MiB")
    if args.hbm_gbps:
        floor_us = per_dev_weight_bytes / (args.hbm_gbps * 1e9) * 1e6
        print(f"weight-stream floor @ {args.hbm_gbps:.0f} GB/s = "
              f"{floor_us:.1f} us")

    variants: dict[str, Callable[[], jax.Array]] = {}

    if "v1" in args.variants:
        w1_v1 = jax.device_put(
            jnp.transpose(w1, (0, 2, 1, 3)),
            NamedSharding(mesh_v1, P("model", None, None, None)))
        w2_v1 = jax.device_put(w2, NamedSharding(mesh_v1, P("model")))
        tok_v1 = jax.device_put(tokens, NamedSharding(mesh_v1, P("model")))
        # v1 stages gating into an act-dtype VMEM buffer; f32 gating fails
        # its DMA lowering (src/dst element type mismatch)
        gate_v1 = jax.device_put(gating.astype(dtype),
                                 NamedSharding(mesh_v1, P("model")))

        def run_v1() -> jax.Array:
            return fused_ep_moe(
                mesh=mesh_v1,
                tokens=tok_v1,
                w1=w1_v1,
                w2=w2_v1,
                gating_output=gate_v1,
                top_k=k,
                renormalize_topk_logits=True,
            )

        variants["v1_ep_a2a"] = run_v1

    v2_selected = "v02" in args.variants
    if v2_selected:
        tok_l = jax.device_put(tokens, NamedSharding(mesh_v2, P("x", None)))
        w_r = jax.device_put(router_w, NamedSharding(mesh_v2, P(None, None)))
        w1_l = jax.device_put(
            w1, NamedSharding(mesh_v2, P(None, None, None, "x")))
        w2_l = jax.device_put(w2, NamedSharding(mesh_v2, P(None, "x", None)))

        def v2_runner(be: int, cap: int, bd1c: int | None,
                      bd2c: int | None) -> Callable[[], jax.Array]:
            def fn(tok: jax.Array, w1x: jax.Array, w2x: jax.Array,
                   r: jax.Array) -> jax.Array:
                w1_flat = w1x.reshape(w1x.shape[0], w1x.shape[1], -1)
                return fused_moe_decode_tp_fused(
                    tok,
                    w1_flat,
                    w2x,
                    r,            # router_w, replicated
                    mesh=mesh_v2,
                    axis_name="x",
                    top_k=k,
                    renormalize_topk_logits=True,
                    capacity=cap,
                    be=be,
                    bd1c=bd1c,
                    bd2c=bd2c,
                    gather_impl=args.gather,
                    interpret=args.interpret,
                )

            jitted = jax.jit(jax.shard_map(
                fn, mesh=mesh_v2,
                in_specs=(P("x", None), P(None, None, None, "x"),
                          P(None, "x", None), P(None, None)),
                out_specs=P("x", None), check_vma=False))
            return lambda: jitted(tok_l, w1_l, w2_l, w_r)

        if "v02" in args.variants:
            variants["v2_tp_inkernel_ag"] = v2_runner(
                be=args.be, cap=args.capacity,
                bd1c=args.bd1c or None, bd2c=args.bd2c or None)

    if args.tune:
        if "v1_ep_a2a" in variants:
            v1_us, _ = _time(variants["v1_ep_a2a"], iters=args.iters,
                             warmup=args.warmup)
            print(f"\n[tune] v1 baseline (its own tuned defaults): "
                  f"{v1_us:.1f} us")
        if not v2_selected:
            return
        # capacity floor from the actual routing
        top_i = jax.lax.top_k(jax.nn.softmax(gating, axis=-1), k)[1]
        max_load = int(jnp.max(jnp.bincount(top_i.reshape(-1), length=e)))
        cap0 = -(-max_load // 8) * 8
        print(f"[tune] max expert load = {max_load} -> capacity >= {cap0}")
        be_cands = [x for x in (4, 8, 16) if e % x == 0]
        cap_cands = sorted({cap0, 2 * cap0})
        bd_cands = [x for x in (512, 1024, 2048) if x < d and d % x == 0]

        for label in ("v02",):
            if label not in args.variants:
                continue
            best_us: float | None = None
            best: tuple[int, int, int | None, int | None] | None = None

            def measure(be: int, cap: int, b1: int | None,
                        b2: int | None) -> None:
                nonlocal best_us, best
                tag = f"  be={be} cap={cap} bd1c={b1 or 0} bd2c={b2 or 0}"
                try:
                    us, _ = _time(
                        v2_runner(be=be, cap=cap, bd1c=b1, bd2c=b2),
                        iters=args.iters, warmup=args.warmup)
                except Exception as ex:  # e.g. VMEM oversubscription
                    print(f"{tag}: failed ({type(ex).__name__})")
                    return
                print(f"{tag}: {us:.1f} us")
                if best_us is None or us < best_us:
                    best_us, best = us, (be, cap, b1, b2)

            print(f"\n[tune] {label} stage 1: be x capacity")
            for be in be_cands:
                for cap in cap_cands:
                    measure(be, cap, None, None)
            if best is None:
                print(f"[tune] {label}: every stage-1 config failed")
                continue
            print(f"[tune] {label} stage 2: gmm1 D-chunk (bd1c)")
            for b1 in bd_cands:
                measure(best[0], best[1], b1, best[3])
            print(f"[tune] {label} stage 3: gmm2 D-chunk (bd2c)")
            for b2 in bd_cands:
                measure(best[0], best[1], best[2], b2)
            be, cap, b1, b2 = best
            print(f"[tune] {label} WINNER: --be={be} "
                  f"--capacity={cap} --bd1c={b1 or 0} --bd2c={b2 or 0} "
                  f"-> {best_us:.1f} us")
        return

    # sanity: variants agree (loose - bf16, and v2 drops capacity overflow)
    outs = {name: np.asarray(fn(), np.float32)
            for name, fn in variants.items()}
    names = list(outs)
    for a in range(1, len(names)):
        diff = np.max(np.abs(outs[names[a]] - outs[names[0]]))
        print(f"max|{names[a]} - {names[0]}| = {diff:.3e}")

    print(f"\n{'variant':<32} {'min us':>10} {'median us':>10}")
    for name, fn in variants.items():
        min_us, med_us = _time(fn, iters=args.iters, warmup=args.warmup)
        print(f"{name:<32} {min_us:>10.1f} {med_us:>10.1f}")

    if args.profile_dir:
        for name, fn in variants.items():
            with jax.profiler.trace(f"{args.profile_dir}/{name}"):
                for _ in range(3):
                    jax.block_until_ready(fn())
        print(f"\ntraces written under {args.profile_dir}/<variant>; open "
              "in xprof/tensorboard - v2 per-stage spans are named moe_*")


if __name__ == "__main__":
    main()
