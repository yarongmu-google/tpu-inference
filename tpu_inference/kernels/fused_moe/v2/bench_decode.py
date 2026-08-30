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
import gzip
import json
import statistics
import time
from pathlib import Path
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


def _iter_trace_events(trace_dir: str):
    for path in sorted(Path(trace_dir).rglob("*.trace.json.gz")):
        with gzip.open(path, "rt") as trace_file:
            yield json.load(trace_file).get("traceEvents", [])


def _device_pids(events) -> tuple:
    """(tc_pid_to_dev, sc_pid_to_dev) maps from the trace metadata -
    pid -> device index parsed from '/device:TPU:<n>'."""
    tc, sc = {}, {}
    for event in events:
        if event.get("ph") == "M" and event.get("name") == "process_name":
            process_name = str(event.get("args", {}).get("name", ""))
            marker = "/device:TPU:"
            if marker in process_name:
                dev = process_name.split(marker, 1)[1].split(" ")[0]
                dev = int("".join(ch for ch in dev if ch.isdigit()) or -1)
                (sc if "SparseCore" in process_name else tc)[
                    event.get("pid")] = dev
    return tc, sc


def _interval_coverage_us(intervals, lo, hi) -> float:
    """Union length of [start,end) intervals clipped to [lo, hi) -
    busy WALL time, immune to parallel-lane overcounting."""
    clipped = sorted((max(a, lo), min(b, hi))
                     for a, b in intervals if b > lo and a < hi)
    total, cur_a, cur_b = 0.0, None, None
    for a, b in clipped:
        if cur_b is None or a > cur_b:
            if cur_b is not None:
                total += cur_b - cur_a
            cur_a, cur_b = a, b
        else:
            cur_b = max(cur_b, b)
    if cur_b is not None:
        total += cur_b - cur_a
    return total


def _device_kernel_ms_per_dispatch_from_trace(
    trace_dir: str,
    *,
    jit_name_prefix: str,
) -> list:
    """Device latency of each matching top-level JIT dispatch (ported
    from bench_stacked_rpa_golden: TensorCore pids only, barrier and
    trailing-copy edges subtracted)."""
    matching_by_pid: dict = {}
    ops_by_pid: dict = {}
    sc_ops_by_dev: dict = {}
    tc_dev: dict = {}
    for events in _iter_trace_events(trace_dir):
        tc_pids, sc_pids = _device_pids(events)
        tc_dev.update(tc_pids)
        for event in events:
            pid = event.get("pid")
            if event.get("ph") != "X" or not event.get("dur"):
                continue
            name = str(event.get("name", ""))
            start_us = float(event["ts"])
            duration_us = float(event["dur"])
            end_us = start_us + duration_us
            if pid in sc_pids:
                sc_ops_by_dev.setdefault(sc_pids[pid], []).append(
                    (start_us, end_us))
                continue
            if pid not in tc_pids:
                continue
            if name.startswith(jit_name_prefix):
                matching_by_pid.setdefault(pid, []).append(
                    (start_us, end_us, duration_us))
            elif not name.startswith("jit_"):
                ops_by_pid.setdefault(pid, []).append(
                    (start_us, end_us, duration_us, name))
    if not matching_by_pid:
        return []
    pid, dispatches = max(matching_by_pid.items(),
                          key=lambda item: len(item[1]))
    ops = ops_by_pid.get(pid, [])
    # SC events of the SAME device only, as interval COVERAGE within the
    # dispatch window: summing durations across parallel subcore lanes
    # (or all 8 devices) overcounts absurdly - 24ms/dispatch measured.
    sc_intervals = sc_ops_by_dev.get(tc_dev.get(pid, -1), [])
    rows = []
    for start_us, end_us, duration_us in sorted(dispatches):
        children = [op for op in ops if start_us <= op[0] < end_us]
        barrier_us = sum(op[2] for op in children
                         if op[3] == "barrier-cores")
        copy_us = 0.0
        if children:
            last = max(children, key=lambda op: op[1])
            if last[3].startswith("copy"):
                copy_us = last[2]
        sc_us = _interval_coverage_us(sc_intervals, start_us, end_us)
        rows.append((max(duration_us - barrier_us - copy_us, 0.0) / 1000.0,
                     sc_us / 1000.0))
    return rows


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
    parser.add_argument("--bg", type=int, default=1,
                        help="v2: grid steps per dispatch group; amortizes "
                        "the accumulator RMW and gather fills without "
                        "growing the weight windows")
    parser.add_argument("--bd1c", type=int, default=0,
                        help="v2: gmm1 D-contraction sub-tile (0 = whole D)")
    parser.add_argument("--bd2c", type=int, default=0,
                        help="v2: gmm2 D-output sub-tile (0 = whole D)")
    parser.add_argument("--bct", "--bcT", dest="bcT", type=int, default=0,
                        help="v2: combine accumulate chunk over T "
                        "(0 = one full-width expression, unpipelineable)")
    parser.add_argument("--vmem-mb", dest="vmem_mb", type=int, default=64,
                        help="v2: VMEM budget handed to the kernel; larger "
                        "values admit larger be (fewer grid steps, less "
                        "accumulator traffic) until the backend rejects it")
    parser.add_argument("--ablate", type=str, default="none",
                        choices=["none", "masks", "gather", "ffn",
                                 "combine", "weights", "routing", "ag",
                                 "all"],
                        help="v2: statically stub one stage (output is "
                        "WRONG); wall-clock differences vs none are the "
                        "per-stage costs - the profiler substitute. "
                        "routing/ag stub the prologue's router and "
                        "all-gather (fixed per-call cost); 'all' stubs "
                        "every per-step stage AND the weight stream")
    parser.add_argument("--tune", action="store_true",
                        help="sweep v2 kernel params, print the winner "
                        "(ignores the one-off param flags above)")
    parser.add_argument("--variants", type=str, default="v1,v02",
                        help="comma list from {v1, v02, env} - env "
                        "adds harness-only and harness+RS rows")
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

    # env rows need the mesh + token shards but not the kernel
    v2_selected = "v02" in args.variants or "env" in args.variants
    if v2_selected:
        tok_l = jax.device_put(tokens, NamedSharding(mesh_v2, P("x", None)))
        w_r = jax.device_put(router_w, NamedSharding(mesh_v2, P(None, None)))
        # Pre-flatten OUTSIDE jit, with columns permuted so device k's
        # contiguous 2I/P-column shard is its own (gate | up) pair.
        # The old in-shard reshape ([E,D,2,I/P] -> [E,D,2I/P]) forced
        # XLA to materialize a layout-conversion COPY of the WHOLE w1
        # EVERY call - measured as the "fixed floor": ~480us + weight
        # bytes at ~660 GB/s (2.3ms at E=512), confirmed by the HLO
        # (%copy.1 bf16[512,4096,2,1024]) and by ablate=all scaling
        # linearly in E (775/1074/1634/2821us at E=64/128/256/512).
        w1_perm = w1.reshape(e, d, 2, p, i // p).transpose(
            0, 1, 3, 2, 4).reshape(e, d, 2 * i)
        w1_l = jax.device_put(
            w1_perm, NamedSharding(mesh_v2, P(None, None, "x")))
        w2_l = jax.device_put(w2, NamedSharding(mesh_v2, P(None, "x", None)))

        def v2_runner(be: int, cap: int, bd1c: int | None,
                      bd2c: int | None,
                      bcT: int | None = None,
                      bg: int = 1) -> Callable[[], jax.Array]:
            def fn(tok: jax.Array, w1x: jax.Array, w2x: jax.Array,
                   r: jax.Array) -> jax.Array:
                # no reshape here: anything reshaping a weight inside
                # the jitted fn re-materializes the tensor per call
                return fused_moe_decode_tp_fused(
                    tok,
                    r,            # router weight, replicated
                    w1x,
                    w2x,
                    mesh=mesh_v2,
                    axis_name="x",
                    top_k=k,
                    renormalize_topk_logits=True,
                    capacity=cap,
                    be=be,
                    bg=bg,
                    bd1c=bd1c,
                    bd2c=bd2c,
                    bcT=bcT,
                    vmem_limit_bytes=args.vmem_mb * 2**20,
                    ablate=args.ablate,
                    interpret=args.interpret,
                )

            jitted = jax.jit(jax.shard_map(
                fn, mesh=mesh_v2,
                in_specs=(P("x", None), P(None, None, "x"),
                          P(None, "x", None), P(None, None)),
                out_specs=P("x", None), check_vma=False))
            return lambda: jitted(tok_l, w1_l, w2_l, w_r)

        if "v02" in args.variants:
            # An ablated row stubs one stage (output wrong on purpose) - tag
            # the name so a differential-timing row is never mistaken for a
            # real measurement of the kernel.
            v2_name = "v2_tp_inkernel_ag"
            if args.ablate != "none":
                v2_name += f"[ablate={args.ablate}]"
            variants[v2_name] = v2_runner(
                be=args.be, cap=args.capacity,
                bd1c=args.bd1c or None, bd2c=args.bd2c or None,
                bcT=args.bcT or None, bg=args.bg)

        if "env" in args.variants:
            # envelope decomposition: what does the measurement itself
            # cost with ZERO kernel? dispatch = jit + shard_map + 8-dev
            # launch + host sync; envelope adds the exit reduce-scatter.
            # (envelope - dispatch) = RS share; (ablate rows - envelope)
            # = true in-kernel time.
            def disp_fn(tok: jax.Array) -> jax.Array:
                return tok

            def env_fn(tok: jax.Array) -> jax.Array:
                full = jnp.tile(tok, (p, 1))         # [T, D] partials
                return jax.lax.psum_scatter(
                    full, "x", scatter_dimension=0, tiled=True)

            disp_jit = jax.jit(jax.shard_map(
                disp_fn, mesh=mesh_v2, in_specs=P("x", None),
                out_specs=P("x", None), check_vma=False))
            env_jit = jax.jit(jax.shard_map(
                env_fn, mesh=mesh_v2, in_specs=P("x", None),
                out_specs=P("x", None), check_vma=False))
            variants["v2_dispatch_only"] = lambda: disp_jit(tok_l)
            variants["v2_envelope_rs"] = lambda: env_jit(tok_l)

    if args.tune:
        if "v1_ep_a2a" in variants:
            v1_us, _ = _time(variants["v1_ep_a2a"], iters=args.iters,
                             warmup=args.warmup)
            print(f"\n[tune] v1 baseline (its own tuned defaults): "
                  f"{v1_us:.1f} us")
        if not v2_selected:
            return
        # capacity floor from the actual routing, rounded to the ACT
        # dtype's sublane tile (bf16: 16) - not f32's 8: x/y hold
        # capacity-row expert blocks in act dtype, so a capacity off the
        # act tile leaves every block's rows sublane-misaligned (relayout
        # glue on each FFN read/write), and a 16-multiple also makes
        # be*capacity whole lane tiles for the ohg lane dim (be=4 -> 128).
        top_i = jax.lax.top_k(jax.nn.softmax(gating, axis=-1), k)[1]
        max_load = int(jnp.max(jnp.bincount(top_i.reshape(-1), length=e)))
        sublane = 32 // jnp.dtype(dtype).itemsize
        cap0 = -(-max_load // sublane) * sublane
        print(f"[tune] max expert load = {max_load} -> capacity >= {cap0}")
        # be scales the double-buffered weight windows (2*be experts of
        # VMEM; be=8 is already past the 64 MiB physical VMEM). bg groups
        # grid steps under ONE dispatch/combine, amortizing the per-call
        # costs (accumulator RMW, gather fills) at only x/y/ohg scratch
        # cost - the budget assert rejects what does not fit.
        be_cands = [x for x in (2, 4, 8) if e % x == 0]
        bg_cands = [1, 2, 4, 8]
        cap_cands = sorted({cap0, 2 * cap0})
        # small chunks matter most: an unchunked contraction emits all the
        # loads/pushes before any matmul, so nothing interleaves. Include
        # 128/256 (one and two lane-tiles) - the MXU's diagonal push does
        # two 128x128 blocks per pass, so K=128 is not a half-empty array.
        bd_cands = [x for x in (128, 256, 512, 1024, 2048)
                    if x < d and d % x == 0]

        for label in ("v02",):
            if label not in args.variants:
                continue
            best_us: float | None = None
            best: tuple[int, int, int | None, int | None,
                        int | None, int] | None = None
            measured: list = []   # (wall_us, config) for finalist re-rank

            def measure(be: int, cap: int, b1: int | None,
                        b2: int | None, bt: int | None = None,
                        bg: int = 1) -> None:
                nonlocal best_us, best
                tag = (f"  be={be} bg={bg} cap={cap} bd1c={b1 or 0} "
                       f"bd2c={b2 or 0} bcT={bt or 0}")
                try:
                    us, _ = _time(
                        v2_runner(be=be, cap=cap, bd1c=b1, bd2c=b2, bcT=bt,
                                  bg=bg),
                        iters=args.iters, warmup=args.warmup)
                except Exception as ex:  # e.g. VMEM oversubscription
                    msg = str(ex).splitlines()[0][:140] if str(ex) else ""
                    print(f"{tag}: failed ({type(ex).__name__}: {msg})")
                    return
                print(f"{tag}: {us:.1f} us")
                measured.append((us, (be, cap, b1, b2, bt, bg)))
                if best_us is None or us < best_us:
                    best_us, best = us, (be, cap, b1, b2, bt, bg)

            print(f"\n[tune] {label} stage 1: be x bg x capacity")
            for be in be_cands:
                for bg in bg_cands:
                    for cap in cap_cands:
                        measure(be, cap, None, None, bg=bg)
            if best is None:
                print(f"[tune] {label}: every stage-1 config failed")
                continue
            print(f"[tune] {label} stage 2: gmm1 D-chunk (bd1c)")
            for b1 in bd_cands:
                measure(best[0], best[1], b1, best[3], best[4], best[5])
            print(f"[tune] {label} stage 3: gmm2 D-chunk (bd2c)")
            for b2 in bd_cands:
                measure(best[0], best[1], best[2], b2, best[4], best[5])
            print(f"[tune] {label} stage 4: combine T-chunk (bcT)")
            for bt in [x for x in (8, 16, 32, 64, 128, 256) if x < t]:
                measure(best[0], best[1], best[2], best[3], bt, best[5])
            # Finalist re-rank on DEVICE time: wall-min orders configs
            # correctly only down to the envelope jitter (~30-40us),
            # while stage sweeps often differ by 10-30us; device-only
            # profiling is sub-us stable. Profile the top finalists and
            # let the device median pick. Requires the matched
            # jax/libtpu pair (the prof env); on the mismatched pair
            # the profiler SEGFAULTS, so gate on the known-good jax.
            finalists = []
            seen = set()
            for us, cfg in sorted(measured):
                if cfg not in seen:
                    finalists.append((us, cfg))
                    seen.add(cfg)
                if len(finalists) == 5:
                    break
            jax_ok = tuple(
                int(x) for x in jax.__version__.split(".")[:2]) >= (0, 11)
            if args.profile_dir and jax_ok and len(finalists) > 1:
                print(f"[tune] {label} finalist re-rank on device time:")
                best_dev = None
                for us, cfg in finalists:
                    fbe, fcap, fb1, fb2, fbt, fbg = cfg
                    trace_dir = (f"{args.profile_dir}/tune_be{fbe}_bg{fbg}"
                                 f"_c{fcap}_{fb1 or 0}_{fb2 or 0}_{fbt or 0}")
                    fn = v2_runner(be=fbe, cap=fcap, bd1c=fb1, bd2c=fb2,
                                   bcT=fbt, bg=fbg)
                    jax.block_until_ready(fn())
                    opts = jax.profiler.ProfileOptions()
                    opts.python_tracer_level = 0
                    opts.device_tracer_level = 2
                    jax.profiler.start_trace(trace_dir,
                                             profiler_options=opts)
                    for _ in range(5):
                        jax.block_until_ready(fn())
                    jax.profiler.stop_trace()
                    rows = _device_kernel_ms_per_dispatch_from_trace(
                        trace_dir, jit_name_prefix="jit_")
                    if not rows:
                        print(f"  {cfg}: no trace rows")
                        continue
                    tc = sorted(r[0] for r in rows)
                    dev_us = tc[len(tc) // 2] * 1e3
                    print(f"  {cfg}: wall {us:.1f} -> device {dev_us:.1f} us")
                    if best_dev is None or dev_us < best_dev[0]:
                        best_dev = (dev_us, cfg)
                if best_dev is not None:
                    best_us, best = best_dev
            elif args.profile_dir and not jax_ok:
                print("[tune] finalist re-rank skipped: profiler needs "
                      f"jax>=0.11 (matched libtpu); this env has "
                      f"{jax.__version__}")
            be, cap, b1, b2, bt, bg = best
            print(f"[tune] {label} WINNER: --be={be} --bg={bg} "
                  f"--capacity={cap} --bd1c={b1 or 0} --bd2c={b2 or 0} "
                  f"--bcT={bt or 0} -> {best_us:.1f} us")
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
        # the stacked_rpa bench's proven recipe (bench_stacked_rpa_golden):
        # python tracer OFF, device tracer level 2 - the default
        # jax.profiler.trace() path segfaulted in ProfilerSession::Create
        # on this stack. The parser then reports DEVICE-ONLY time per
        # dispatch from TensorCore pids: the kernel-only number the bench
        # wall-clock (harness + RS + dispatch) cannot give.
        for name, fn in variants.items():
            trace_dir = f"{args.profile_dir}/{name}"
            opts = jax.profiler.ProfileOptions()
            opts.python_tracer_level = 0
            opts.device_tracer_level = 2
            jax.profiler.start_trace(trace_dir, profiler_options=opts)
            for _ in range(5):
                jax.block_until_ready(fn())
            jax.profiler.stop_trace()
            rows = _device_kernel_ms_per_dispatch_from_trace(
                trace_dir, jit_name_prefix="jit_")
            if rows:
                tc = sorted(r[0] for r in rows)
                sc = sorted(r[1] for r in rows)
                print(f"{name:<32} device-only per dispatch: "
                      f"TC min {tc[0]*1e3:8.1f} us  "
                      f"median {tc[len(tc)//2]*1e3:8.1f} us  "
                      f"| SC median {sc[len(sc)//2]*1e3:6.1f} us  "
                      f"({len(rows)} dispatches)")
            else:
                print(f"{name:<32} no matching device dispatches in trace")
        print(f"\ntraces under {args.profile_dir}/<variant>; open in "
              "xprof/tensorboard - v2 per-stage spans are named moe_*")


if __name__ == "__main__":
    main()
