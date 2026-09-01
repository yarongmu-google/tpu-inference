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

All variants are timed in the SAME serving context (DP attention):
each device starts with T/P tokens and ends with its own T/P output rows.
v1 achieves this with expert-sharded weights and an in-kernel a2a; v2 with
I-sharded weights, an in-kernel token all-gather, and a token-axis
reduce-scatter out. --variants=rs adds the experimental fused EP kernel
(kernels/experimental/fused_moe): expert-sharded weights, entry
all-gather OUTSIDE the kernel (charged in the timed jit), exit
reduce-scatter fused IN - the mirror of v2's fusion boundary.

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
from tpu_inference.kernels.fused_moe.v2.decode_kernel import (
    fused_moe_decode_tp_fused, fused_moe_decode_tp_serving)


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
                # leaf ops only: wrapper spans (jit_*, OFFLOAD_COLLECTIVE)
                # cover the whole async region incl. WAITING for the TC
                # kernel's output - occupancy, not work (measured 898us
                # of span around 28us of actual reduce-scatter)
                if (name.startswith("jit_") or name == "OFFLOAD_COLLECTIVE"
                        or name.isdigit()):
                    continue
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


def _qpc(w: jax.Array, axis: int) -> tuple[jax.Array, jax.Array]:
    """Per-channel e4m3 quantization (the serving default requant
    contract): one f32 scale per output channel over the whole
    contraction; clip at 448 (e4m3 has no Inf)."""
    wf = w.astype(jnp.float32)
    amax = jnp.max(jnp.abs(wf), axis=axis, keepdims=True)
    s = amax / 448.0
    q = jnp.clip(wf / s, -448.0, 448.0).astype(jnp.float8_e4m3fn)
    return q, s.astype(jnp.float32)


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
                                 "quant", "scales", "all"],
                        help="v2: statically stub one stage (output is "
                        "WRONG); wall-clock differences vs none are the "
                        "per-stage costs - the profiler substitute. "
                        "routing/ag stub the prologue's router and "
                        "all-gather (fixed per-call cost); 'all' stubs "
                        "every per-step stage AND the weight stream")
    parser.add_argument("--tune", action="store_true",
                        help="sweep v2 kernel params, print the winner "
                        "(ignores the one-off param flags above)")
    parser.add_argument("--wdtype", type=str, default="bf16",
                        choices=("bf16", "fp8"),
                        help="v2 weight dtype: fp8 = e4m3 weights + "
                        "per-channel f32 scales (the w8a8 path; v1 "
                        "stays bf16 - its fp8 wiring is separate)")
    parser.add_argument("--act-scale", dest="act_scale", type=str,
                        default="token", choices=("token", "tensor"),
                        help="fp8 activation-scale mode: token = "
                        "per-token dynamic (reference numerics); "
                        "tensor = one dispatch-global dynamic scale "
                        "(deletes the OHS/s_x VALU machinery)")
    parser.add_argument("--variants", type=str, default="v1,v02",
                        help="comma list from {v1, rs, v02, env} - rs is "
                        "the experimental fused EP kernel (in-kernel exit "
                        "reduce-scatter, routing + entry all-gather "
                        "outside - the mirror of v02's fusion boundary); "
                        "env adds harness-only and harness+RS rows")
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
        # v1 and rs drive real ICI writes; neither survives interpret mode
        args.variants = ",".join(v for v in args.variants.split(",")
                                 if v not in ("v1", "rs"))

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
    fp8 = args.wdtype == "fp8"

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

    if "rs" in args.variants:
        # Experimental fused EP MoE (kernels/experimental/fused_moe):
        # gather -> GMM1 -> act -> GMM2 -> in-kernel ICI reduce-scatter
        # (the SP exit). Routing and the ENTRY token all-gather stay
        # outside the kernel - the mirror of v02's fusion boundary.
        # Tokens enter token-sharded exactly like serving, so the XLA
        # all-gather this design needs is charged inside the timed jit;
        # the exit is the kernel's own RS, so no post-kernel collective.
        # Its blocks come from its internal selector - no flags apply.
        #
        # jax >= 0.11 compat shim, applied in OUR bench so his module
        # stays untouched: the kernel targets the serving env's jax
        # 0.10 pltpu API; the semaphore ops moved to pl in 0.11 and
        # pltpu's names were removed. Alias them back so the prof env
        # (matched jax/libtpu for device-time profiling) can run it.
        from jax.experimental import pallas as _pl
        from jax.experimental.pallas import tpu as _pltpu
        for _name in ("semaphore_signal", "semaphore_wait",
                      "semaphore_read"):
            if not hasattr(_pltpu, _name) and hasattr(_pl, _name):
                setattr(_pltpu, _name, getattr(_pl, _name))
        from tpu_inference.kernels.experimental.fused_moe import \
            fused_moe_func_rs

        # ShardingAxisName-compatible axes: MLP_DATA='data' (size 1),
        # expert axis resolves to 'model' (the only EP-candidate axis
        # present) -> 8-way EP, matching v1's weight arrangement.
        mesh_rs = Mesh(np.array(devices).reshape(1, p), ("data", "model"))
        tok_rs = jax.device_put(
            tokens, NamedSharding(mesh_rs, P(("data", "model"), None)))
        gate_rs = jax.device_put(
            gating, NamedSharding(mesh_rs, P(("data", "model"), None)))
        # its GMM1 wants [E, D, 2I] with [all_gate | all_up] on the last
        # axis (no TP column permute - EP shards on E), GMM2 [E, I, D]
        w1_flat = w1.reshape(e, d, 2 * i)
        ep_spec = NamedSharding(mesh_rs, P("model", None, None))
        if fp8:
            w1_rsq, w1s_rs = _qpc(w1_flat, 1)     # scale [E,1,2I]
            w2_rsq, w2s_rs = _qpc(w2, 1)          # scale [E,1,D]
            w1_rs = jax.device_put(w1_rsq, ep_spec)
            w2_rs = jax.device_put(w2_rsq, ep_spec)
            # scale dim1 == 1 -> its _recover_quant_block_size returns
            # the whole K: per-channel, same contract as the v02 row
            w1s_rs = jax.device_put(w1s_rs, ep_spec)
            w2s_rs = jax.device_put(w2s_rs, ep_spec)
        else:
            w1_rs = jax.device_put(w1_flat, ep_spec)
            w2_rs = jax.device_put(w2, ep_spec)
            w1s_rs = w2s_rs = None

        def run_rs() -> jax.Array:
            # fused_moe_func_rs is itself jitted (mesh static); output is
            # [T, D] token-sharded over ('data','model') - the same
            # serving exit contract as v02's reduce-scatter output
            return fused_moe_func_rs(
                tok_rs, w1_rs, w2_rs, w1s_rs, w2s_rs, None, None,
                gate_rs, k, True, mesh_rs, "silu", "softmax")

        variants["rs_ep_fused" + ("[fp8]" if fp8 else "")] = run_rs

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
        if fp8:
            w1_q, w1_s = _qpc(w1_perm, 1)         # [E,1,2I] scale
            w2_q, w2_s = _qpc(w2, 1)              # [E,1,D] scale
            w1_l = jax.device_put(
                w1_q, NamedSharding(mesh_v2, P(None, None, "x")))
            w2_l = jax.device_put(
                w2_q, NamedSharding(mesh_v2, P(None, "x", None)))
            w1s_l = jax.device_put(
                w1_s.reshape(e, -1),
                NamedSharding(mesh_v2, P(None, "x")))     # [E, 2I]
            w2s_l = jax.device_put(
                w2_s.reshape(e, -1),
                NamedSharding(mesh_v2, P(None, None)))    # [E, D] repl
            per_dev_weight_bytes = (w1_l.nbytes + w2_l.nbytes) // p
            print(f"v2 weights: e4m3 + per-channel scales -> "
                  f"{per_dev_weight_bytes / 2**20:.0f} MiB/device "
                  f"(+ scales "
                  f"{(w1s_l.nbytes // p + w2s_l.nbytes) / 2**20:.1f})")
        else:
            w1s_l = w2s_l = None
            w1_l = jax.device_put(
                w1_perm, NamedSharding(mesh_v2, P(None, None, "x")))
            w2_l = jax.device_put(
                w2, NamedSharding(mesh_v2, P(None, "x", None)))

        def v2_runner(be: int, cap: int, bd1c: int | None,
                      bd2c: int | None,
                      bcT: int | None = None,
                      bg: int = 1) -> Callable[[], jax.Array]:
            def fn(tok: jax.Array, w1x: jax.Array, w2x: jax.Array,
                   r: jax.Array, s1: jax.Array | None = None,
                   s2: jax.Array | None = None) -> jax.Array:
                # no reshape here: anything reshaping a weight inside
                # the jitted fn re-materializes the tensor per call
                return fused_moe_decode_tp_fused(
                    tok,
                    r,            # router weight, replicated
                    w1x,
                    w2x,
                    w1_scale=s1,
                    w2_scale=s2,
                    act_scale=args.act_scale,
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

            in_specs = [P("x", None), P(None, None, "x"),
                        P(None, "x", None), P(None, None)]
            operands = [tok_l, w1_l, w2_l, w_r]
            if fp8:
                in_specs += [P(None, "x"), P(None, None)]
                operands += [w1s_l, w2s_l]
            jitted = jax.jit(jax.shard_map(
                fn, mesh=mesh_v2,
                in_specs=tuple(in_specs),
                out_specs=P("x", None), check_vma=False))
            return lambda: jitted(*operands)

        if "v02" in args.variants:
            # An ablated row stubs one stage (output wrong on purpose) - tag
            # the name so a differential-timing row is never mistaken for a
            # real measurement of the kernel.
            v2_name = "v2_tp_inkernel_ag"
            if fp8:
                v2_name += "[fp8]"
                if args.act_scale != "token":
                    v2_name += f"[as={args.act_scale}]"
            if args.ablate != "none":
                v2_name += f"[ablate={args.ablate}]"
            variants[v2_name] = v2_runner(
                be=args.be, cap=args.capacity,
                bd1c=args.bd1c or None, bd2c=args.bd2c or None,
                bcT=args.bcT or None, bg=args.bg)

        if "v02s" in args.variants:
            # the SERVING entry (router_fused=False: precomputed gating,
            # the serving capacity formula, be=4/bg=1 defaults) - the
            # exact path vllm engages, which the v02 rows never
            # exercise. Purpose: reproduce serving-only failures at
            # serving decode shapes (sweep --tokens over the num-reqs
            # buckets) without standing up a server.
            gate_l = jax.device_put(
                gating, NamedSharding(mesh_v2, P("x", None)))
            # serving scale layout is 4D ([E,1,1,N]); reshape on the
            # HOST, never inside the jit (weight-adjacent transforms
            # in the traced path re-materialize per call)
            if fp8:
                w1s4_l = jax.device_put(
                    w1_s.reshape(e, 1, 1, -1),
                    NamedSharding(mesh_v2, P(None, None, None, "x")))
                w2s4_l = jax.device_put(
                    w2_s.reshape(e, 1, 1, -1),
                    NamedSharding(mesh_v2, P(None, None, None, None)))
            else:
                w1s4_l = w2s4_l = None
            cap_serving = min(t, max(16, -(-2 * t * k // (e * 8)) * 8))

            def v02s_fn(tok: jax.Array, gate: jax.Array,
                        w1x: jax.Array, w2x: jax.Array,
                        s1: jax.Array | None,
                        s2: jax.Array | None) -> jax.Array:
                return fused_moe_decode_tp_serving(
                    tok,
                    gate,
                    w1x,
                    w2x,
                    s1,
                    s2,
                    act_scale=args.act_scale,
                    mesh=mesh_v2,
                    axis_name="x",
                    top_k=k,
                    renormalize_topk_logits=True,
                    capacity=cap_serving,
                    interpret=args.interpret,
                )

            v02s_jit = jax.jit(v02s_fn)
            name = "v2_serving_entry" + ("[fp8]" if fp8 else "")
            variants[name] = lambda: v02s_jit(
                tok_l, gate_l, w1_l, w2_l, w1s4_l, w2s4_l)

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
        # v1/rs pick their own blocks internally - time them once as
        # fixed baselines, then sweep only v02's parameters
        for base_name, base_fn in variants.items():
            if base_name.startswith("v2_"):
                continue
            base_us, _ = _time(base_fn, iters=args.iters,
                               warmup=args.warmup)
            print(f"\n[tune] {base_name} baseline (its own tuned "
                  f"defaults): {base_us:.1f} us")
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
        # fp8 x rows pack 4/word: the kernel asserts capacity % 32 == 0
        # regardless of the act dtype - a 16-multiple candidate would
        # just die in the tuner's except and silently push the sweep
        # to 2x capacity (an MXU-LINEAR cost under fp8: gather goes
        # mm-bound and combine slot-rows scale with C).
        sublane = 32 if fp8 else 32 // jnp.dtype(dtype).itemsize
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
