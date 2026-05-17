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
"""Per-(kernel, service) sweep recipes.

Architectural premise (per `tmp/benchmark_recipe.md`):
  - The user-facing input is the `.workload` (model + traffic + hardware).
  - Which scheduler/kernel knobs are tuned vs swept vs fixed is a
    property of the (kernel implementation, service implementation)
    pair, NOT a per-workload concern.
  - Workloads only differ in MODEL / INPUT_LEN / OUTPUT_LEN / MAX_MODEL_LEN /
    NUM_PROMPTS / hardware geometry. All of that lives in the `.workload`.

So the previous `.service` files were a leaky abstraction: they
conflated kernel/service knob taxonomy (static) with workload identity
(per-case). This module replaces them with a single source of truth
keyed on (kernel_id, service_id). The orchestrator looks up the recipe
and synthesizes a sweep spec at runtime — no hand-curated `.service`
JSON per workload.

Today there is one entry: ("rpa_v3", "vllm"). Future kernels (RPA v4,
FlashAttention) and services (SGLang) drop in as additional rows.

Phase 1 limitation: the sweep_axes ranges are hardcoded "reasonable
defaults". Phase 2 will derive bounds dynamically from workload +
hardware (e.g., MAX_NUM_SEQS upper bound from HBM budget; K range
from MAX_MODEL_LEN). The kernel-side VMEM pruning already does this
for block sizes; the service-side analogue is a follow-up.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


# ============================================================================
# GPU memory utilization estimator
# ============================================================================
#
# vllm's `--gpu-memory-utilization` is the fraction of total HBM vllm reserves
# for its own state (weights + KV cache). The remaining (1 - util) × HBM is
# headroom for everything XLA/TPU needs outside vllm's view:
#   - HLO program scratch (activations during the forward pass; SCALES with MNB)
#   - TPU runtime overhead (libtpu reservations, IPC buffers)
#   - vllm internal allocations that scale with in-flight prompt count
#
# Larger MNB ⇒ larger forward-pass activations ⇒ more HLO scratch needed ⇒
# LESS room for KV cache ⇒ lower max util. This formula captures that trade-off.
#
# Calibration: four end-to-end debug runs on 2026-05-16 with Llama 3 8B on
# v7x-1 / TP=1, all using SKIP_BUCKET_AUTOGEN=1 to isolate one bucket's
# scratch:
#     MNB     observed working util     HLO observed     overhead observed
#     262144  0.88                      9.01 GiB         2.01 GiB
#     327680  0.85                     11.26 GiB         2.49 GiB
#     393216  0.82                     13.51 GiB         2.98 GiB
#     458752  0.79                     15.76 GiB         3.48 GiB
#
# HLO scratch is LINEAR in MNB (verified within 0.1 GiB across all 4 points):
#     HLO ≈ 9.01 / 262144 × MNB GiB
#
# Overhead scales roughly LINEARLY with prompts-in-flight (= MNB / INPUT_LEN):
#     overhead ≈ 1.0 + 0.062 × (MNB / INPUT_LEN) GiB
# (0.062 × 32 prompts + 1.0 = 2.984 ≈ 2.01 observed — slightly conservative)
#
# The formula rounds util DOWN to nearest 0.01 for safety (it should never
# return a value HIGHER than the empirical floor). The sweep widens by
# +0.01 / +0.02 to recover the real working value and explore tighter fits.

def estimate_gpu_memory_utilization(
    mnb: int,
    input_len: int = 8192,
    *,
    hbm_total_gib: float = 94.75,
    hlo_per_mnb_gib: float = 9.01 / 262144,
    overhead_base_gib: float = 1.0,
    overhead_per_prompt_gib: float = 0.062,
) -> float:
    """Estimate the maximum GPU_MEMORY_UTILIZATION that fits a given MNB.

    Returns a value rounded DOWN to the nearest 0.01 — calling code should
    treat this as the safe FLOOR. To find the empirical sweet spot, sweep
    {est, est + 0.01, est + 0.02} and let the sweep pick the winner; values
    above est+0.02 typically OOM in the runs we have data for.

    Args:
      mnb: max_num_batched_tokens (tokens per kernel call).
      input_len: prompt length in tokens — needed to compute the
        in-flight prompt count for overhead scaling. Defaults to 8192
        (the canonical prefill_heavy workload).
      hbm_total_gib: total HBM per chip. Default 94.75 G (v7x-1; observed
        via vllm's tpu_worker `total_hbm_limit_gb` line).
      hlo_per_mnb_gib: empirical HLO-scratch slope (GiB per MNB token).
        Calibrated from the verified MNB=262144 run.
      overhead_base_gib: constant overhead component (libtpu runtime).
      overhead_per_prompt_gib: per-in-flight-prompt overhead growth.

    Returns:
      Estimated max util as a float rounded DOWN to 2 decimals (e.g. 0.85).

    Defaults are pinned to Llama 3 8B / v7x-1 / TP=1 because that is the
    sole configuration the formula has been calibrated against. Other
    (model, hardware, TP) combinations need their own coefficients
    re-derived from a verification sweep.
    """
    hlo_gib = hlo_per_mnb_gib * mnb
    prompts_in_flight = mnb / input_len
    overhead_gib = (
        overhead_base_gib + overhead_per_prompt_gib * prompts_in_flight
    )
    headroom_needed_gib = hlo_gib + overhead_gib
    util = 1.0 - headroom_needed_gib / hbm_total_gib
    return math.floor(util * 100) / 100


def _throughput_coupled_axes() -> list[dict[str, Any]]:
    """Build (MAX_NUM_BATCHED_TOKENS, GPU_MEMORY_UTILIZATION) pairs for
    the throughput recipe.

    For each MNB in the HBM-feasible axis (capped at 458752 per the
    single-chip-TP=1 KV+HLO budget), generate three util candidates:
    {est, est+0.01, est+0.02} where est is the formula floor. The sweep
    runs all three; the higher values may OOM and get marked failed
    (handled gracefully by sweep.py — exit 0 on partial success). The
    winner is the (MNB, util) combo with the highest req/s — found via
    the existing build_service_registry ranking.

    Total combos generated here: 5 MNB × 3 util = 15 (cross-product with
    MAX_NUM_SEQS axis in the recipe gives the final combo count).
    """
    axis_mnb = [131072, 262144, 327680, 393216, 458752]
    pairs: list[dict[str, Any]] = []
    for mnb in axis_mnb:
        est = estimate_gpu_memory_utilization(mnb, input_len=8192)
        for delta in (0.00, 0.01, 0.02):
            pairs.append({
                "MAX_NUM_BATCHED_TOKENS": mnb,
                # round() avoids 0.83 + 0.01 = 0.8400000000000001
                "GPU_MEMORY_UTILIZATION": round(est + delta, 2),
            })
    return pairs


# (kernel_id, service_id) -> recipe.
#
# `sweep_axes`     — knobs the orchestrator varies via cartesian product.
# `fixed`          — knobs the orchestrator pins for every combo.
# `timeout_seconds`— per-combo budget passed to sweep.py.
# `rank_metric`    — optional; metric the orchestrator passes as
#                    --metric to build_service_registry (default
#                    metrics.RequestThroughput).
# `rank_descending`— optional; direction (default True). False = "smaller
#                    is better" (latency).
#
# Block sizes (RPA_D/P/M_BLOCK_SIZES) are NOT in the recipe — they are
# auto-linked from `production.kernel` per (case, page_size, K) by
# sweep.enumerate_combos. The kernel tuner produces the registry; the
# recipe just declares which knobs the kernel/service-pair exposes
# at the scheduler level.
RECIPES: dict[tuple[str, str], dict[str, Any]] = {
    ("rpa_v3", "vllm"): {
        # MAX_NUM_BATCHED_TOKENS lives in coupled_axes (paired with
        # GPU_MEMORY_UTILIZATION), NOT in sweep_axes. Each MNB has a
        # DIFFERENT optimal util (larger MNB needs more HLO headroom ->
        # lower util) so cartesian-product would generate many infeasible
        # combos. See _throughput_coupled_axes() for the per-MNB util
        # bands the sweep explores.
        "sweep_axes": {
            "MAX_NUM_SEQS":                 [128, 256, 1000],
        },
        # 2026-05-16: per-MNB GPU_MEMORY_UTILIZATION sweep, generated by
        # estimate_gpu_memory_utilization(). For each MNB in
        # [131072..458752], sweep {est, est+0.01, est+0.02}. Smaller MNBs
        # have wider headroom margins so higher util is achievable;
        # larger MNBs need progressively lower util because HLO scratch
        # scales linearly with MNB. The sweep finds the winning
        # (MNB, util) per the rank_metric; values that OOM are marked
        # failed and don't affect the winner.
        # 524288+ excluded entirely: single-chip TP=1 cannot fit the
        # required KV cache (64 prompts × 1 GiB) plus HLO scratch
        # (~18 GiB) plus weights (15 GiB) in 94.75 GiB HBM at any util.
        # TP>1 is the path to >458752.
        "coupled_axes": _throughput_coupled_axes(),
        "fixed": {
            # page_size winner from the all-three-flavors tune.
            "BLOCK_SIZE": 128,
            # Decoupled-K is the kernel default for this recipe: the L
            # kernel processes chunked prefills, kernel_K is pinned by
            # the kernel-tune winner, and LONG_PREFILL_TOKEN_THRESHOLD
            # is derived (= mnss * RPA_KERNEL_K) by
            # sweep.py:_apply_auto_link and validated by the server at
            # runner init. Both LPTT and RPA_KERNEL_K therefore leave
            # sweep_axes — the kernel registry is the source of truth.
            # Today there is one L winner at K=256 for this workload;
            # widen this pin to a sweep axis once the tuner produces
            # winners at other K values.
            "RPA_KERNEL_K": 256,
            # 2026-05-16: skip vllm's default bucket auto-generation
            # (16, 32, ..., MNB). For a fixed-shape benchmark the other
            # 13+ buckets are pure compile-time waste (~30s per combo)
            # and contribute no throughput. tpu_inference reads this
            # env var and, when set, uses only the buckets listed in
            # --additional-config compilation_sizes (which run_benchmark.sh
            # auto-pins to [MAX_NUM_BATCHED_TOKENS] when this is set).
            "SKIP_BUCKET_AUTOGEN": 1,
        },
        "timeout_seconds": 1800,
        # Default rank: best throughput. Stated explicitly so a future
        # diff that flips the default in build_service_registry does
        # not silently change which row this recipe selects.
        "rank_metric": "metrics.RequestThroughput",
        "rank_descending": True,
    },
    # Single-prompt / low-concurrency LATENCY recipe. MIXED-only routing
    # (no static-K PREFILL kernel) — the .workload pairs that consume
    # this recipe are *_latency.workload with NUM_PROMPTS=1.
    #
    # Differences from ("rpa_v3", "vllm"):
    #   - LONG_PREFILL_TOKEN_THRESHOLD pinned to 0 (MIXED-only).
    #   - MAX_NUM_SEQS pinned to 1 (single in-flight sequence).
    #   - Rank by Mean TTFT, ascending (smaller wins).
    #   - Per-combo timeout halved (600s) — single-prompt benches
    #     finish in seconds, not minutes; 600s catches a hung server.
    #
    # MAX_NUM_BATCHED_TOKENS axis covers the chunking lever: with
    # vLLM V1 chunked-prefill on (default), MNB < INPUT_LEN forces the
    # prefill to span multiple scheduler steps -> multiple MIXED kernel
    # calls. The axis must include values BELOW the workload's
    # INPUT_LEN to actually exercise chunking; values >= INPUT_LEN
    # collapse to a single MIXED call.
    #
    # For the canonical workloads this recipe consumes:
    #   8B  / INPUT_LEN=8191 : chunking happens at MNB <= 8191, so 2048
    #                          and 4096 are the two interesting points.
    #   32B / INPUT_LEN=8191 : same as 8B.
    #   70B / INPUT_LEN=32767: 2048->16 chunks, 4096->8, 8192->4,
    #                          16384->2, 32768/65536->1 (single call).
    # The axis below covers both: 2048 and 4096 chunk all three;
    # 8192/16384 chunk only 70B; 32768/65536 are one-shot for all.
    ("rpa_v3", "vllm_latency"): {
        "sweep_axes": {
            "MAX_NUM_BATCHED_TOKENS": [
                2048, 4096, 8192, 16384, 32768, 65536,
            ],
        },
        "fixed": {
            "BLOCK_SIZE": 128,
            "MAX_NUM_SEQS": 1,
            "LONG_PREFILL_TOKEN_THRESHOLD": 0,
        },
        "timeout_seconds": 600,
        "rank_metric": "metrics.MeanTTFT",
        "rank_descending": False,
    },
    # Single-prompt LATENCY recipe with L (decoupled-K) routing.
    # Differences from ("rpa_v3", "vllm_latency"):
    #   - RPA_KERNEL_K=256 pinned: engages the L kernel.
    #   - LPTT NOT in fixed; auto-derived (= mnss * kernel_K)
    #     by sweep.py:_apply_auto_link, validated server-side.
    #   - MAX_NUM_BATCHED_TOKENS floor raised to 8192 (the 8191-token
    #     prefill must fit in one scheduler step -> one L pallas_call;
    #     anything smaller chunks across steps and defeats L's premise).
    ("rpa_v3", "vllm_latency_decoupled_k"): {
        "sweep_axes": {
            "MAX_NUM_BATCHED_TOKENS": [8192, 16384, 32768],
        },
        "fixed": {
            "BLOCK_SIZE": 128,
            "MAX_NUM_SEQS": 1,
            "RPA_KERNEL_K": 256,
            # The latency tune produced mnss=33 (the only valid value at
            # MNS=1 / MNB=8192 / K=256). production.kernel still carries
            # older LOGICAL entries with mnss=4224 (Phase 1 throughput
            # tune); sweep.py:_apply_auto_link does first-match-by-
            # (page, K) and would otherwise grab the wrong mnss. Pin
            # here until auto-link learns to filter by code_revision.
            "RPA_MAX_NUM_SUBSEQS": 33,
        },
        "timeout_seconds": 600,
        "rank_metric": "metrics.MeanTTFT",
        "rank_descending": False,
    },
}


def synthesize_service_spec(
    workload_path: str,
    kernel_id: str = "rpa_v3",
    service_id: str = "vllm",
) -> dict[str, Any]:
    """Build an in-memory sweep spec from (workload, recipe).

    Result is functionally equivalent to a hand-curated `.service` JSON.
    Caller writes it to a file and passes that to sweep.sh — sweep.py
    handles it with no change.

    The synthesized spec references:
      - `case_file`: the workload path (absolute, so relative-resolution
        against any spec-dir is a no-op).
      - `kernel_registry`: `<workload_dir>/production.kernel` by
        convention. Layer 1 (kernel tuning) writes there; Layer 2
        (sweep) reads from there via auto-link.
    """
    key = (kernel_id, service_id)
    if key not in RECIPES:
        raise ValueError(
            f"No sweep recipe for kernel={kernel_id!r}, "
            f"service={service_id!r}. Known recipes: "
            f"{sorted(RECIPES.keys())}")
    recipe = RECIPES[key]

    workload_path = str(Path(workload_path).resolve())
    workload_dir = os.path.dirname(workload_path)
    # Use removesuffix (3.9+) rather than .replace(".workload", "") —
    # the latter strips embedded occurrences too (e.g. a workload named
    # `my.workload.workload` would round-trip to `my`).
    workload_basename = os.path.basename(workload_path).removesuffix(
        ".workload")

    # sweep_name is the SECOND component of the bench result dir
    # template (run_benchmark.sh: tmp/bench_${CASE_NAME}_${TAG}, where
    # CASE_NAME = workload basename and TAG = sweep_name). Including
    # the workload basename here would double it (e.g.
    # tmp/bench_prefill_heavy_prefill_heavy_rpa_v3_vllm/). Use just
    # kernel+service as the suffix; the bench dir naturally becomes
    # tmp/bench_<workload>_<kernel>_<service>/ which reads cleanly.
    #
    # Tradeoff: per-workload log files derived from sweep_name (e.g.
    # script_build_service_registry_<sweep_name>.txt) now collide
    # across workloads that share a (kernel, service) pair. Acceptable
    # under the existing "no concurrent pipelines" constraint; the
    # log overwrites on each run.
    spec = {
        "_comment": [
            f"Synthesized from {workload_path}",
            f"via tools/benchmark/sweep_recipes.py "
            f"({kernel_id}+{service_id} recipe)",
            "Do NOT hand-edit; regenerate by re-running the orchestrator.",
            "To change the sweep axes, edit sweep_recipes.py.",
        ],
        "case_file": workload_path,
        "kernel_registry": os.path.join(workload_dir, "production.kernel"),
        "sweep_name": f"{kernel_id}_{service_id}",
        "timeout_seconds": recipe["timeout_seconds"],
        "sweep_axes": {k: list(v) for k, v in recipe["sweep_axes"].items()},
        # Recipes can include a coupled_axes list (e.g., per-MNB
        # GPU_MEMORY_UTILIZATION pairing — see _throughput_coupled_axes()).
        # Default empty for recipes that only use sweep_axes + fixed.
        "coupled_axes": [dict(d) for d in recipe.get("coupled_axes", [])],
        "fixed": dict(recipe["fixed"]),
        # Rank fields are sweep.py-irrelevant (sweep.py only enumerates
        # combos), so the validator there ignores them. The orchestrator
        # reads them back out to drive build_service_registry — same
        # source-of-truth as the rest of the recipe.
        "rank_metric": recipe.get("rank_metric", "metrics.RequestThroughput"),
        "rank_descending": recipe.get("rank_descending", True),
    }
    return spec


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Synthesize a sweep .service spec from a workload + recipe.")
    p.add_argument("--workload", required=True, type=str,
                   help="Path to the .workload file.")
    p.add_argument("--kernel", default="rpa_v3",
                   help="Kernel identifier (default: rpa_v3).")
    p.add_argument("--service", default="vllm",
                   help="Service identifier (default: vllm).")
    p.add_argument("--out", required=True, type=str,
                   help="Output path for the synthesized .service JSON.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    spec = synthesize_service_spec(
        args.workload, kernel_id=args.kernel, service_id=args.service)
    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(spec, f, indent=2)
    print(f"Synthesized: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
