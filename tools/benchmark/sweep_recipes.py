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
import os
import sys
from pathlib import Path
from typing import Any


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
        "sweep_axes": {
            "MAX_NUM_BATCHED_TOKENS":       [8192, 16384, 32768, 65536, 131072, 262144, 524288, 1081344],
            "MAX_NUM_SEQS":                 [128, 256, 1000],
            # LONG_PREFILL_TOKEN_THRESHOLD is category-5 service-tuned
            # per docs/tuning_architecture.md §2: independent of the
            # kernel's mnss × kernel_K (kernel handles LPTT / kernel_K
            # iters per pallas_call). Candidates are multiples of the
            # pinned kernel_K=256; sweep.enumerate_combos enforces the
            # `LPTT % RPA_KERNEL_K == 0` filter so non-multiples are
            # dropped at enumerate time. The 256-and-up range covers
            # latency-style single-chunk prefills through Phase 5's
            # 1,081,344 mega-chunk regime.
            "LONG_PREFILL_TOKEN_THRESHOLD": [
                256, 1024, 4096, 8192, 16384, 32768, 65536,
                131072, 262144, 524288, 1081344,
            ],
        },
        "fixed": {
            # page_size winner from the all-three-flavors tune.
            "BLOCK_SIZE": 128,
            # Decoupled-K is the kernel default for this recipe: the L
            # kernel processes chunked prefills. RPA_KERNEL_K is pinned
            # from the kernel-tune winner (one L winner at K=256 today
            # for this workload — widen to a sweep axis once the tuner
            # produces winners at other K values). The auto-derive
            # fallback at sweep.py:_apply_auto_link (LPTT = mnss ×
            # RPA_KERNEL_K) now only fires when LPTT is NOT in
            # sweep_axes.
            "RPA_KERNEL_K": 256,
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
        "coupled_axes": [],
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
