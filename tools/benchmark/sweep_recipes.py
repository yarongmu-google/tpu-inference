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
#
# Block sizes (RPA_D/P/M_BLOCK_SIZES) are NOT in the recipe — they are
# auto-linked from `production.kernel` per (case, page_size, K) by
# sweep.enumerate_combos. The kernel tuner produces the registry; the
# recipe just declares which knobs the kernel/service-pair exposes
# at the scheduler level.
RECIPES: dict[tuple[str, str], dict[str, Any]] = {
    ("rpa_v3", "vllm"): {
        "sweep_axes": {
            # vllm scheduler knobs (orthogonal to kernel; swept end-to-end).
            #
            # MAX_NUM_BATCHED_TOKENS values:
            #   2048 — minimum that fits one full single-prompt prefill
            #          for the prefill_heavy workload (8K input, K=2048).
            #   4096, 8192 — power-of-two steps for headroom/throughput
            #          scaling.
            #   10275 — bm-infras (QC repo) v7x default for the standard
            #          sonnet 1024/1024 workload; included for apples-to-
            #          apples comparison against their dashboards.
            "MAX_NUM_BATCHED_TOKENS":       [2048, 4096, 8192, 10275],
            # MAX_NUM_SEQS values:
            #   128 — bm-infra v6e/v7x default; the floor.
            #   256 — middle rung; useful at long-context (32K+) where
            #          per-request KV is large enough that 1000 is not
            #          actually reachable but 128 leaves throughput on
            #          the table.
            #   1000 — saturate concurrency on v7xs HBM-backed capacity;
            #          vllm auto-caps the actual concurrency to whatever
            #          fits, so 1000 is "use as much as you can".
            "MAX_NUM_SEQS":                 [128, 256, 1000],
            # K=0 folds the baseline (chunk-prefill OFF) into the same
            # sweep — the highest-throughput row wins regardless of K.
            # Powers-of-two from 128 up to MAX_MODEL_LEN cover the K
            # values the kernel tuner already produces winners for.
            "LONG_PREFILL_TOKEN_THRESHOLD": [0, 128, 256, 512, 1024, 2048],
        },
        "fixed": {
            # page_size winner from the all-three-flavors tune. If the
            # tuner ever produces a stronger winner at a different
            # page_size, update here OR sweep BLOCK_SIZE in sweep_axes
            # (auto-link will inject D/P/M block sizes for the chosen
            # page_size from the kernel registry).
            "BLOCK_SIZE": 128,
        },
        "timeout_seconds": 1800,
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
