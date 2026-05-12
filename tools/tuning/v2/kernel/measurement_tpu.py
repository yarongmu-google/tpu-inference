# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""TPU-side `measurement_fn` adapter for the rpa_v3 kernel plugin.

Architecture doc §13.2: each kernel plugin owns an `adapter.py`
that builds a `measurement_fn(tuning_key, tunable_params) ->
result_dict` matching the v2 contract. This module is that adapter
for the existing v1 `RpaV3KernelTuner.run()` — translates field
names, forwards the call, maps the response.

Why an adapter instead of porting the v1 kernel-tune loop directly:
the v1 kernel-launch path (input generation, donated buffers, JAX
async dispatch, SMEM / VMEM estimators) is well-tested and on the
TPU's hot path. The v2 contribution is the orchestration around it
(append-only stores, resume, kernel↔service handoff, lookup); the
kernel-launch itself stays put for now.

Field-name translations (v2 ↔ v1):
  kernel_K (v2) ↔ chunk_prefill_size (v1)
  mnss     (v2) ↔ max_num_subseqs    (v1)

All other field names match (they were renamed when v2's vocabulary
diverged from v1's only in those two cases).

Import shape:
  Lazy. Importing this module DOES NOT import JAX, Pallas, or any v1
  tuner module. `make_measurement_fn()` defers those imports until
  the returned function is first called. Tests mock those imports.
"""

import os
import sys
from typing import Any, Callable


# Mapping v2 tuning_key field names -> v1 TuningKey dataclass kwargs.
# Two fields are renamed (kernel_K, mnss); the rest pass through.
_V2_TO_V1_TUNING_KEY: dict[str, str] = {
    "page_size":      "page_size",
    "q_dtype":        "q_dtype",
    "kv_dtype":       "kv_dtype",
    "num_q_heads":    "num_q_heads",
    "num_kv_heads":   "num_kv_heads",
    "head_dim":       "head_dim",
    "max_model_len":  "max_model_len",
    "sliding_window": "sliding_window",
    "case":           "case",
    "kernel_K":       "chunk_prefill_size",
    "code_revision":  "code_revision",
}

_V2_TO_V1_TUNABLE_PARAMS: dict[str, str] = {
    "bq_sz":   "bq_sz",
    "bkv_sz":  "bkv_sz",
    "bq_csz":  "bq_csz",
    "bkv_csz": "bkv_csz",
    "mnss":    "max_num_subseqs",
}


class _MinimalStorageManager:
    """Stub satisfying v1 KernelTunerBase's `assert storage_manager is
    not None` constructor check.

    The v1 tuner's `.run()` method (the only one we call) never
    touches storage_manager — only the higher-level `.tune_all()`
    driver does. v2 has its own raw_store / projection / commit-and-
    push pipeline; the v1 storage manager would be duplicated state.
    """


# OOM-signature substrings emitted by XLA and JAX. Must match all of
# them case-insensitively because XLA's TPU JIT emits
# `RESOURCE_EXHAUSTED` (uppercase + underscore) while JAX emits
# `ResourceExhausted` (Pascal case). Matching only one silently
# misclassifies the other as retryable UNKNOWN_ERROR — May-11
# incident: 45 prefill_heavy rows where the kernel JIT exceeded the
# 14 GB bottom-of-memory region were recorded as UNKNOWN_ERROR
# (retryable), so any resume would re-attempt every doomed combo.
_OOM_SIGNATURES = (
    "resource_exhausted",
    "resourceexhausted",
    "out of memory",
    "runtimeprogramallocationfailure",
)


def _classify_exception(err: BaseException, *, phase: str) -> dict[str, Any]:
    msg = str(err)
    msg_lower = msg.lower()
    if any(sig in msg_lower for sig in _OOM_SIGNATURES):
        return {
            "status": "FAILED_OOM",
            "error":  f"{phase} OOM: {type(err).__name__}: {msg}",
        }
    return {
        "status": "UNKNOWN_ERROR",
        "error":  f"{phase} raised: {type(err).__name__}: {msg}",
    }


def _translate_jax_dtype(dt: Any) -> Any:
    """Map a JSON-string dtype back to the jnp.dtype the v1 tuner
    expects. v2 stamps `"bfloat16"` as a string; v1 dataclasses hold
    `jnp.bfloat16` (a JAX dtype object)."""
    if not isinstance(dt, str):
        return dt
    # Lazy import — only fires when actually running on TPU.
    import jax.numpy as jnp
    return {
        "bfloat16":     jnp.bfloat16,
        "float16":      jnp.float16,
        "float32":      jnp.float32,
        "float8_e4m3fn": jnp.float8_e4m3fn,
    }.get(dt, dt)


def make_measurement_fn(
    *,
    iters: int = 10,
    warmup_iters: int = 2,
) -> Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]:
    """Build a v2-shaped measurement_fn driving the rpa_v3 TPU kernel.

    Args:
      iters: timed iterations per (tuning_key, tunable_params) combo.
             v2's `latency_us` is the average over these iters
             (rpa_v3 returns total_ns / iters).
      warmup_iters: untimed warmup iterations. JIT compile + cache
                    prime happens here so the timed pass is steady.
                    Set to 0 to skip (faster smoke tests; noisier).

    Returns:
      `measurement_fn(tuning_key, tunable_params) -> dict`. The
      returned function is stateful — it lazy-builds the v1 tuner
      on first call and reuses it across combos. v1 reads workload
      env vars (MAX_NUM_SEQS, MAX_NUM_BATCHED_TOKENS, ...) at
      construction time, so the caller must export those into
      `os.environ` BEFORE the first measurement_fn call. The v2 CLI
      does this by sourcing the .workload file.

    Output dict shape:
      `{"status": "SUCCESS", "latency_us": float}` on success.
      `{"status": "FAILED_OOM"}` / `{"status": "SKIPPED"}` for
      v1-known failure modes (mapped from v1's TuningStatus enum).
      `{"status": "UNKNOWN_ERROR", "error": str}` for anything else,
      including translation errors (bad field shape) and the v1
      tuner raising. The v2 runner's broad-except wraps any
      exception we let escape; we surface a typed error first.

    Cross-plugin guard: refuses any tuning_key with
    `kernel_variant != "rpa_v3"`. A future TPU kernel or GPU port
    builds its own adapter under `tools/tuning/v2/kernels/<variant>/`
    and the runner dispatches by variant (architecture doc §13.4.1).
    """
    tuner_instance: list = [None]   # mutable cell for closure

    def _ensure_tuner():
        if tuner_instance[0] is None:
            # Lazy import — pulls in JAX / Pallas / v1 tuner modules.
            from tools.kernel.tuner.v1.rpa_v3_kernel_tuner import (
                RpaV3KernelTuner,
            )
            tuner_instance[0] = RpaV3KernelTuner(
                storage_manager=_MinimalStorageManager(),
            )
        return tuner_instance[0]

    def measurement(
        tuning_key: dict[str, Any],
        tunable_params: dict[str, Any],
    ) -> dict[str, Any]:
        # MOCK_TPU=1 (mirror of MOCK_BENCH=1 in service/measurement_bench):
        # bypass the v1 tuner + JAX entirely and return synthetic
        # deterministic latency. Lets the full v2 pipeline run on a
        # non-TPU host for wiring / progress-line / file-layout
        # verification.
        # Latency varies deterministically per combo so the projection
        # has a real ordering signal — picks the "fastest" mock combo
        # consistently across runs. Not real measurements.
        if os.environ.get("MOCK_TPU") == "1":
            mnss = tunable_params.get("mnss") or 1
            bq_sz = tunable_params.get("bq_sz") or 1
            fake_latency_us = 1000.0 + 0.1 * mnss + 0.01 * bq_sz
            return {
                "status":     "SUCCESS",
                "latency_us": fake_latency_us,
                "mock":       True,
            }

        variant = tuning_key.get("kernel_variant", "rpa_v3")
        if variant != "rpa_v3":
            return {
                "status": "UNKNOWN_ERROR",
                "error":  (
                    f"measurement_tpu.py supports kernel_variant="
                    f"'rpa_v3' only; got {variant!r}. Build a sibling "
                    f"adapter for this variant under "
                    f"tools/tuning/v2/kernels/{variant}/."
                ),
            }

        try:
            from tools.kernel.tuner.v1.common.kernel_tuner_base import (
                TuningStatus,
            )
            from tools.kernel.tuner.v1.rpa_v3_kernel_tuner import (
                TunableParams,
                TuningKey,
            )
        except ImportError as e:
            return {
                "status": "UNKNOWN_ERROR",
                "error":  f"v1 tuner import failed: {e}",
            }

        try:
            tk_kwargs = {
                v1_name: (
                    _translate_jax_dtype(tuning_key[v2_name])
                    if v2_name in ("q_dtype", "kv_dtype")
                    else tuning_key[v2_name]
                )
                for v2_name, v1_name in _V2_TO_V1_TUNING_KEY.items()
                if v2_name in tuning_key
            }
            # PREFILL / DECODE / MIXED don't carry kernel_K; v1's
            # TuningKey expects chunk_prefill_size=0 for those.
            tk_kwargs.setdefault("chunk_prefill_size", 0)
            v1_tk = TuningKey(**tk_kwargs)

            tp_kwargs = {
                v1_name: tunable_params[v2_name]
                for v2_name, v1_name in _V2_TO_V1_TUNABLE_PARAMS.items()
                if v2_name in tunable_params
            }
            v1_tp = TunableParams(**tp_kwargs)
        except (KeyError, TypeError) as e:
            return {
                "status": "UNKNOWN_ERROR",
                "error":  f"translation failed: "
                          f"{type(e).__name__}: {e}",
            }

        try:
            tuner = _ensure_tuner()
        except Exception as e:    # pylint: disable=broad-except
            return {
                "status": "UNKNOWN_ERROR",
                "error":  f"tuner construction failed: "
                          f"{type(e).__name__}: {e}",
            }

        # Warmup (untimed). Surface warmup failures as UNKNOWN_ERROR
        # so the runner records them and operators can investigate;
        # don't silently drop and proceed to a meaningless measure.
        # Exception is: HBM OOMs (XLA `RESOURCE_EXHAUSTED`, JAX
        # `Out of memory`, etc.) get classified as FAILED_OOM
        # (PERMANENT_STATUSES) so resume doesn't re-attempt the
        # same doomed combo — May-11 incident: 45 UNKNOWN_ERROR rows
        # would otherwise all be retried on `--from tune_kernel`.
        # Defense-in-depth with v1's run() classification (which
        # catches the same family inside the v1 measurement loop).
        if warmup_iters > 0:
            try:
                tuner.run(v1_tk, v1_tp, iters=warmup_iters)
            except Exception as e:    # pylint: disable=broad-except
                return _classify_exception(e, phase="warmup")

        try:
            status, avg_latency_ns, _total_ns = tuner.run(
                v1_tk, v1_tp, iters=iters,
            )
        except Exception as e:    # pylint: disable=broad-except
            return _classify_exception(e, phase="measure")

        # v1 TuningStatus enum -> v2 string. Names match by design;
        # `.value` is the canonical string.
        out: dict[str, Any] = {"status": status.value}
        if status == TuningStatus.SUCCESS:
            out["latency_us"] = avg_latency_ns / 1000.0
        return out

    return measurement


def main(argv: list[str] | None = None) -> int:
    """Smoke entry. Builds the adapter, runs ONE measurement, prints
    the result. Useful for verifying TPU env is alive end-to-end
    without going through the full tune pipeline.

    Reads `tuning_key` + `tunable_params` from a tiny JSON file
    (path argv[1]) so this entry stays free of plugin-specific args.
    """
    import argparse
    import json
    p = argparse.ArgumentParser(
        description="Run one rpa_v3 measurement and print the result.",
    )
    p.add_argument("combo_json", help="Path to a JSON with keys "
                                      "tuning_key + tunable_params.")
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--warmup", type=int, default=2)
    args = p.parse_args(argv)

    from tools.tuning.v2.core.logs import configure as configure_logging
    configure_logging()

    with open(args.combo_json, "r", encoding="utf-8") as f:
        combo = json.load(f)
    measure = make_measurement_fn(
        iters=args.iters, warmup_iters=args.warmup,
    )
    result = measure(combo["tuning_key"], combo["tunable_params"])
    print(json.dumps(result, indent=2))
    return 0 if result.get("status") == "SUCCESS" else 1


if __name__ == "__main__":   # pragma: no cover
    sys.exit(main())
