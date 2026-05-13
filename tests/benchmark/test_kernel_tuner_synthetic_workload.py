# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for the kv_len cap in `get_mixed_example` + validate-fail
SKIPPED reclassification.

These functions live in `tools.kernel.tuner.v1.rpa_v3_kernel_tuner`
which imports JAX/vllm at module top — not importable on a dev
laptop. We sys.modules-stub the heavy deps before importing so the
pure-Python functions become unit-testable.
"""

import sys
import unittest
from types import SimpleNamespace, ModuleType
from unittest.mock import MagicMock


def _stub_heavy_modules() -> None:
    """Pre-populate sys.modules with minimal stand-ins for the
    JAX/vllm/tpu_inference imports rpa_v3_kernel_tuner.py performs
    at module load. Only the symbols we touch (jax.numpy types,
    jnp.array, jax.errors.JaxRuntimeError, the kernel module's
    `dynamic_validate_inputs` / `get_*_bytes` / `ragged_paged_attention`)
    are wired — everything else is a Mock that accepts attribute access.
    """
    fake_modules = {
        "jax": MagicMock(),
        "jax.numpy": MagicMock(),
        "jax.errors": MagicMock(),
        "jax.errors.JaxRuntimeError": type("JaxRuntimeError", (Exception,), {}),
        "jax_tpu_embedding": MagicMock(),
        "vllm": MagicMock(),
        "vllm.logger": MagicMock(),
        "tpu_inference": MagicMock(),
        "tpu_inference.tpu_info": MagicMock(),
        "tpu_inference.logger": MagicMock(),
        "tpu_inference.kernels": MagicMock(),
        "tpu_inference.kernels.ragged_paged_attention": MagicMock(),
        "tpu_inference.kernels.ragged_paged_attention.v3": MagicMock(),
        "tpu_inference.kernels.ragged_paged_attention.v3.kernel": MagicMock(),
    }
    for name, mod in fake_modules.items():
        if name not in sys.modules:
            sys.modules[name] = mod


_stub_heavy_modules()

# Now safe to import the v1 tuner's pure functions.
from tools.kernel.tuner.v1.rpa_v3_kernel_tuner import (
    get_mixed_example,
)


class TestGetMixedExampleKvLenCap(unittest.TestCase):
    """Regression: at MNB ≫ MML, each prefill seq's q_len exceeds
    max_model_len, and kv_len = q_len overflows the model context.
    Triggered the May-12 dynamic_validate_inputs crash
    (kv_len=154,478 vs MML=8192, page_cnt=2414 > pages_per_seq=128).
    After the fix: kv_len is clamped to min(q_len, max_model_len)
    so the synthetic input is always shape-valid regardless of how
    big the caller's max_num_tokens is."""

    def test_kv_len_capped_at_mml_when_q_exceeds(self):
        """May-12 incident exact shape: actual_num_seqs=8,
        max_num_tokens=1,081,344, max_model_len=8192. Per-seq q_len
        ≈ 154k each. After fix, each kv_len ≤ 8192."""
        _, kv_lens, _, _ = get_mixed_example(
            actual_num_seqs=8,
            max_num_tokens=1_081_344,
            max_model_len=8192,
        )
        for kv in kv_lens:
            self.assertLessEqual(
                kv, 8192,
                f"kv_len={kv} exceeds max_model_len=8192 "
                f"— the cap regressed",
            )

    def test_decode_seq_kv_len_is_full_mml(self):
        """The decode-mode seq (q_len=1) sets kv_len=max_model_len
        — represents a fully-prefilled seq now in decode phase.
        That's a model-level invariant; the cap must NOT clobber it
        to 1. Different code path: `max_model_len if q_len == 1 else
        ...`, so q_len=1 goes through the first branch."""
        _, kv_lens, decode_end, _ = get_mixed_example(
            actual_num_seqs=2,
            max_num_tokens=1024,
            max_model_len=4096,
        )
        # Seq 0 is the decode seq (q_len=1), kv_len should be full MML.
        self.assertEqual(decode_end, 1)
        self.assertEqual(kv_lens[0], 4096)

    def test_kv_len_unchanged_when_q_fits_in_mml(self):
        """Normal regime — MNB ≤ MML × num_seqs. q_len per seq fits
        in MML, kv_len = q_len (no cap needed)."""
        _, kv_lens, decode_end, _ = get_mixed_example(
            actual_num_seqs=4,
            max_num_tokens=8000,
            max_model_len=8192,
        )
        # 3 prefill seqs share (8000-1) tokens = ~2666 each. < MML.
        for kv in kv_lens[decode_end:]:
            self.assertLessEqual(kv, 8192)
            # Confirm we're NOT capping; the q_len values are well
            # under MML.
            self.assertLess(kv, 3000)

    def test_actual_num_seqs_eq_1_treats_all_tokens_as_one_seq(self):
        """Single-seq path: cu_q_lens = [0, max_num_tokens]. The kv_len
        for that one seq is q_len=max_num_tokens, capped at MML."""
        _, kv_lens, decode_end, _ = get_mixed_example(
            actual_num_seqs=1,
            max_num_tokens=20_000,
            max_model_len=8192,
        )
        # q_len = 20000, but cap kicks in.
        self.assertEqual(len(kv_lens), 1)
        self.assertLessEqual(kv_lens[0], 8192)


if __name__ == "__main__":
    unittest.main()
