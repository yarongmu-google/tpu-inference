# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for tools.tuning.v2.cli.lookup."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.tuning.v2.cli.lookup import lookup_env, main as lookup_main


def _kernel_doc_with_cases(cases: list[str]) -> dict:
    """Build a `.kernel` doc with one winner per requested case."""
    winners = []
    for case in cases:
        winners.append({
            "tuning_key": {
                "case": case,
                "page_size": 128,
                "kernel_K": 256 if case in ("prefill", "logical") else 0,
                "max_num_seqs": 128,
                "code_revision": "abc",
                "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128,
                "max_model_len": 8192,
                "q_dtype": "bfloat16", "kv_dtype": "bfloat16",
                "sliding_window": None,
            },
            "tunable_params": _block_sizes_for(case),
            "latency_us": 1000.0,
        })
    return {
        "schema_version": 1,
        "workload": "test",
        "code_revision": "abc",
        "raw_source": "abc.jsonl",
        "n_winners": len(winners),
        "winners": winners,
    }


def _block_sizes_for(case: str) -> dict:
    base = {"bq_sz": 256, "bkv_sz": 2048, "bq_csz": 256, "bkv_csz": 512}
    if case == "decode":
        base = {"bq_sz": 1, "bkv_sz": 512, "bq_csz": 1, "bkv_csz": 256}
    if case == "mixed":
        base = {"bq_sz": 128, "bkv_sz": 512, "bq_csz": 128, "bkv_csz": 256}
    if case == "logical":
        base["mnss"] = 4224
    return base


_DEFAULT_PIN_KEYS = {
    "case":          "logical",
    "page_size":     128,
    "kernel_K":      256,
    "code_revision": "abc12345",
}


def _service_doc(pin_keys: dict | None = None) -> dict:
    """Service-doc fixture. Each winner carries kernel_pin_keys —
    required by the lookup contract (post-fix-1)."""
    pk = pin_keys if pin_keys is not None else _DEFAULT_PIN_KEYS
    return {
        "schema_version": 1,
        "workload": "test",
        "service_revision": "sha",
        "raw_source": "sha.jsonl",
        "winners": {
            "throughput_max": {
                "status": "SUCCESS",
                "combo": {
                    "MAX_NUM_BATCHED_TOKENS": 131072,
                    "MAX_NUM_SEQS": 1000,
                },
                "kernel_pin_keys": pk,
                "metrics": {
                    "req_per_sec": 4.90,
                    "ttft_mean_ms": 104120,
                    "ttft_p99_ms": 203560,
                },
            },
            "ttft_min": {
                "status": "SUCCESS",
                "combo": {
                    "MAX_NUM_BATCHED_TOKENS": 8192,
                    "MAX_NUM_SEQS": 1,
                },
                "kernel_pin_keys": pk,
                "metrics": {
                    "req_per_sec": 2.57,
                    "ttft_mean_ms": 388,
                    "ttft_p99_ms": 388,
                },
            },
            "p99_min": None,
        },
    }


class TestLookupEnv(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _write_pair(self, kernel_doc: dict, service_doc: dict) -> None:
        (self.dir / "test.kernel").write_text(json.dumps(kernel_doc))
        (self.dir / "test.service").write_text(json.dumps(service_doc))

    # ---- Missing inputs ----

    def test_missing_kernel_raises(self):
        (self.dir / "test.service").write_text(json.dumps(_service_doc()))
        with self.assertRaises(FileNotFoundError):
            lookup_env(self.dir, "test")

    def test_missing_service_raises(self):
        self._write_pair(_kernel_doc_with_cases(["logical"]), {})   # placeholder
        (self.dir / "test.service").unlink()
        with self.assertRaises(FileNotFoundError):
            lookup_env(self.dir, "test")

    def test_unknown_objective_raises(self):
        self._write_pair(
            _kernel_doc_with_cases(["logical"]), _service_doc(),
        )
        with self.assertRaises(KeyError):
            lookup_env(self.dir, "test", objective="no_such_objective")

    def test_none_winner_raises(self):
        """A winner explicitly set to None in the .service doc isn't
        usable; lookup raises KeyError."""
        self._write_pair(
            _kernel_doc_with_cases(["logical"]), _service_doc(),
        )
        with self.assertRaises(KeyError):
            lookup_env(self.dir, "test", objective="p99_min")

    def test_no_logical_or_prefill_raises(self):
        """A registry with only DECODE / MIXED can't satisfy decoupled-K
        because we have no BLOCK_SIZE or RPA_KERNEL_K source."""
        self._write_pair(
            _kernel_doc_with_cases(["decode", "mixed"]), _service_doc(),
        )
        with self.assertRaises(KeyError):
            lookup_env(self.dir, "test")

    # ---- Happy paths ----

    def test_logical_winner_sets_decoupled_k_pins(self):
        self._write_pair(
            _kernel_doc_with_cases(["decode", "mixed", "logical"]),
            _service_doc(),
        )
        env = lookup_env(self.dir, "test")
        self.assertEqual(env["BLOCK_SIZE"], "128")
        self.assertEqual(env["RPA_KERNEL_K"], "256")
        self.assertEqual(env["RPA_MAX_NUM_SUBSEQS"], "4224")
        # Invariant: LPTT = mnss x kernel_K = 4224 * 256 = 1081344.
        self.assertEqual(env["LONG_PREFILL_TOKEN_THRESHOLD"], "1081344")
        self.assertEqual(env["RPA_P_BLOCK_SIZES"], "256,2048,256,512")
        self.assertEqual(env["RPA_D_BLOCK_SIZES"], "1,512,1,256")
        self.assertEqual(env["RPA_M_BLOCK_SIZES"], "128,512,128,256")
        # Service combo present.
        self.assertEqual(env["MAX_NUM_BATCHED_TOKENS"], "131072")
        self.assertEqual(env["MAX_NUM_SEQS"], "1000")

    def test_prefill_winner_without_logical_uses_coupled_k(self):
        """Legacy path: only PREFILL exists (P kernel coupled-K). Sets
        RPA_P_BLOCK_SIZES + BLOCK_SIZE but OMITS RPA_KERNEL_K (and the
        decoupled-K-derived vars). Emitting RPA_KERNEL_K=0 under
        coupled-K is ambiguous (fix #7) — consumers might interpret
        0 as "decoupled-K with K=0" instead of "decoupled-K absent".
        """
        # When the service was swept against a PREFILL kernel,
        # pin_keys reflects that (case=prefill, kernel_K=256).
        prefill_pk = {
            "case": "prefill", "page_size": 128,
            "kernel_K": 256, "code_revision": "abc",
        }
        self._write_pair(
            _kernel_doc_with_cases(["decode", "mixed", "prefill"]),
            _service_doc(pin_keys=prefill_pk),
        )
        env = lookup_env(self.dir, "test")
        self.assertEqual(env["BLOCK_SIZE"], "128")
        self.assertEqual(env["RPA_P_BLOCK_SIZES"], "256,2048,256,512")
        # Under coupled-K (no LOGICAL match), decoupled-K env vars are
        # OMITTED entirely.
        self.assertNotIn("RPA_KERNEL_K", env)
        self.assertNotIn("RPA_MAX_NUM_SUBSEQS", env)
        self.assertNotIn("LONG_PREFILL_TOKEN_THRESHOLD", env)

    def test_logical_preferred_over_prefill_when_both_present(self):
        """When both PREFILL and LOGICAL exist (transition state), the
        LOGICAL winner takes precedence — it carries the decoupled-K
        invariant (mnss × K)."""
        self._write_pair(
            _kernel_doc_with_cases(["decode", "mixed", "prefill", "logical"]),
            _service_doc(),
        )
        env = lookup_env(self.dir, "test")
        self.assertIn("RPA_MAX_NUM_SUBSEQS", env)
        self.assertIn("LONG_PREFILL_TOKEN_THRESHOLD", env)

    def test_ttft_min_objective_uses_different_combo(self):
        self._write_pair(
            _kernel_doc_with_cases(["logical"]), _service_doc(),
        )
        env = lookup_env(self.dir, "test", objective="ttft_min")
        self.assertEqual(env["MAX_NUM_BATCHED_TOKENS"], "8192")
        self.assertEqual(env["MAX_NUM_SEQS"], "1")

    def test_returned_values_are_strings(self):
        """env vars must be strings for downstream `eval` / subprocess use."""
        self._write_pair(
            _kernel_doc_with_cases(["logical"]), _service_doc(),
        )
        env = lookup_env(self.dir, "test")
        for k, v in env.items():
            self.assertIsInstance(k, str)
            self.assertIsInstance(v, str)

    def test_missing_kernel_pin_keys_raises(self):
        """fix #1: a service winner without kernel_pin_keys is a
        stale .service file (pre-fix-1) — lookup must fail loud,
        not fall back to first-match-by-case."""
        bad_service = _service_doc()
        # Strip pin_keys from the throughput_max winner.
        del bad_service["winners"]["throughput_max"]["kernel_pin_keys"]
        self._write_pair(
            _kernel_doc_with_cases(["logical"]), bad_service,
        )
        with self.assertRaisesRegex(KeyError, "kernel_pin_keys"):
            lookup_env(self.dir, "test")

    def test_pin_keys_mismatch_raises(self):
        """fix #1: if the service's pin_keys don't match any kernel
        winner, lookup must NOT silently pick the wrong kernel — it
        must raise. This is the load-bearing fix: shipping wrong pins
        because the kernel was re-tuned at a different SHA between
        sweep and deploy."""
        kernel_doc = _kernel_doc_with_cases(["logical"])
        # Build a service doc that pins to kernel_K=512 (no match in
        # the kernel registry, which only has K=256).
        mismatch_pk = {
            "case": "logical", "page_size": 128,
            "kernel_K": 512, "code_revision": "deadbeef",
        }
        self._write_pair(kernel_doc, _service_doc(pin_keys=mismatch_pk))
        with self.assertRaisesRegex(KeyError, "matching pin_keys"):
            lookup_env(self.dir, "test")

    def test_pin_keys_page_size_mismatch_raises(self):
        """Same as above but the page_size differs."""
        kernel_doc = _kernel_doc_with_cases(["logical"])
        mismatch_pk = {
            "case": "logical", "page_size": 64,
            "kernel_K": 256, "code_revision": "abc",
        }
        self._write_pair(kernel_doc, _service_doc(pin_keys=mismatch_pk))
        with self.assertRaisesRegex(KeyError, "matching pin_keys"):
            lookup_env(self.dir, "test")

    def test_decode_mixed_match_on_page_size_only(self):
        """DECODE / MIXED winners don't carry kernel_K; the pin_keys
        filter on those should match by page_size alone."""
        # Kernel doc has decode + mixed + logical at page=128, K=256.
        # Service pin_keys also at page=128, K=256.
        self._write_pair(
            _kernel_doc_with_cases(["decode", "mixed", "logical"]),
            _service_doc(),
        )
        env = lookup_env(self.dir, "test")
        self.assertEqual(env["RPA_D_BLOCK_SIZES"], "1,512,1,256")
        self.assertEqual(env["RPA_M_BLOCK_SIZES"], "128,512,128,256")

    # ---- kernel_variant dispatch (architecture doc §13.4.1) ----

    def test_pin_keys_without_kernel_variant_defaults_to_rpa_v3(self):
        """Forward-compat: older .service files predate the
        kernel_variant stamp; lookup defaults to rpa_v3 and still
        emits the RPA_* env vars."""
        pk_no_variant = {
            "case": "logical", "page_size": 128,
            "kernel_K": 256, "code_revision": "abc12345",
            # no kernel_variant
        }
        self._write_pair(
            _kernel_doc_with_cases(["logical"]),
            _service_doc(pin_keys=pk_no_variant),
        )
        env = lookup_env(self.dir, "test")
        self.assertIn("RPA_KERNEL_K", env)

    def test_pin_keys_with_known_kernel_variant_dispatches(self):
        pk = dict(_DEFAULT_PIN_KEYS)
        pk["kernel_variant"] = "rpa_v3"
        self._write_pair(
            _kernel_doc_with_cases(["logical"]),
            _service_doc(pin_keys=pk),
        )
        env = lookup_env(self.dir, "test")
        self.assertIn("RPA_KERNEL_K", env)

    def test_unknown_kernel_variant_raises(self):
        """Mismatching-is-fatal: a pin_keys.kernel_variant that the
        lookup doesn't know about means a new plugin landed without
        an emit branch. Fail loud rather than emit RPA_* env vars
        for a non-RPA kernel."""
        pk = dict(_DEFAULT_PIN_KEYS)
        pk["kernel_variant"] = "flash_attn_cuda"
        self._write_pair(
            _kernel_doc_with_cases(["logical"]),
            _service_doc(pin_keys=pk),
        )
        with self.assertRaisesRegex(KeyError, "kernel_variant"):
            lookup_env(self.dir, "test")


class TestCaseWinnerHelper(unittest.TestCase):
    """Direct tests for _case_winner — production callers always pass
    pin_keys, but the helper accepts None for ad-hoc lookups."""

    def test_pin_keys_none_returns_first_match_by_case(self):
        from tools.tuning.v2.cli.lookup import _case_winner
        doc = {
            "winners": [
                {"tuning_key": {"case": "logical",
                                "page_size": 128, "kernel_K": 256}},
                {"tuning_key": {"case": "decode",  "page_size": 128}},
            ],
        }
        # No pin_keys filter: first-by-case wins.
        w = _case_winner(doc, "logical", pin_keys=None)
        self.assertIsNotNone(w)
        self.assertEqual(w["tuning_key"]["case"], "logical")


class TestCliMain(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_workload_not_found_returns_1(self):
        with mock.patch.object(sys, "stderr", new=open(os.devnull, "w")):
            rc = lookup_main([str(self.dir / "absent.workload")])
        self.assertEqual(rc, 1)

    def test_missing_kernel_returns_1(self):
        w = self.dir / "x.workload"
        w.write_text("MODEL=foo\n")
        with mock.patch.object(sys, "stderr", new=open(os.devnull, "w")):
            rc = lookup_main([str(w)])
        self.assertEqual(rc, 1)

    def test_happy_path_returns_0_and_prints_sorted_env(self):
        w = self.dir / "x.workload"
        w.write_text("MODEL=foo\n")
        (self.dir / "x.kernel").write_text(
            json.dumps(_kernel_doc_with_cases(["logical"])),
        )
        (self.dir / "x.service").write_text(json.dumps(_service_doc()))
        captured = []
        with mock.patch("builtins.print", side_effect=captured.append):
            rc = lookup_main([str(w)])
        self.assertEqual(rc, 0)
        # Sorted output, in K=V form.
        keys = [c.split("=", 1)[0] for c in captured]
        self.assertEqual(keys, sorted(keys))
        self.assertIn("BLOCK_SIZE=128", captured)
        self.assertIn("LONG_PREFILL_TOKEN_THRESHOLD=1081344", captured)


if __name__ == "__main__":
    unittest.main()
