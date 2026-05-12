# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for tools.tuning.v2.kernel.measurement_tpu.

The adapter does ONE thing per call: translate v2 dicts into v1
dataclass kwargs, invoke v1's RpaV3KernelTuner.run(), translate the
response. We mock the v1 tuner module + JAX so these tests run on
any host. The TPU hardware path is exercised by item (m), not here.
"""

import sys
import types
import unittest
from unittest import mock


def _install_v1_stubs() -> dict[str, mock.Mock]:
    """Install a fake v1 tuner module tree into sys.modules and
    return the mocks for assertion. Patched modules are cleaned up
    in each test's tearDown."""
    # TuningStatus enum-like with .SUCCESS / .FAILED_OOM / etc, each
    # holding a .value matching the v2 status string.
    class _Status:
        def __init__(self, name): self.value = name
        def __eq__(self, other): return self.value == other.value
        def __hash__(self): return hash(self.value)
    SUCCESS = _Status("SUCCESS")
    FAILED_OOM = _Status("FAILED_OOM")
    SKIPPED = _Status("SKIPPED")
    UNKNOWN_ERROR = _Status("UNKNOWN_ERROR")

    base_mod = types.ModuleType(
        "tools.kernel.tuner.v1.common.kernel_tuner_base",
    )
    base_mod.TuningStatus = types.SimpleNamespace(
        SUCCESS=SUCCESS, FAILED_OOM=FAILED_OOM,
        SKIPPED=SKIPPED, UNKNOWN_ERROR=UNKNOWN_ERROR,
    )

    # TuningKey / TunableParams: accept arbitrary kwargs, store as
    # attributes. The adapter only constructs them — never inspects
    # the result — so simple data holders are enough.
    class _DataHolder:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    tuner_mod = types.ModuleType(
        "tools.kernel.tuner.v1.rpa_v3_kernel_tuner",
    )
    tuner_mod.TuningKey = _DataHolder
    tuner_mod.TunableParams = _DataHolder

    # The tuner instance. .run() is a Mock so tests can program its
    # return per case.
    run_mock = mock.Mock(return_value=(SUCCESS, 2391_000.0, 23910_000.0))
    tuner_instance = mock.Mock()
    tuner_instance.run = run_mock
    tuner_mod.RpaV3KernelTuner = mock.Mock(return_value=tuner_instance)

    sys.modules["tools.kernel.tuner.v1.common.kernel_tuner_base"] = base_mod
    sys.modules["tools.kernel.tuner.v1.rpa_v3_kernel_tuner"] = tuner_mod
    return {
        "run_mock":         run_mock,
        "tuner_class":      tuner_mod.RpaV3KernelTuner,
        "tuner_instance":   tuner_instance,
        "TuningStatus":     base_mod.TuningStatus,
    }


def _v2_combo(case: str = "logical") -> tuple[dict, dict]:
    """Standard v2 tuning_key / tunable_params for the rpa_v3 path."""
    return (
        {
            "kernel_variant": "rpa_v3",
            "hardware":       "tpu_v7x",
            "schema_version": 1,
            "case":           case,
            "page_size":      128,
            "kernel_K":       256,
            "max_num_seqs":   128,
            "code_revision":  "abc12345",
            "num_q_heads":    32,
            "num_kv_heads":   8,
            "head_dim":       128,
            "max_model_len":  8192,
            "q_dtype":        "bfloat16",
            "kv_dtype":       "bfloat16",
            "sliding_window": None,
        },
        {
            "bq_sz":   256, "bkv_sz": 2048,
            "bq_csz":  256, "bkv_csz": 512,
            "mnss":    4224,
        },
    )


class TestMeasurementAdapter(unittest.TestCase):

    def setUp(self):
        # Save and replace the two v1 modules.
        self._saved = {
            k: sys.modules.get(k) for k in (
                "tools.kernel.tuner.v1.common.kernel_tuner_base",
                "tools.kernel.tuner.v1.rpa_v3_kernel_tuner",
            )
        }
        self.stubs = _install_v1_stubs()

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    def _make_fn(self, **kw):
        from tools.tuning.v2.kernel.measurement_tpu import (
            make_measurement_fn,
        )
        return make_measurement_fn(**kw)

    # ---- happy paths ----

    def test_success_returns_latency_us_in_microseconds(self):
        """v1 returns latency_ns; v2 contract is latency_us. The
        adapter divides by 1000."""
        # 2391000 ns -> 2391.0 us.
        tk, tp = _v2_combo()
        fn = self._make_fn(iters=1, warmup_iters=0)
        result = fn(tk, tp)
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["latency_us"], 2391.0)

    def test_field_translation_kernel_K_to_chunk_prefill_size(self):
        """v2's kernel_K is v1's chunk_prefill_size. The adapter must
        rename, not pass through verbatim."""
        tk, tp = _v2_combo()
        fn = self._make_fn(iters=1, warmup_iters=0)
        fn(tk, tp)
        v1_tk, _v1_tp = self.stubs["run_mock"].call_args[0][:2]
        self.assertEqual(v1_tk.chunk_prefill_size, 256)
        self.assertFalse(hasattr(v1_tk, "kernel_K"))

    def test_field_translation_mnss_to_max_num_subseqs(self):
        tk, tp = _v2_combo()
        fn = self._make_fn(iters=1, warmup_iters=0)
        fn(tk, tp)
        _v1_tk, v1_tp = self.stubs["run_mock"].call_args[0][:2]
        self.assertEqual(v1_tp.max_num_subseqs, 4224)
        self.assertFalse(hasattr(v1_tp, "mnss"))

    def test_non_logical_case_defaults_chunk_prefill_size_to_zero(self):
        """PREFILL / DECODE / MIXED don't carry kernel_K in v2;
        v1's TuningKey expects chunk_prefill_size=0 for those cases."""
        tk, tp = _v2_combo(case="decode")
        del tk["kernel_K"]
        fn = self._make_fn(iters=1, warmup_iters=0)
        fn(tk, tp)
        v1_tk, _ = self.stubs["run_mock"].call_args[0][:2]
        self.assertEqual(v1_tk.chunk_prefill_size, 0)

    def test_iters_arg_threaded_to_v1_run(self):
        tk, tp = _v2_combo()
        fn = self._make_fn(iters=42, warmup_iters=0)
        fn(tk, tp)
        # iters is the 3rd positional or named kwarg.
        kwargs = self.stubs["run_mock"].call_args.kwargs
        self.assertEqual(kwargs.get("iters"), 42)

    def test_warmup_runs_first_then_measure(self):
        tk, tp = _v2_combo()
        fn = self._make_fn(iters=10, warmup_iters=3)
        fn(tk, tp)
        # Two calls: warmup (iters=3), then measure (iters=10).
        self.assertEqual(self.stubs["run_mock"].call_count, 2)
        first_iters = self.stubs["run_mock"].call_args_list[0].kwargs["iters"]
        second_iters = self.stubs["run_mock"].call_args_list[1].kwargs["iters"]
        self.assertEqual(first_iters, 3)
        self.assertEqual(second_iters, 10)

    def test_warmup_zero_skips_warmup_call(self):
        tk, tp = _v2_combo()
        fn = self._make_fn(iters=10, warmup_iters=0)
        fn(tk, tp)
        # Only one call: the timed measure.
        self.assertEqual(self.stubs["run_mock"].call_count, 1)

    def test_tuner_constructed_once_across_combos(self):
        """The v1 tuner reads env vars in __init__ (slow). The adapter
        must reuse one instance across calls."""
        tk, tp = _v2_combo()
        fn = self._make_fn(iters=1, warmup_iters=0)
        fn(tk, tp)
        fn(tk, tp)
        fn(tk, tp)
        self.assertEqual(self.stubs["tuner_class"].call_count, 1)

    # ---- v1 status mapping ----

    def test_failed_oom_propagates_without_latency_field(self):
        """OOM is permanent; the adapter omits latency_us so the
        projection step doesn't try to compare it numerically."""
        self.stubs["run_mock"].return_value = (
            self.stubs["TuningStatus"].FAILED_OOM,
            float("inf"), float("inf"),
        )
        tk, tp = _v2_combo()
        fn = self._make_fn(iters=1, warmup_iters=0)
        result = fn(tk, tp)
        self.assertEqual(result["status"], "FAILED_OOM")
        self.assertNotIn("latency_us", result)

    def test_skipped_propagates(self):
        """v1 SKIPPED (e.g. SMEM/VMEM estimator above limit) maps to
        v2's SKIPPED, which is in PERMANENT_STATUSES."""
        self.stubs["run_mock"].return_value = (
            self.stubs["TuningStatus"].SKIPPED,
            float("inf"), float("inf"),
        )
        tk, tp = _v2_combo()
        fn = self._make_fn(iters=1, warmup_iters=0)
        result = fn(tk, tp)
        self.assertEqual(result["status"], "SKIPPED")
        self.assertNotIn("latency_us", result)

    # ---- error paths ----

    def test_mock_tpu_env_bypasses_v1_tuner(self):
        """MOCK_TPU=1 short-circuits before any v1 import. Lets the
        full v2 pipeline run on a non-TPU host for wiring tests."""
        # Even if the v1 tuner is broken, MOCK_TPU bypasses it.
        self.stubs["tuner_class"].side_effect = RuntimeError(
            "v1 should not be touched under MOCK_TPU",
        )
        tk, tp = _v2_combo()
        fn = self._make_fn(iters=1, warmup_iters=0)
        import os as _os
        with mock.patch.dict(_os.environ, {"MOCK_TPU": "1"}):
            result = fn(tk, tp)
        self.assertEqual(result["status"], "SUCCESS")
        self.assertTrue(result["mock"])
        self.assertIn("latency_us", result)
        # v1 tuner constructor was NOT called.
        self.stubs["tuner_class"].assert_not_called()

    def test_mock_tpu_latency_varies_per_combo_for_projection(self):
        """Synthetic latency depends on (mnss, bq_sz) so different
        combos in a MOCK_TPU sweep get a real ordering, not all
        tied at one value. Projection's argmin still works."""
        import os as _os
        fn = self._make_fn(iters=1, warmup_iters=0)
        with mock.patch.dict(_os.environ, {"MOCK_TPU": "1"}):
            tk, tp = _v2_combo()
            tp_a = dict(tp); tp_a["mnss"] = 100
            tp_b = dict(tp); tp_b["mnss"] = 9999
            la = fn(tk, tp_a)["latency_us"]
            lb = fn(tk, tp_b)["latency_us"]
        self.assertLess(la, lb)

    def test_mock_tpu_env_non_one_does_not_bypass(self):
        """Only literal '1' enables mock — same contract as the
        other SMOKE_TEST / NO_PUSH / NO_COMMIT / MOCK_BENCH knobs."""
        tk, tp = _v2_combo()
        fn = self._make_fn(iters=1, warmup_iters=0)
        import os as _os
        with mock.patch.dict(_os.environ, {"MOCK_TPU": "true"}):
            fn(tk, tp)
        # v1 tuner WAS constructed (mock didn't trip).
        self.stubs["tuner_class"].assert_called_once()

    def test_unknown_kernel_variant_refuses(self):
        """Cross-plugin guard (architecture doc §13.4.1)."""
        tk, tp = _v2_combo()
        tk["kernel_variant"] = "flash_attn_cuda"
        fn = self._make_fn(iters=1, warmup_iters=0)
        result = fn(tk, tp)
        self.assertEqual(result["status"], "UNKNOWN_ERROR")
        self.assertIn("flash_attn_cuda", result["error"])

    def test_measure_exception_caught_as_unknown_error(self):
        self.stubs["run_mock"].side_effect = RuntimeError("device gone")
        tk, tp = _v2_combo()
        fn = self._make_fn(iters=1, warmup_iters=0)
        result = fn(tk, tp)
        self.assertEqual(result["status"], "UNKNOWN_ERROR")
        self.assertIn("RuntimeError", result["error"])
        self.assertIn("device gone", result["error"])

    def test_warmup_exception_caught_as_unknown_error(self):
        """A failing warmup must NOT cascade into the timed measure —
        the adapter records UNKNOWN_ERROR and stops."""
        self.stubs["run_mock"].side_effect = RuntimeError("compile fail")
        tk, tp = _v2_combo()
        fn = self._make_fn(iters=10, warmup_iters=1)
        result = fn(tk, tp)
        self.assertEqual(result["status"], "UNKNOWN_ERROR")
        self.assertIn("warmup", result["error"])
        # Only the warmup call fired; no timed measure attempted.
        self.assertEqual(self.stubs["run_mock"].call_count, 1)

    def test_tuner_construction_failure_caught(self):
        self.stubs["tuner_class"].side_effect = ValueError(
            "MAX_NUM_SEQS env var not set",
        )
        tk, tp = _v2_combo()
        fn = self._make_fn(iters=1, warmup_iters=0)
        result = fn(tk, tp)
        self.assertEqual(result["status"], "UNKNOWN_ERROR")
        self.assertIn("tuner construction failed", result["error"])

    def test_v1_import_failure_caught(self):
        """Defensive: if the v1 tuner modules are missing (e.g. v1
        was retired or this is a stripped-down install), the adapter
        returns UNKNOWN_ERROR with a typed message instead of
        crashing the whole tune loop."""
        # Replace the stub with a module that raises on attribute
        # access (simulating a partial install).
        del sys.modules["tools.kernel.tuner.v1.rpa_v3_kernel_tuner"]
        # Provide a meta_path finder that raises ImportError for the
        # v1 module — emulates the package not being installed.
        class _BlockFinder:
            def find_module(self, name, path=None):
                if name == "tools.kernel.tuner.v1.rpa_v3_kernel_tuner":
                    return self
                return None
            def load_module(self, name):
                raise ImportError(f"no module {name}")
            def find_spec(self, name, path, target=None):
                if name == "tools.kernel.tuner.v1.rpa_v3_kernel_tuner":
                    raise ImportError(f"no module {name}")
                return None
        finder = _BlockFinder()
        sys.meta_path.insert(0, finder)
        try:
            from tools.tuning.v2.kernel.measurement_tpu import (
                make_measurement_fn,
            )
            tk, tp = _v2_combo()
            fn = make_measurement_fn(iters=1, warmup_iters=0)
            result = fn(tk, tp)
        finally:
            sys.meta_path.remove(finder)
        self.assertEqual(result["status"], "UNKNOWN_ERROR")
        self.assertIn("v1 tuner import failed", result["error"])

    def test_translation_failure_caught(self):
        """A tuning_key missing required fields shouldn't crash —
        coerced to UNKNOWN_ERROR with a typed message."""
        # Construct a TuningKey class that requires `page_size`.
        from tools.tuning.v2.kernel.measurement_tpu import (
            make_measurement_fn,
        )
        v1_mod = sys.modules["tools.kernel.tuner.v1.rpa_v3_kernel_tuner"]

        class StrictTK:
            def __init__(self, *, page_size, **kw):
                self.page_size = page_size
                self.__dict__.update(kw)
        v1_mod.TuningKey = StrictTK

        tk, tp = _v2_combo()
        del tk["page_size"]
        fn = make_measurement_fn(iters=1, warmup_iters=0)
        result = fn(tk, tp)
        self.assertEqual(result["status"], "UNKNOWN_ERROR")
        self.assertIn("translation failed", result["error"])


class TestSmokeMain(unittest.TestCase):
    """Cover the standalone `measurement_tpu.main` smoke entry."""

    def setUp(self):
        self._saved = {
            k: sys.modules.get(k) for k in (
                "tools.kernel.tuner.v1.common.kernel_tuner_base",
                "tools.kernel.tuner.v1.rpa_v3_kernel_tuner",
            )
        }
        _install_v1_stubs()
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        from pathlib import Path
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        self._tmp.cleanup()

    def _write_combo(self):
        import json
        tk, tp = _v2_combo()
        path = self.dir / "combo.json"
        path.write_text(json.dumps({
            "tuning_key": tk, "tunable_params": tp,
        }))
        return path

    def test_main_prints_success_and_returns_0(self):
        from tools.tuning.v2.kernel.measurement_tpu import main
        path = self._write_combo()
        with mock.patch("sys.stdout", new=mock.MagicMock()) as out:
            rc = main([str(path), "--iters", "1", "--warmup", "0"])
        self.assertEqual(rc, 0)
        # First print call carried the JSON result.
        printed = "".join(
            c.args[0] for c in out.print.mock_calls
            if c.args
        ) if hasattr(out, "print") else ""
        # The print() builtin writes via stdout.write; just assert
        # write was called at all.
        self.assertTrue(out.write.called)

    def test_main_returns_1_when_measurement_fails(self):
        # Force the underlying run to raise -> UNKNOWN_ERROR -> rc=1.
        v1_mod = sys.modules["tools.kernel.tuner.v1.rpa_v3_kernel_tuner"]
        v1_mod.RpaV3KernelTuner.return_value.run.side_effect = RuntimeError(
            "boom",
        )
        from tools.tuning.v2.kernel.measurement_tpu import main
        path = self._write_combo()
        with mock.patch("sys.stdout", new=mock.MagicMock()):
            rc = main([str(path), "--iters", "1", "--warmup", "0"])
        self.assertEqual(rc, 1)


class TestDtypeTranslation(unittest.TestCase):
    """`q_dtype` / `kv_dtype` cross the JSON boundary as strings;
    v1 expects jnp.bfloat16 etc. The adapter swaps via a small map."""

    def test_passes_jax_dtype_through_for_non_string(self):
        """Already-translated values (in case a caller passes them
        directly in tests) pass through unchanged."""
        from tools.tuning.v2.kernel.measurement_tpu import (
            _translate_jax_dtype,
        )
        sentinel = object()
        self.assertIs(_translate_jax_dtype(sentinel), sentinel)


if __name__ == "__main__":
    unittest.main()
