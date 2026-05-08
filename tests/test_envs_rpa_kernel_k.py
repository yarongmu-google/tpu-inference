# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for the RPA_KERNEL_K env-var entry in tpu_inference/envs.py.

The envs.py module is pure stdlib but uses a lazy ``__getattr__`` for env
lookups, so we re-load the module via importlib for each scenario to
ensure stable behaviour. The walrus form should:

  - return ``None`` when the env var is unset OR set to the empty string,
  - return the int value otherwise (including ``"0"``, which the
    downstream validator then rejects as K_kernel must be > 1).
"""

import importlib.util
import os
import sys
import unittest


def _load_envs_module():
    """Direct-load envs.py without going through tpu_inference/__init__.py.

    The package init transitively imports vllm/jax/torch which arent
    available on a stdlib-only dev box. envs.py itself is pure stdlib,
    so direct loading is correct."""
    spec = importlib.util.spec_from_file_location(
        "envs_under_test",
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "tpu_inference", "envs.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["envs_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


class TestRpaKernelKEnvVar(unittest.TestCase):

    def setUp(self):
        # Save and restore RPA_KERNEL_K around each test.
        self._saved = os.environ.pop("RPA_KERNEL_K", None)

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("RPA_KERNEL_K", None)
        else:
            os.environ["RPA_KERNEL_K"] = self._saved

    def test_unset_returns_none(self):
        mod = _load_envs_module()
        self.assertIsNone(mod.RPA_KERNEL_K)

    def test_empty_string_returns_none(self):
        os.environ["RPA_KERNEL_K"] = ""
        mod = _load_envs_module()
        self.assertIsNone(mod.RPA_KERNEL_K)

    def test_zero_returns_zero_int(self):
        # "0" must NOT collapse to None — it returns 0, which the
        # downstream validator rejects with a clear ValueError.
        # Documents the contract; protects against regression to a
        # truthy-string predicate.
        os.environ["RPA_KERNEL_K"] = "0"
        mod = _load_envs_module()
        self.assertEqual(mod.RPA_KERNEL_K, 0)
        self.assertIsInstance(mod.RPA_KERNEL_K, int)

    def test_typical_int_returns_int(self):
        os.environ["RPA_KERNEL_K"] = "256"
        mod = _load_envs_module()
        self.assertEqual(mod.RPA_KERNEL_K, 256)

    def test_large_int(self):
        os.environ["RPA_KERNEL_K"] = "32768"
        mod = _load_envs_module()
        self.assertEqual(mod.RPA_KERNEL_K, 32768)

    def test_negative_int_passes_through(self):
        # The env-var parser does not validate signs — that responsibility
        # is on the consumer (evaluate_decoupled_k_config rejects K<=1).
        # Lock in the parser contract so downstream tests can assume
        # the value reached the validator unchanged.
        os.environ["RPA_KERNEL_K"] = "-5"
        mod = _load_envs_module()
        self.assertEqual(mod.RPA_KERNEL_K, -5)


if __name__ == "__main__":
    unittest.main()
