# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for tools.kernel.tuner.v1.cache_key_utils.

Two surfaces under test:
  - should_skip_commit_cache_check: env var truthiness gate.
  - tuning_key_hash: stable JSON hash that strips code_revision when
    the skip is enabled. Must normalize identically for the load path
    (registry entries) and the lookup path (newly generated keys);
    otherwise the skip flag silently has no effect.

Lives outside the rpa_v3_kernel_tuner module deliberately so it has
no JAX/TPU import dependency and can run on any box.
"""

import unittest
from unittest.mock import patch

from tools.kernel.tuner.v1.cache_key_utils import (
    should_skip_commit_cache_check, tuning_key_hash)


class TestShouldSkipCommitCacheCheck(unittest.TestCase):

    def test_default_false_when_unset(self):
        self.assertFalse(should_skip_commit_cache_check(env={}))

    def test_true_only_for_literal_one(self):
        self.assertTrue(
            should_skip_commit_cache_check(env={"SKIP_COMMIT_CACHE_CHECK": "1"}))

    def test_false_for_zero(self):
        self.assertFalse(
            should_skip_commit_cache_check(env={"SKIP_COMMIT_CACHE_CHECK": "0"}))

    def test_false_for_truthy_strings_other_than_one(self):
        # We deliberately do NOT accept "true"/"yes"/"on" — the env-var
        # contract is the single sentinel "1", matching sweep.py.
        for val in ("true", "yes", "on", "True", "TRUE", "y", " 1 ", ""):
            with self.subTest(val=val):
                self.assertFalse(
                    should_skip_commit_cache_check(
                        env={"SKIP_COMMIT_CACHE_CHECK": val}),
                    f"expected False for {val!r}")

    def test_reads_real_environ_when_no_env_kwarg(self):
        # When called with env=None (default), reads os.environ.
        with patch.dict("os.environ", {"SKIP_COMMIT_CACHE_CHECK": "1"}, clear=False):
            self.assertTrue(should_skip_commit_cache_check())
        with patch.dict("os.environ", {"SKIP_COMMIT_CACHE_CHECK": "0"}, clear=False):
            self.assertFalse(should_skip_commit_cache_check())


class TestTuningKeyHash(unittest.TestCase):

    def _key(self, code_revision="abc123", **extras):
        # Minimal TuningKey-shaped dict. The hash function is field-name
        # agnostic except for code_revision, so this is sufficient.
        base = {
            "page_size": 128,
            "case": "decode",
            "chunk_prefill_size": 0,
            "code_revision": code_revision,
        }
        base.update(extras)
        return base

    def test_hash_deterministic_under_key_order(self):
        a = {"page_size": 128, "case": "decode", "code_revision": "x"}
        b = {"code_revision": "x", "case": "decode", "page_size": 128}
        self.assertEqual(
            tuning_key_hash(a, skip_commit_cache_check=False),
            tuning_key_hash(b, skip_commit_cache_check=False))

    def test_default_includes_code_revision(self):
        h_old = tuning_key_hash(
            self._key(code_revision="old"), skip_commit_cache_check=False)
        h_new = tuning_key_hash(
            self._key(code_revision="new"), skip_commit_cache_check=False)
        self.assertNotEqual(
            h_old, h_new,
            "code_revision MUST participate in the cache key by default — "
            "otherwise commit-bump invalidation is broken")

    def test_skip_strips_code_revision(self):
        h_old = tuning_key_hash(
            self._key(code_revision="old"), skip_commit_cache_check=True)
        h_new = tuning_key_hash(
            self._key(code_revision="new"), skip_commit_cache_check=True)
        self.assertEqual(
            h_old, h_new,
            "with skip enabled, code_revision must NOT influence the key — "
            "otherwise the escape hatch silently has no effect")

    def test_skip_does_not_strip_other_fields(self):
        h_p128 = tuning_key_hash(
            self._key(page_size=128), skip_commit_cache_check=True)
        h_p256 = tuning_key_hash(
            self._key(page_size=256), skip_commit_cache_check=True)
        self.assertNotEqual(
            h_p128, h_p256,
            "skip flag must only strip code_revision, not other fields")

    def test_skip_does_not_mutate_input(self):
        # Critical: the load path passes a dict freshly parsed from JSON,
        # but the lookup path passes dataclasses.asdict(tuning_key). Both
        # are short-lived, but a function that mutates its arg is a
        # footgun. Confirm we copy, not pop.
        d = self._key(code_revision="x")
        before = dict(d)
        tuning_key_hash(d, skip_commit_cache_check=True)
        self.assertEqual(d, before)

    def test_load_and_lookup_paths_match_when_only_commit_differs(self):
        # End-to-end invariant: an entry written under commit A is
        # considered "already tuned" when the lookup constructs the
        # same key under commit B, IFF skip is enabled. This is the
        # actual contract the tuner relies on for the escape hatch.
        loaded = tuning_key_hash(
            self._key(code_revision="commit_A"), skip_commit_cache_check=True)
        new_key = tuning_key_hash(
            self._key(code_revision="commit_B"), skip_commit_cache_check=True)
        self.assertEqual(loaded, new_key)

        # And the contrapositive: with skip OFF, the same scenario MUST
        # miss — otherwise commit-bump re-tuning would break.
        loaded_strict = tuning_key_hash(
            self._key(code_revision="commit_A"), skip_commit_cache_check=False)
        new_key_strict = tuning_key_hash(
            self._key(code_revision="commit_B"), skip_commit_cache_check=False)
        self.assertNotEqual(loaded_strict, new_key_strict)


if __name__ == "__main__":
    unittest.main()
