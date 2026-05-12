# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for tools.tuning.v2.core.keyset."""

import unittest

from tools.tuning.v2.core.keyset import (
    canonical_json,
    combo_key,
    service_combo_key,
)


class TestCanonicalJson(unittest.TestCase):

    def test_sorts_keys(self):
        self.assertEqual(canonical_json({"b": 2, "a": 1}), '{"a":1,"b":2}')

    def test_compact_no_whitespace(self):
        self.assertEqual(
            canonical_json({"k": "v", "n": 1}),
            '{"k":"v","n":1}',
        )

    def test_handles_nested_dicts(self):
        self.assertEqual(
            canonical_json({"o": {"y": 2, "x": 1}, "a": 1}),
            '{"a":1,"o":{"x":1,"y":2}}',
        )

    def test_handles_lists(self):
        self.assertEqual(canonical_json([1, 2, 3]), "[1,2,3]")

    def test_handles_scalars(self):
        self.assertEqual(canonical_json("string"), '"string"')
        self.assertEqual(canonical_json(42), "42")
        self.assertEqual(canonical_json(True), "true")
        self.assertEqual(canonical_json(None), "null")

    def test_unicode_passthrough(self):
        """ensure_ascii=False so non-ASCII chars stay readable on disk."""
        self.assertEqual(canonical_json({"n": "café"}), '{"n":"café"}')

    def test_stable_across_calls(self):
        d = {"b": 2, "a": 1, "c": {"y": 1, "x": 2}}
        self.assertEqual(canonical_json(d), canonical_json(dict(d)))

    def test_falls_back_to_str_for_unknown_types(self):
        """default=str handles non-JSON-serialisable types like sets."""
        # Sets get str()-coerced to their repr — not great for stable
        # keys, but the helper shouldn't crash.
        out = canonical_json({"s": {1, 2, 3}})
        # Just confirm it didn't raise; the exact string is set-order-
        # dependent in some Pythons.
        self.assertIn("s", out)


class TestComboKey(unittest.TestCase):

    def test_returns_tuple_of_two_strings(self):
        k = combo_key({"page_size": 128}, {"bq_sz": 256})
        self.assertEqual(k, ('{"page_size":128}', '{"bq_sz":256}'))

    def test_key_order_does_not_matter(self):
        k1 = combo_key({"a": 1, "b": 2}, {"x": 3, "y": 4})
        k2 = combo_key({"b": 2, "a": 1}, {"y": 4, "x": 3})
        self.assertEqual(k1, k2)

    def test_distinct_pairs_produce_distinct_keys(self):
        k1 = combo_key({"a": 1}, {"x": 1})
        k2 = combo_key({"a": 2}, {"x": 1})
        k3 = combo_key({"a": 1}, {"x": 2})
        self.assertEqual(len({k1, k2, k3}), 3)

    def test_hashable_for_skip_set_use(self):
        k = combo_key({"p": 128}, {"b": 256})
        s = {k}
        self.assertIn(k, s)


class TestServiceComboKey(unittest.TestCase):

    def test_returns_single_canonical_string(self):
        self.assertEqual(
            service_combo_key({"MNB": 8192, "MNS": 128}),
            '{"MNB":8192,"MNS":128}',
        )

    def test_dict_order_does_not_matter(self):
        a = {"MNB": 8192, "MNS": 128}
        b = {"MNS": 128, "MNB": 8192}
        self.assertEqual(service_combo_key(a), service_combo_key(b))


class TestWorkerBucket(unittest.TestCase):
    """v1-parity: each worker measures only the combos whose stable
    hash bucket matches its id. See core/keyset.worker_bucket
    docstring for the architecture."""

    def test_buckets_in_range(self):
        from tools.tuning.v2.core.keyset import worker_bucket
        for i in range(100):
            b = worker_bucket({"i": i}, worker_count=8)
            self.assertGreaterEqual(b, 0)
            self.assertLess(b, 8)

    def test_stable_across_calls(self):
        """Reproducibility: same key + same worker_count -> same
        bucket every invocation. SHA-256-based; not subject to
        Python's PYTHONHASHSEED randomization."""
        from tools.tuning.v2.core.keyset import worker_bucket
        k = {"page_size": 128, "kernel_K": 256}
        self.assertEqual(
            worker_bucket(k, 4), worker_bucket(k, 4),
        )

    def test_distribution_roughly_uniform(self):
        """Sanity: a few hundred keys should spread across all
        buckets, not pile up. (Not a statistical guarantee — just
        a smoke check.)"""
        from collections import Counter
        from tools.tuning.v2.core.keyset import worker_bucket
        counts = Counter(
            worker_bucket({"i": i}, 4) for i in range(400)
        )
        self.assertEqual(set(counts), {0, 1, 2, 3})
        # Each bucket should get ~100 ± slack.
        for c in counts.values():
            self.assertGreater(c, 50)
            self.assertLess(c, 150)

    def test_single_worker_always_returns_zero(self):
        from tools.tuning.v2.core.keyset import worker_bucket
        self.assertEqual(worker_bucket("anything", 1), 0)
        self.assertEqual(worker_bucket({"x": 1}, 1), 0)

    def test_invalid_worker_count_raises(self):
        from tools.tuning.v2.core.keyset import worker_bucket
        with self.assertRaises(ValueError):
            worker_bucket("k", 0)
        with self.assertRaises(ValueError):
            worker_bucket("k", -1)


if __name__ == "__main__":
    unittest.main()
