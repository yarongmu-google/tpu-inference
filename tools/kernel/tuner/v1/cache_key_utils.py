# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Cache-key helpers for the rpa_v3 kernel tuner.

Lives in its own module (no JAX/TPU imports) so the env-var wiring and
the dict-normalization can be unit-tested without instantiating the
full RpaV3KernelTuner — the latter pulls in JAX and the RPA kernel,
which arent available outside the TPU VM.
"""

import json
import os

# Field name stripped from the TuningKey when SKIP_COMMIT_CACHE_CHECK
# is enabled. Keeping it as a module-level constant so the producer
# (TuningKey.code_revision) and the consumer (this module) cant drift.
_COMMIT_FIELD = "code_revision"


def should_skip_commit_cache_check(env=None):
    """Return True when the user has opted into commit-agnostic caching.

    Mirrors sweep.py: env var SKIP_COMMIT_CACHE_CHECK == "1" enables.
    Anything else (unset, "0", "true", "yes", garbage) keeps the
    default behaviour of including the commit SHA in the cache key.
    """
    env = env if env is not None else os.environ
    return env.get("SKIP_COMMIT_CACHE_CHECK", "0") == "1"


def tuning_key_hash(tk_dict, *, skip_commit_cache_check):
    """Stable hash of a TuningKey dict, optionally stripping the commit field.

    When skip_commit_cache_check is True, code_revision is excluded
    before hashing so an entry tuned under commit A matches a key
    constructed under commit B. Used on both the load path (reading
    an existing registry) and the lookup path (deciding whether to
    skip a generated case) — they MUST normalize identically or the
    skip logic silently misses.
    """
    if skip_commit_cache_check:
        tk_dict = {k: v for k, v in tk_dict.items() if k != _COMMIT_FIELD}
    return json.dumps(tk_dict, sort_keys=True)
