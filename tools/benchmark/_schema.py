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
"""Single source of truth for filenames + canonical-metric strings used
across the benchmark toolchain.

Without this, the same string lived independently in
parse_bench_log.py (writer), sweep.py (completion gate), and
compare.py (default sort key) — three places to keep in sync. Renaming
the canonical metric or one of the filenames would break two call
sites silently.

run_benchmark.sh writes its own copies of these strings; the shell
can't import Python. So a typo on the bash side still slips through —
that's what tests/benchmark/test_meta_schema.py catches with a static
cross-check.
"""

# Files written into each combo's result dir by run_benchmark.sh.
METRICS_FILENAME = "metrics.txt"
META_FILENAME = "meta.txt"

# Canonical "did this combo actually finish" metric. parse_bench_log
# emits it as a key, sweep.is_completed gates resumability on it being
# non-empty, compare.py uses it as the default sort key.
THROUGHPUT_METRIC = "RequestThroughput"
