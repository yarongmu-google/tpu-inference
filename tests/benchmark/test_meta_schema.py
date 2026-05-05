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
"""Cross-language drift detector for meta.txt schema.

The meta.txt key namespace is duplicated across two languages:
``run_benchmark.sh`` writes ``echo "<key>=$VAR"`` lines; ``compare.py``
references those keys as ``meta.<key>`` strings in DEFAULT_COLUMNS. A
rename or removal on either side breaks the rendered table silently
(missing field shows blank instead of raising). This test parses the
bash script statically and asserts every ``meta.<key>`` referenced by
DEFAULT_COLUMNS has a corresponding writer.

Static parse — no shell execution. We only need to find the writers,
not run them.
"""

import re
import unittest
from pathlib import Path

from tools.benchmark import build_service_registry as compare


# Meta-writer pattern: an `echo "<key>=...` form, where <key> is a
# lowercase identifier. This shape matches every meta line in
# run_benchmark.sh and excludes diagnostic echos like
# `echo "case file not found: $CASE_FILE"` (no `=` after first word)
# or `echo "----"` (no leading identifier).
_META_WRITE_RE = re.compile(r'^\s*echo\s+"([a-z][a-z0-9_]*)=')


def _meta_keys_from_script(script_path: Path) -> set[str]:
    """Return the set of meta keys written by `echo "<key>=..."` lines."""
    keys: set[str] = set()
    with open(script_path) as f:
        for line in f:
            m = _META_WRITE_RE.match(line)
            if m:
                keys.add(m.group(1))
    return keys


def _meta_columns_in_default() -> set[str]:
    """Return the set of `meta.<key>` keys referenced by DEFAULT_COLUMNS."""
    out: set[str] = set()
    for key, _label in compare.DEFAULT_COLUMNS:
        if key.startswith("meta."):
            out.add(key.split(".", 1)[1])
    return out


def _run_benchmark_path() -> Path:
    return (Path(__file__).resolve().parent.parent.parent
            / "tools" / "benchmark" / "run_benchmark.sh")


class TestMetaSchema(unittest.TestCase):

    def test_every_default_column_meta_key_is_written_by_bash(self):
        """compare.DEFAULT_COLUMNS shouldn't reference meta.X with no writer.

        If this fails, either the column was added without a bash writer
        (typo or premature column add) or the bash writer was renamed
        without updating compare.DEFAULT_COLUMNS.
        """
        bash_keys = _meta_keys_from_script(_run_benchmark_path())
        column_keys = _meta_columns_in_default()
        missing = column_keys - bash_keys
        self.assertFalse(
            missing,
            f"DEFAULT_COLUMNS references meta.{{{', '.join(sorted(missing))}}}"
            " but run_benchmark.sh has no matching `echo \"<key>=...\"` writer."
            f" Bash writers: {sorted(bash_keys)}")

    def test_meta_writer_extraction_finds_known_keys(self):
        """Sanity check: the regex extracts the keys we expect.

        Not exhaustive — just guards against a regex change that would
        accidentally start matching nothing (which would make the
        drift-detector test above silently pass).
        """
        keys = _meta_keys_from_script(_run_benchmark_path())
        # A representative subset that should always be present.
        for k in ("case_name", "model", "max_num_seqs", "request_rate",
                  "block_size", "git_commit", "bench_duration_seconds"):
            self.assertIn(k, keys)


if __name__ == "__main__":
    unittest.main()
