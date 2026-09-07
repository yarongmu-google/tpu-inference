#!/usr/bin/env bash
# Synthetic probes only. Run locally; this script never contacts a remote host.
# Default: CPU interpretation. TPU compile/bench modes must be selected explicitly.
# Override LAYER_PROBE_PYTHON to select an existing environment; no installs.
# Complete pipeline campaign on an already attached TPU:
#   bash tmp/run_layer_probes.sh --suite pipeline --mode bench --profile --budget 7200
# Baseline families only: add --sweep none. Every case gets a bounded child
# process; capability rejections are recorded and do not abort later cases.
set -euo pipefail
probe_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
probe_python="${LAYER_PROBE_PYTHON:-python}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
exec "$probe_python" "$probe_script_dir/run_layer_probes.py" "$@"
