#!/usr/bin/env bash
# Install the supplied source trees inside the runner's diagnostic capture.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_SOURCE_DIR="${VLLM_SOURCE_DIR:-/opt/vllm}"
[[ $# -eq 1 ]] || { echo "Expected the benchmark script path" >&2; exit 2; }
for source in "$VLLM_SOURCE_DIR" "$ROOT"; do
  if [[ ! -f "$source/setup.py" && ! -f "$source/pyproject.toml" ]]; then
    echo "ERROR: missing installable source tree: $source" >&2
    exit 2
  fi
  if revision="$(git -C "$source" rev-parse HEAD 2>/dev/null)"; then
    echo "Source: $source; Git revision: $revision"
  else
    echo "Source: $source; Git metadata unavailable; packaged image: ${RUN_IMAGE:-unknown}"
  fi
done
# Preserve the captured environment; missing build/runtime dependencies must
# be resolved in the image rather than silently upgraded on each TPU attempt.
set -x
VLLM_TARGET_DEVICE=tpu python3 -m pip install --no-deps --no-build-isolation -e "$VLLM_SOURCE_DIR"
python3 -m pip install --no-deps --no-build-isolation -e "$ROOT"
set +x
python3 - "$VLLM_SOURCE_DIR" "$ROOT" <<'PY_SOURCE'
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import vllm
import tpu_inference

metadata = {"python": sys.version, "vllm_runtime_version": vllm.__version__,
            "packages": {}}
for name, module, source in (("vllm", vllm, sys.argv[1]),
                             ("tpu_inference", tpu_inference, sys.argv[2])):
    path = Path(module.__file__).resolve()
    if not path.is_relative_to(Path(source).resolve()):
        raise RuntimeError(f"{name} imports from {path}, outside requested source {source}")
    metadata["packages"][name] = {
        "distribution_version": importlib.metadata.version(name), "import_path": str(path)}
text = json.dumps(metadata, indent=2) + "\n"
print(text, end="", flush=True)
(Path(os.environ["RUN_METADATA_DIR"]) / "installed-sources.json").write_text(text)
PY_SOURCE
exec bash "$1"
