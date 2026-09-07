#!/usr/bin/env bash
# Check imports and CLI startup without loading a model or accessing TPUs.
set -euo pipefail
export JAX_PLATFORMS=cpu
set -x
python3 - <<'PY'
import importlib
import json
import jax
import torch

assert torch.version.cuda is None, "Expected CPU Torch in the TPU image"

for name in ("torch", "torchvision", "jax", "torchax", "transformers", "vllm",
             "vllm.spinloop", "vllm.fs_io_C", "tpu_inference"):
    module = importlib.import_module(name)
    print(json.dumps({"module": name, "version": getattr(module, "__version__", None),
                      "path": module.__file__}), flush=True)
assert jax.default_backend() == "cpu"
print("CPU backend initialized", flush=True)
PY
vllm --help
python3 "$INFERENCEX_REPO/utils/bench_serving/benchmark_serving.py" --help
set +x
# The input environment already has reported dependency conflicts. Retain the
# updated report separately from executable import/CLI failures.
if python3 -m pip check; then
  echo "Dependency metadata check: passed"
else
  echo "Dependency metadata check: conflicts remain; review the build log"
fi
