"""Check editable source identity and CPU imports before publishing an image."""
from __future__ import annotations

import importlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import torch
import torchvision
import jax

for name, root in (('vllm', '/opt/vllm'), ('tpu_inference', '/opt/tpu-inference')):
    module = importlib.import_module(name)
    if not Path(module.__file__).resolve().is_relative_to(root):
        raise RuntimeError(f'{name} imports outside its selected source')
if torch.version.cuda is not None:
    raise RuntimeError('Expected CPU Torch for this TPU runtime')
importlib.import_module('tpu_inference.layers.jax.moe')
subprocess.run(args=[sys.executable, '-m', 'vllm.entrypoints.cli.main', 'serve', '--help'], check=True)
metadata = Path('/opt/jobset/image-source.json')
value = json.loads(metadata.read_text())
value['packages'] = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}
value['python'] = sys.version
metadata.write_text(json.dumps(value, indent=2) + '\n')
print('Runtime CPU import and CLI checks passed', flush=True)
