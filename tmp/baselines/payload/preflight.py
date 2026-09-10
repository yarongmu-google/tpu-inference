"""Validate runtime access and exact CLI arguments before loading model weights."""
from __future__ import annotations

import argparse
import fnmatch
import os
import re
import shutil
import importlib
import importlib.metadata
import json
from pathlib import Path
import runpy
import sys
from unittest.mock import patch


def checkpoint(model: str, destination: Path) -> None:
    from huggingface_hub import HfApi, snapshot_download
    scratch = Path('/run-scratch')
    if not any(line.split()[4] == str(scratch) for line in Path('/proc/self/mountinfo').read_text().splitlines()):
        raise RuntimeError('Checkpoint scratch mount is missing')
    cache = Path(os.environ['HF_HUB_CACHE']).resolve()
    if not cache.is_relative_to(scratch.resolve()):
        raise ValueError('Checkpoint cache must be inside the scratch mount')
    cache.mkdir(parents=True, exist_ok=True)
    info = HfApi().model_info(repo_id=model, files_metadata=True)
    if not re.fullmatch('[a-f0-9]{40}', info.sha):
        raise ValueError('Cannot pin checkpoint revision')
    patterns = ['*.safetensors', '*.json', '*.txt', '*.model', '*.tiktoken']
    files = [entry for entry in info.siblings if any(fnmatch.fnmatchcase(entry.rfilename, pattern) for pattern in patterns)]
    if not files or not any(entry.rfilename.endswith('.safetensors') for entry in files):
        raise ValueError('Checkpoint has no safetensors weights')
    if any(type(entry.size) is not int or entry.size <= 0 for entry in files):
        raise ValueError('Checkpoint file sizes are missing')
    needed = sum(entry.size for entry in files)
    free = shutil.disk_usage(cache).free
    reserve = 64 * 1024**3
    record = {'model': model, 'revision': info.sha, 'checkpoint_bytes': needed,
              'free_bytes_before': free, 'reserve_bytes': reserve, 'cache': str(cache),
              'files': [{'path': entry.rfilename, 'bytes': entry.size} for entry in files]}
    destination.write_text(json.dumps(record, indent=2) + '\n')
    print(f'CHECKPOINT_CAPACITY: need {needed / 1024**3:.1f} GiB + 64 GiB headroom; free {free / 1024**3:.1f} GiB', flush=True)
    if free < needed + reserve:
        raise RuntimeError('Insufficient checkpoint scratch space; no weight download started')
    snapshot = Path(snapshot_download(repo_id=model, revision=info.sha, cache_dir=str(cache),
                                     allow_patterns=patterns, max_workers=2))
    if not snapshot.resolve().is_relative_to(cache):
        raise ValueError('Downloaded snapshot is outside the scratch cache')
    for entry in files:
        path = snapshot / entry.rfilename
        if not path.resolve().is_relative_to(cache) or not path.is_file() or path.stat().st_size != entry.size:
            raise ValueError(f'Incomplete checkpoint file: {entry.rfilename}')
    record.update(snapshot=str(snapshot), free_bytes_after=shutil.disk_usage(cache).free)
    destination.write_text(json.dumps(record, indent=2) + '\n')
    print('CHECKPOINT_READY: all selected files verified by size at one revision', flush=True)


def hardware(expected_devices: int) -> None:
    import jax
    import numpy as np
    devices = jax.devices()
    if len(devices) != expected_devices or any(device.platform != 'tpu' for device in devices):
        raise RuntimeError(f'Expected {expected_devices} TPU devices, got {devices}')
    for device in devices:
        value = jax.device_put(np.array([1], dtype=np.int32), device=device)
        result = jax.jit(lambda x: x + 1)(value)
        if int(np.asarray(result.block_until_ready())[0]) != 2:
            raise RuntimeError(f'TPU execution check failed on {device}')
    for name in ('tpu_inference.layers.jax.moe',):
        importlib.import_module(name)
    print(json.dumps({'devices': [str(device) for device in devices], 'versions': {
        name: importlib.metadata.version(name) for name in ('jax', 'jaxlib', 'libtpu', 'vllm', 'tpu-inference')}}), flush=True)


def server(argv: list[str]) -> None:
    from vllm.entrypoints.cli.serve import ServeSubcommand
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    parser = FlexibleArgumentParser()
    command = ServeSubcommand()
    command.subparser_init(parser.add_subparsers(dest='subparser'))
    args = parser.parse_args(['serve', *argv])
    command.validate(args)
    print('Server arguments validated without loading model weights', flush=True)


def client(script: Path, argv: list[str]) -> None:
    class Parsed(Exception):
        pass
    original = argparse.ArgumentParser.parse_args
    def parse(parser, *args, **kwargs):
        value = original(parser, *args, **kwargs)
        raise Parsed(value)
    # Run the original parser and imports, stopping before its benchmark main.
    with patch.object(sys, 'argv', [str(script), *argv]), \
            patch.object(sys, 'path', [str(script.parent), *sys.path]), \
            patch.object(argparse.ArgumentParser, 'parse_args', parse):
        try:
            runpy.run_path(str(script), run_name='__main__')
        except Parsed as parsed:
            args = parsed.args[0]
        else:
            raise RuntimeError('Client did not expose the expected argument parser')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    if not tokenizer.encode('runtime check', add_special_tokens=False):
        raise RuntimeError('Tokenizer produced no tokens')
    print('Client arguments and tokenizer access validated without sending requests', flush=True)


if __name__ == '__main__':
    mode, *args = sys.argv[1:]
    if mode == 'checkpoint':
        checkpoint(model=args[0], destination=Path(args[1]))
    elif mode == 'hardware':
        hardware(expected_devices=int(args[0]))
    elif mode == 'server':
        server(argv=args)
    elif mode == 'client':
        client(script=Path(args[0]), argv=args[1:])
    else:
        raise ValueError(f'Unknown preflight mode: {mode}')
