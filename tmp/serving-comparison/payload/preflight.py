"""Validate runtime access and exact CLI arguments before loading model weights."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import runpy
import sys
from unittest.mock import patch


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
    for name in ('tpu_inference.layers.jax.moe', 'tpu_inference.kernels.fused_moe.v2.decode_kernel'):
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
    if mode == 'hardware':
        hardware(expected_devices=int(args[0]))
    elif mode == 'server':
        server(argv=args)
    elif mode == 'client':
        client(script=Path(args[0]), argv=args[1:])
    else:
        raise ValueError(f'Unknown preflight mode: {mode}')
