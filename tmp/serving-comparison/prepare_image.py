"""Require matching source branches before invoking the existing image builder."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tmp'))
import prepare_runtime_image


def check_branch(root: Path) -> None:
    branch = subprocess.check_output(
        args=['git', '-C', str(root), 'branch', '--show-current'], text=True).strip()
    if branch != 'topk':
        raise RuntimeError(f'Expected topk source in {root}, got {branch or "detached HEAD"}')


def main() -> None:
    check_branch(root=ROOT)
    spec = importlib.util.find_spec('vllm')
    if spec is None or spec.origin is None:
        raise RuntimeError('Activate the existing vllm12 environment with editable topk vLLM')
    check_branch(root=Path(spec.origin).resolve().parent)
    prepare_runtime_image.main()


if __name__ == '__main__':
    main()
