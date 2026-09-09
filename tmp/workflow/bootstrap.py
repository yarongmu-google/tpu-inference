"""Prepare a workflow-local YAML environment without modifying the runtime image."""
import fcntl
from pathlib import Path
import subprocess
import sys
import venv

root = Path(__file__).resolve().parent
with (root / 'local/setup.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    python = root / '.venv/bin/python'
    if not python.exists():
        print('Creating workflow-local Python environment', flush=True)
        venv.EnvBuilder(with_pip=True).create(root / '.venv')
    check = subprocess.run(args=[str(python), '-c', 'import yaml; assert yaml.__version__ == "6.0.3"'],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if check.returncode:
        print('Installing workflow YAML dependency from PyPI', flush=True)
        subprocess.run(args=[str(python), '-m', 'pip', 'install', '--disable-pip-version-check',
                             '-r', str(root / 'requirements.txt')], check=True)
