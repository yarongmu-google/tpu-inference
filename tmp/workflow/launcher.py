"""Keep local collection running when its terminal log viewer disconnects."""
from __future__ import annotations

import gzip
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parent


def worker(log: Path, arguments: list[str]) -> int:
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    child = None
    stopped = 0

    def terminate(number: int, _frame: object) -> None:
        nonlocal stopped
        stopped = number
        if child is not None and child.poll() is None:
            child.send_signal(number)

    signal.signal(signal.SIGTERM, terminate)
    code = 1
    try:
        for command in ([sys.executable, str(ROOT / 'bootstrap.py')],
                        [str(ROOT / '.venv/bin/python'), str(ROOT / 'controller.py'), *arguments]):
            if stopped:
                break
            print('+', ' '.join(command), flush=True)
            child = subprocess.Popen(args=command)
            code = child.wait()
            if code:
                break
        if stopped:
            code = 128 + stopped
        elif code < 0:
            code = 128 - code
    except Exception:
        traceback.print_exc()
        code = 1
    finally:
        print(f'Launcher {"completed" if code == 0 else "FAILED"} (exit {code})', flush=True)
        packed = Path(str(log) + '.gz')
        partial = Path(str(packed) + '.partial')
        try:
            print(f'Compressed launcher log: {packed}', flush=True)
            with log.open('rb') as source, gzip.open(partial, 'wb') as target:
                shutil.copyfileobj(source, target)
            with gzip.open(partial, 'rb') as verified:
                while verified.read(1024 * 1024):
                    pass
            partial.replace(packed)
        except Exception:
            traceback.print_exc()
            print(f'Launcher log compression failed; raw log retained at {log}', flush=True)
            code = code or 1
        result = Path(str(log) + '.exit')
        temporary = Path(str(result) + '.partial')
        temporary.write_text(str(code) + '\n')
        temporary.replace(result)
    return code


def launch(arguments: list[str]) -> int:
    logs = ROOT / 'local/logs'
    logs.mkdir(parents=True, exist_ok=True)
    descriptor, filename = tempfile.mkstemp(prefix=time.strftime('launch-%Y%m%dT%H%M%SZ-', time.gmtime()),
                                          suffix='.log', dir=logs)
    log = Path(filename)
    # Inherit stdin for initial interactive configuration. All controller output
    # goes to a regular file, and the worker has a separate process session.
    with os.fdopen(descriptor, 'wb') as output:
        process = subprocess.Popen(args=[sys.executable, str(Path(__file__).resolve()),
                                         '--worker', str(log), *arguments],
                                   stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
                                   env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1'})
    Path(str(log) + '.pid').write_text(str(process.pid) + '\n')
    try:
        print(f'Local controller PID: {process.pid}\nLauncher log: {log}', flush=True)
        print('Ctrl-C or closing this terminal detaches the viewer; collection continues on the CPU VM.', flush=True)
        with log.open('rb') as source:
            while True:
                data = source.read(65536)
                if data:
                    sys.stdout.write(data.decode(errors='replace'))
                    sys.stdout.flush()
                elif process.poll() is not None:
                    sys.stdout.write(source.read().decode(errors='replace'))
                    sys.stdout.flush()
                    break
                else:
                    time.sleep(0.1)
        result = Path(str(log) + '.exit')
        if not result.is_file():
            print(f'Controller exited before finalization; raw log: {log}', flush=True)
            return 1
        return int(result.read_text())
    except (KeyboardInterrupt, BrokenPipeError, OSError):
        try:
            print(f'Viewer detached; controller PID {process.pid} continues. Log: {log}', file=sys.stderr)
        except OSError:
            pass
        return 130


if __name__ == '__main__':
    if sys.argv[1:2] == ['--worker']:
        sys.exit(worker(log=Path(sys.argv[2]), arguments=sys.argv[3:]))
    sys.exit(launch(arguments=sys.argv[1:]))
