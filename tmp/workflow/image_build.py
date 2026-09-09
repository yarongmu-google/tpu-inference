"""Resolve a described image preparation command to one immutable image."""
from __future__ import annotations

import os
import fcntl
from pathlib import Path
import re
import shlex
import signal
import subprocess
import threading

from core import read_document, save

IMAGE = re.compile(r'[a-z0-9._:/-]+@sha256:[a-f0-9]{64}')
REPOSITORY = re.compile(r'[a-z0-9-]+-docker\.pkg\.dev/[a-z0-9-]+/[a-z0-9._-]+/[a-z0-9._/-]+')


def resolve(build: dict, repository: str, directory: Path, stop: threading.Event, run_id: str) -> dict:
    directory.mkdir(parents=True, exist_ok=False)
    result = directory / 'result.json'
    state = directory / 'status.json'
    save(path=state, value={'phase': 'preparing', 'repository': repository, 'build': build})
    print(f'Image preparation logs: {directory}', flush=True)
    print('+ ' + shlex.join(build['argv']), flush=True)
    process = None
    pump = None
    log = (directory / 'command.log').open('w')
    lock = (directory / '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        process = subprocess.Popen(args=build['argv'], cwd=build['cwd'],
            env={**os.environ, 'IMAGE_REPOSITORY': repository, 'IMAGE_RESULT': str(result),
                 'IMAGE_BUILD_LOG_DIR': str(directory), 'IMAGE_RUN_ID': run_id},
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True, pass_fds=(lock.fileno(),))
        def stream() -> None:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end='', flush=True)
        pump = threading.Thread(target=stream, daemon=True)
        pump.start()
        elapsed = 0
        while process.poll() is None:
            if stop.wait(timeout=1):
                raise RuntimeError('Image preparation interrupted; no job submitted')
            elapsed += 1
            if elapsed % 30 == 0:
                print(f'Image preparation still running ({elapsed}s); log: {directory / "command.log"}', flush=True)
            if elapsed >= build['timeout_seconds']:
                raise TimeoutError('Image preparation exceeded its timeout')
        pump.join(timeout=5)
        if process.returncode != 0:
            raise RuntimeError(f'Image preparation exited {process.returncode}; see {directory}')
        value = read_document(path=result)
        image = value.get('image', '')
        if not isinstance(image, str) or not IMAGE.fullmatch(image) or not image.startswith(repository + '@'):
            raise ValueError('Image preparation did not return a digest in the configured repository')
        if value.get('owner_run_id') != run_id:
            raise ValueError('Image preparation returned a different owner run ID')
        save(path=state, value={**value, 'phase': 'ready'})
        print(f'Runtime image: {image}', flush=True)
        return value
    except BaseException as error:
        save(path=state, value={'phase': 'failed', 'error': str(error)})
        raise
    finally:
        if process is not None:
            # Include descendants if an interrupted build left a Docker CLI running.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        if pump is not None:
            pump.join(timeout=5)
            if pump.is_alive():
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                pump.join(timeout=5)
        if process is not None and process.stdout is not None:
            process.stdout.close()
        log.close()
        lock.close()
