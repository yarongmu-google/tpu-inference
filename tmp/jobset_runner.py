#!/usr/bin/env python3
"""Run a script and publish success or failure diagnostics to a mounted output path."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def publish(local: Path, remote: Path, status: dict) -> None:
    remote.mkdir(parents=True, exist_ok=True)
    files = []
    for source in sorted(local.rglob("*")):
        if not source.is_file() or source.is_symlink():
            continue
        relative = source.relative_to(local)
        target = remote / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        # Close each snapshot before publishing the manifest. Readers verify hashes.
        shutil.copyfile(src=source, dst=target)
        files.append({"path": relative.as_posix(), "bytes": target.stat().st_size,
                      "sha256": digest(path=target)})
    manifest = {"format": "sweep-jobset-v1", **status, "files": files}
    with (remote / "manifest.json").open("w") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")


def restore(previous: Path, local: Path, image: str) -> None:
    manifest = json.loads((previous / "manifest.json").read_text())
    if manifest.get("format") != "sweep-jobset-v1" or manifest.get("image") != image:
        raise ValueError("Resume bundle must come from the same immutable image")
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid resume artifact path")
        if not relative.parts or relative.parts[0] != "client" or relative.suffix != ".json":
            continue
        source = previous / relative
        if source.is_symlink() or digest(path=source) != entry["sha256"]:
            raise ValueError(f"Resume artifact checksum mismatch: {relative}")
        target = local / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src=source, dst=target)


def main() -> int:
    output = os.environ.get("CDK_OUTPUT_DIR", "")
    if not output or output.startswith("gs://") or not Path(output).is_dir():
        print("CDK_OUTPUT_DIR must be an existing writable mounted directory", file=sys.stderr)
        return 2
    attempt = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:8]
    local = Path(tempfile.mkdtemp(prefix="sweep-jobset-"))
    remote = Path(output) / "attempts" / attempt
    log_path = local / "job.log"
    image = os.environ.get("RUN_IMAGE", "")
    status = {"attempt": attempt, "image": image, "state": "running", "exit_code": None}
    child = None
    reader = None
    received_signal = 0
    signal_deadline = None
    upload_failed = False

    def forward_signal(number: int, _frame: object) -> None:
        nonlocal received_signal, signal_deadline
        received_signal = number
        signal_deadline = time.monotonic() + 30
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, number)
            except ProcessLookupError:
                pass

    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, forward_signal)

    with log_path.open("w", buffering=1) as log:
        def report(message: str) -> None:
            print(message, flush=True)
            print(message, file=log, flush=True)

        try:
            report(f"Attempt {attempt}; persistent results: {remote}")
            publish(local=local, remote=remote, status=status)
            if "@sha256:" not in image:
                raise ValueError("RUN_IMAGE must identify the image by registry digest")
            root = Path(__file__).resolve().parent.parent
            script = Path(os.environ.get("SWEEP_SCRIPT", "tests/e2e/benchmarking/inferencex/qwen3.5/sweep.sh"))
            script = (root / script).resolve()
            if not script.is_relative_to(root) or not script.is_file():
                raise ValueError("SWEEP_SCRIPT must name a script inside the image's repository")
            packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
            (local / "environment.json").write_text(json.dumps({
                "python": sys.version, "packages": packages, "script": str(script),
                "image": image,
            }, indent=2) + "\n")
            resume = os.environ.get("RESUME_DIR", "")
            if resume:
                restore(previous=Path(resume), local=local, image=image)
                status["resumed_from"] = resume
            env = os.environ.copy()
            env["RESULT_DIR"] = str(local / "client")
            env["SERVER_LOG_DIR"] = str(local / "server")
            env["PYTHONUNBUFFERED"] = "1"
            env["RUN_ATTEMPT"] = attempt
            env["RUN_METADATA_DIR"] = str(local)
            child = subprocess.Popen(args=["bash", str(root / "tmp/jobset_bootstrap.sh"), str(script)], cwd=root, env=env,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     start_new_session=True)

            def copy_output() -> None:
                assert child is not None and child.stdout is not None
                for line in iter(child.stdout.readline, b""):
                    text = line.decode("utf-8", errors="replace")
                    log.write(text)
                    log.flush()
                    try:
                        sys.stdout.write(text)
                        sys.stdout.flush()
                    except BrokenPipeError:
                        pass

            reader = threading.Thread(target=copy_output, daemon=True)
            reader.start()
            next_publish = time.monotonic() + 30
            while child.poll() is None:
                if received_signal and signal_deadline is not None and time.monotonic() >= signal_deadline:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                if time.monotonic() >= next_publish:
                    try:
                        publish(local=local, remote=remote, status=status)
                    except Exception as error:
                        report(f"ERROR publishing diagnostics: {error}")
                        upload_failed = True
                    next_publish = time.monotonic() + 30
                time.sleep(0.2)
            code = child.wait()
            code = 128 + received_signal if received_signal else (128 - code if code < 0 else code)
        except Exception:
            report(traceback.format_exc())
            code = 1
        finally:
            if child is not None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                if reader is not None:
                    reader.join(timeout=5)
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
                if reader is not None:
                    reader.join(timeout=5)
        if upload_failed and code == 0:
            code = 1
        status.update(state="succeeded" if code == 0 else "failed", exit_code=code)
        report(f"Run {status['state']}; exit_code={code}")
        (local / "status.json").write_text(json.dumps(status, indent=2) + "\n")
        try:
            publish(local=local, remote=remote, status=status)
        except Exception:
            report("ERROR: final diagnostic upload failed\n" + traceback.format_exc())
            return code or 1
    return code


if __name__ == "__main__":
    sys.exit(main())
