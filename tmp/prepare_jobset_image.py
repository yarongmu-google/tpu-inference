#!/usr/bin/env python3
"""Snapshot the active Linux environment and tracked sources for a local image."""

import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from collections.abc import Callable


RUNTIME_DIRECTORIES = (
    "bin", "lib", "lib64", "include", "share", "ssl", "libexec",
    "compiler_compat", "x86_64-conda-linux-gnu",
)


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(args=["git", "-C", str(root), *args], text=True).strip()


def snapshot(root: Path, target: Path, keep: Callable[[str], bool]) -> str:
    revision = git(root, "rev-parse", "HEAD")
    tracked = [p for p in git(root, "ls-files", "-z").split("\0") if p and keep(p)]
    changes = set(git(root, "diff", "--name-only", "-z", "HEAD").split("\0"))
    dirty = changes.intersection(tracked)
    if dirty:
        raise RuntimeError("Commit the source changes before building: " + ", ".join(sorted(dirty)))
    target.mkdir(parents=True, exist_ok=True)
    for relative in tracked:
        source = root / relative
        if not source.is_file():
            raise RuntimeError(f"Source snapshot requires a regular file: {source}")
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src=source, dst=destination)
    (target / "JOBSET_REVISION").write_text(revision + "\n")
    return revision


def application_file(path: str) -> bool:
    return (path.startswith("tpu_inference/")
            or path.startswith("tests/e2e/benchmarking/inferencex/qwen3.5/")
            or path in {"setup.py", "pyproject.toml", "README.md", "LICENSE",
                        "tmp/jobset_runner.py", "tmp/jobset_bootstrap.sh", "tmp/jobset_smoke.sh"}
            or ("/" not in path and path.startswith("requirements")))


def main() -> None:
    if len(sys.argv) != 5:
        raise SystemExit("Expected CONTEXT REPOSITORY INFERENCEX DIAGNOSTICS")
    context, repository, client, diagnostics = (Path(p).resolve() for p in sys.argv[1:])
    prefix = Path(sys.prefix).resolve()
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("Build on the Linux x86_64 CPU VM")
    if sys.version_info[:2] != (3, 12) or not (prefix / "conda-meta").is_dir():
        raise RuntimeError("Activate the existing Python 3.12 vllm12 environment first")
    if os.environ.get("CONDA_DEFAULT_ENV") != "vllm12":
        raise RuntimeError("Expected the active conda environment to be vllm12")
    if any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_./-" for c in str(prefix)):
        raise RuntimeError("The environment prefix contains unsupported Docker path characters")
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        raise RuntimeError("Cannot locate vLLM source in the active environment")
    vllm = Path(git(Path(spec.origin).resolve().parent, "rev-parse", "--show-toplevel"))
    from setuptools_scm import get_version
    version = get_version(root=str(vllm))
    version += ".tpu" if "+" in version else "+tpu"
    metadata = {
        "python": sys.version, "environment_prefix": str(prefix),
        "platform": platform.platform(), "vllm_version": version,
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "source_revisions": {},
    }
    metadata["source_revisions"]["tpu_inference"] = snapshot(
        root=repository, target=context / "tpu-inference", keep=application_file)
    metadata["source_revisions"]["vllm"] = snapshot(
        root=vllm, target=context / "vllm",
        keep=lambda p: p.split("/", 1)[0] not in {"tmp", "docs", "examples", ".github", ".buildkite"})
    # The sweep uses bench_serving; the separate aiperf submodule is not needed.
    metadata["source_revisions"]["InferenceX"] = snapshot(
        root=client, target=context / "InferenceX",
        keep=lambda p: p.startswith("utils/") and p != "utils/aiperf")
    # Preserve already-built native modules needed by the editable source tree.
    for artifact in (vllm / "vllm").rglob("*"):
        if artifact.is_file() and (artifact.name.endswith(".so") or ".so." in artifact.name or artifact.name == "vllm-rs"):
            destination = context / "vllm" / artifact.relative_to(vllm)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src=artifact, dst=destination)
    (context / "vllm" / "JOBSET_VERSION").write_text(version + "\n")
    environment = context / "environment"
    environment.mkdir()
    for name in RUNTIME_DIRECTORIES:
        source = prefix / name
        if source.is_symlink():
            (environment / name).symlink_to(os.readlink(source))
        elif source.is_dir():
            print(f"Snapshotting environment directory: {name}", flush=True)
            shutil.copytree(src=source, dst=environment / name, symlinks=True)
    # Keep the original prefix so Python scripts and compiled-library paths
    # remain valid without installing or activating conda inside the image.
    (context / "image-source.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (diagnostics / "image-source.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (diagnostics / "environment-prefix.txt").write_text(str(prefix) + "\n")
    shutil.copy2(src=repository / "tmp/Dockerfile.vllm12", dst=context / "Dockerfile")
    print(json.dumps(metadata["source_revisions"], indent=2), flush=True)


if __name__ == "__main__":
    main()
