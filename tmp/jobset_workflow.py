#!/usr/bin/env python3
"""Build, publish, submit, monitor and collect one resumable CDK sweep."""

import argparse
import configparser
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = "us-central1-docker.pkg.dev/cloud-tpu-inference-test/vllm-tpu-rdna"
LETTER_HASH = "4093b4cb6404558219be50b15b409be1485fd7f64405eb20030c0e9b062ce31f"
JOB_ID = re.compile(r"j-[a-zA-Z0-9-]+")
IMAGE = re.compile(r"[a-z0-9._/-]+@sha256:[a-f0-9]{64}")
TERMINAL = {"Succeeded", "Failed", "Deleted"}
ARCHIVE_FAILED = {"LogArchiveFailed", "TraceGenFailed", "FinalizingFailed", "K8sJobDeleted"}


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".new")
    with temporary.open("w") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def cdk_paths() -> tuple[Path, Path]:
    # Read only the two path settings. Never log or copy the credential-bearing INI.
    values = {}
    config = Path.home() / ".cdk.ini"
    if config.exists():
        parser = configparser.ConfigParser(interpolation=None)
        try:
            content = config.read_text()
            parser.read_string(content if content.lstrip().startswith("[") else "[DEFAULT]\n" + content)
            for section in [parser.defaults(), *(parser[s] for s in parser.sections())]:
                for key in ("cdk_root_dir", "cdk_job_outputs_dir"):
                    if section.get(key):
                        values[key] = section[key]
        except configparser.Error:
            raise RuntimeError("Cannot parse CDK path settings; check ~/.cdk.ini locally") from None
    root = Path(os.environ.get("CDK_SOURCE_DIR", values.get("cdk_root_dir", str(Path.home() / "cloud-devkit"))))
    outputs = Path(os.environ.get("CDK_JOB_OUTPUTS_DIR", values.get("cdk_job_outputs_dir", str(Path.home() / "cloud-devkit-data/job-outputs"))))
    return root.expanduser().resolve(), outputs.expanduser().resolve()


def verify_artifacts(root: Path, expected_image: str | None) -> bool:
    manifests = []
    for path in sorted(root.rglob("manifest.json")):
        data = json.loads(path.read_text())
        if data.get("format") != "sweep-jobset-v1":
            continue
        if expected_image and data.get("image") != expected_image:
            raise ValueError("Downloaded attempt belongs to a different image")
        for entry in data["files"]:
            relative = Path(entry["path"])
            artifact = path.parent / relative
            if (relative.is_absolute() or ".." in relative.parts or artifact.is_symlink()
                    or not artifact.resolve().is_relative_to(path.parent.resolve())):
                raise ValueError("Invalid artifact path")
            if artifact.stat().st_size != entry["bytes"]:
                raise ValueError(f"Artifact size mismatch: {relative}")
            with artifact.open("rb") as stream:
                checksum = hashlib.file_digest(stream, "sha256").hexdigest()
            if checksum != entry["sha256"]:
                raise ValueError(f"Artifact checksum mismatch: {relative}")
        print(f"Verified attempt {data['attempt']}: {data['state']}, exit={data['exit_code']}", flush=True)
        manifests.append(data)
    if not manifests:
        raise ValueError("No sweep manifest; available CDK logs are retained")
    return all(m.get("state") == "succeeded" and m.get("exit_code") == 0 for m in manifests)


class Workflow:
    def __init__(self, directory: Path, state: dict) -> None:
        self.directory = directory
        self.state = state
        self.cdk_root, self.outputs_root = cdk_paths()
        self.poll_seconds = int(os.environ.get("JOBSET_POLL_SECONDS", "30"))
        self.wait_seconds = int(os.environ.get("JOBSET_WAIT_SECONDS", "86400"))
        self.archive_seconds = int(os.environ.get("JOBSET_ARCHIVE_SECONDS", "1200"))
        if min(self.poll_seconds, self.wait_seconds, self.archive_seconds) < 1:
            raise ValueError("Polling and wait durations must be positive")
        (directory / "commands").mkdir(exist_ok=True)

    def save(self, **updates: object) -> None:
        self.state.update(updates)
        atomic_json(path=self.directory / "state.json", value=self.state)

    def command(self, args: list[str], timeout: int = 180, check: bool = True) -> tuple[int, str]:
        number = self.state.get("command_number", 0) + 1
        self.save(command_number=number)
        base = self.directory / "commands" / f"{number:05d}-{Path(args[0]).name}"
        print(f"+ {shlex.join(args)}\n  output: {base}.stdout", flush=True)
        with base.with_suffix(".stdout").open("wb") as stdout, base.with_suffix(".stderr").open("wb") as stderr:
            child = subprocess.Popen(args=args, cwd=ROOT, stdin=subprocess.DEVNULL,
                                     stdout=stdout, stderr=stderr, start_new_session=True)
            code = 1
            try:
                deadline = time.monotonic() + timeout
                while True:
                    try:
                        code = child.wait(timeout=min(30, max(0, deadline - time.monotonic())))
                        break
                    except subprocess.TimeoutExpired:
                        if time.monotonic() >= deadline:
                            raise
                        print(f"Still running {args[0]}; output: {base}.stdout", flush=True)
            except subprocess.TimeoutExpired:
                code = 124
                raise
            except KeyboardInterrupt:
                code = 130
                raise
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                base.with_suffix(".exit").write_text(str(code) + "\n")
        text = base.with_suffix(".stdout").read_text(errors="replace")
        if check and code:
            raise RuntimeError(f"Command exited {code}; see {base}.stderr")
        return code, text

    def cdk(self, args: list[str], check: bool = True, timeout: int = 180) -> tuple[int, str]:
        _, letter = self.command(args=["cdk", "agent-letter"])
        codes = re.findall(r"\[ack (\d+)/(\d+): ([A-Za-z0-9]+)\]", letter)
        normal = " ".join(re.sub(r"\[ack \d+/\d+: [A-Za-z0-9]+\]", "[ack]", letter).split())
        if hashlib.sha256(normal.encode()).hexdigest() != LETTER_HASH:
            raise RuntimeError("CDK agent instructions changed; review the saved letter before continuing")
        if not codes or any(int(i) != n or int(total) != len(codes) for n, (i, total, _) in enumerate(codes, 1)):
            raise RuntimeError("Incomplete CDK acknowledgement codes")
        ack = "-".join(code for _, _, code in codes)
        return self.command(args=["cdk", f"--agent-code={ack}", *args], timeout=timeout, check=check)

    def jobs(self) -> list[dict]:
        _, text = self.cdk(args=["job", "list", "-H", "-n", "100", "-t", self.state["tag"], "-o", "json"])
        rows = json.loads(text)
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError("Unexpected CDK job-list JSON; inspect command output")
        return rows

    def discover(self) -> dict | None:
        rows = self.jobs()
        if len(rows) > 1:
            raise RuntimeError("Multiple jobs match this workflow tag; refusing to choose")
        if not rows:
            return None
        job = rows[0]
        tags = job.get("tags", [])
        tags = tags.split(",") if isinstance(tags, str) else tags
        if (self.state["tag"] not in tags or job.get("user") != self.state["user"]
                or job.get("recipe") != self.state["recipe"]):
            raise ValueError("CDK returned a job outside this workflow")
        if not JOB_ID.fullmatch(job.get("id", "")):
            raise ValueError("Invalid CDK job ID")
        if self.state.get("job_id") and job["id"] != self.state["job_id"]:
            raise ValueError("CDK returned a different job for the saved tag")
        self.save(job_id=job["id"], last_job=job)
        return job

    def register_recipe(self) -> None:
        template = (ROOT / "tmp/jobset.yml").read_bytes()
        checksum = hashlib.sha256(template).hexdigest()[:12]
        name = f"qwen-sweep-{self.state['user_slug']}-{checksum}"
        relative = f"recipes/experimental/{name}/jobset.yml"
        registry = self.cdk_root / "recipes.yml"
        if not registry.is_file():
            raise RuntimeError("CDK recipes.yml missing; check CDK_SOURCE_DIR")
        entry = {"name": name, "owner": self.state["user"], "k8s_file": relative, "require_gcs_mount": True}
        line = "- " + json.dumps(entry)
        # Append one JSON-style YAML entry; preserve all existing recipe text.
        with (self.cdk_root / ".qwen-sweep-recipes.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            before = registry.read_text()
            _, text = self.cdk(args=["recipe", "list", "-o", "json"])
            recipes = json.loads(text)
            if not isinstance(recipes, list) or any(not isinstance(r, dict) or not isinstance(r.get("name"), str) for r in recipes):
                raise ValueError("Unexpected recipe-list JSON")
            matches = [r for r in recipes if r.get("name") == name]
            target = self.cdk_root / relative
            if matches and (len(matches) != 1 or line not in before.splitlines()):
                raise RuntimeError("Recipe name already registered outside this runner")
            if target.exists() and target.read_bytes() != template:
                raise RuntimeError("Existing recipe file differs; refusing to overwrite it")
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                with target.open("xb") as stream:
                    stream.write(template)
            if not matches:
                if before.lstrip().startswith("["):
                    raise RuntimeError("Expected recipes.yml to contain a block-style YAML list")
                temporary = registry.with_name(".recipes.qwen-sweep.new")
                temporary.write_text(before.rstrip() + "\n" + line + "\n")
                shutil.copymode(src=registry, dst=temporary)
                if registry.read_text() != before:
                    raise RuntimeError("recipes.yml changed concurrently; retry registration")
                temporary.replace(registry)
        _, text = self.cdk(args=["recipe", "list", "-o", "json"])
        if sum(r.get("name") == name for r in json.loads(text)) != 1:
            raise RuntimeError("CDK did not discover the registered recipe")
        self.save(recipe=name)

    def cached_image(self) -> str | None:
        spec = importlib.util.spec_from_file_location("snapshot", ROOT / "tmp/prepare_jobset_image.py")
        snapshot = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(snapshot)
        vllm_spec = importlib.util.find_spec("vllm")
        if vllm_spec is None or vllm_spec.origin is None:
            return None
        source = str(Path(vllm_spec.origin).parent)
        _, vllm_revision = self.command(args=["git", "-C", source, "rev-parse", "HEAD"])
        _, dirty = self.command(args=["git", "-C", source, "diff", "--name-only", "HEAD"])
        if dirty.strip():
            raise RuntimeError("Commit or revert vLLM source changes before submitting an image")
        for folder in sorted((ROOT / "tmp/vllm_logs").glob("jobset-build-*"), reverse=True):
            try:
                if (folder / "exit-code.txt").read_text().strip() != "0":
                    continue
                metadata = json.loads((folder / "image-source.json").read_text())
                if metadata["source_revisions"]["vllm"] != vllm_revision.strip():
                    continue
                revision = metadata["source_revisions"]["tpu_inference"]
                code, diff = self.command(args=["git", "diff", "--name-only", revision], check=False)
                relevant = lambda p: snapshot.application_file(p) or p in {"tmp/Dockerfile.vllm12", "tmp/prepare_jobset_image.py"}
                if code or any(relevant(p) for p in diff.splitlines()):
                    continue
                image = (folder / "image-tag.txt").read_text().strip()
                if not re.fullmatch(r"vllm12:topk-[a-f0-9]+", image):
                    continue
                code, _ = self.command(args=["docker", "image", "inspect", "--format", "{{.Id}}", image], check=False)
                if code:
                    continue
                code, actual = self.command(args=["docker", "run", "--pull=never", "--rm", "--network=none", image,
                                                    "cat", "/opt/jobset/image-source.json"], check=False)
                if code or json.loads(actual) != metadata:
                    continue
                self.save(build_directory=str(folder))
                return image
            except (OSError, ValueError, KeyError):
                continue
        return None

    def prepare_image(self, rebuild: bool) -> None:
        if self.state.get("image"):
            return
        local = None if rebuild else self.cached_image()
        if local is None:
            self.command(args=["bash", str(ROOT / "tmp/build_jobset_image.sh")], timeout=14400)
            local = self.cached_image()
        if local is None:
            raise RuntimeError("No verified local image found after building")
        repository = self.state["image_repository"]
        hostname = repository.split("/", 1)[0]
        self.command(args=["gcloud", "auth", "configure-docker", hostname, "--quiet"])
        tag = repository + ":" + self.state["tag"]
        self.command(args=["docker", "tag", local, tag])
        self.command(args=["docker", "push", tag], timeout=14400)
        _, text = self.command(args=["docker", "image", "inspect", "--format", "{{json .RepoDigests}}", tag])
        digests = [d for d in json.loads(text) if d.startswith(repository + "@") and IMAGE.fullmatch(d)]
        if len(digests) != 1:
            raise RuntimeError("Cannot resolve the published registry digest")
        self.save(image=digests[0], local_image=local)

    def resume_bundle(self) -> list[str]:
        previous = self.state.get("resume_from")
        if not previous:
            return []
        source = Path(previous)
        verify_artifacts(root=source, expected_image=self.state["image"])
        spec = importlib.util.spec_from_file_location("jobset_runner", ROOT / "tmp/jobset_runner.py")
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        bundle = self.directory / "resume-input"
        bundle.mkdir(exist_ok=True)
        runner.restore(previous=source, local=bundle, image=self.state["image"])
        shutil.copyfile(src=source / "manifest.json", dst=bundle / "manifest.json")
        if ":" in str(bundle):
            raise ValueError("Resume input path cannot contain a colon")
        return ["--map-dir", f"{bundle}:tpu-worker:/opt/jobset-resume"]

    def collect(self) -> bool:
        job = self.state.get("job_id")
        if not job:
            return False
        destination = self.directory / "collections" / (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:6])
        destination.mkdir(parents=True)
        self.save(collection=str(destination))
        for args, name in ((["job", "desc", job, "--no-color"], "description.yml"),
                           (["job", "recipe", job, "--no-color"], "submitted-recipe.yml"),
                           (["job", "log", job, "-c", "runner"], "container.log")):
            try:
                code, text = self.cdk(args=args, check=False)
                (destination / name).write_text(text)
                if code:
                    print(f"Diagnostic command failed: {name}; stderr retained", flush=True)
                if name == "container.log" and "console.cloud.google.com/logs" in text:
                    print("CDK returned a Log Explorer link; container log retrieval may be incomplete", flush=True)
            except Exception as error:
                print(f"Diagnostic collection error: {error}", flush=True)
        for attempt in range(3):
            try:
                code, _ = self.cdk(args=["job", "sync-outputs", job], check=False, timeout=1800)
                source = self.outputs_root / job
                target_root = destination / f"artifacts-{attempt + 1}"
                safe = True
                if source.is_dir():
                    for path in sorted(source.rglob("*")):
                        if path.is_symlink() or not path.resolve().is_relative_to(source.resolve()):
                            print(f"Skipping unsafe output path: {path.name}", flush=True)
                            safe = False
                            continue
                        if path.is_file():
                            target = target_root / path.relative_to(source)
                            target.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copyfile(src=path, dst=target)
                success = verify_artifacts(root=target_root, expected_image=self.state.get("image"))
                self.save(artifacts_verified=True, artifacts_succeeded=success)
                if code == 0 and safe:
                    return success
                raise RuntimeError("Sync failed or unsafe paths found; partial artifacts retained")
            except Exception as error:
                self.save(artifacts_verified=False, collection_error=str(error))
                print(f"Collection attempt {attempt + 1} incomplete: {error}", flush=True)
                if attempt < 2:
                    time.sleep(min(self.poll_seconds, 10))
        return False

    def run(self, rebuild: bool) -> int:
        if self.state.get("finished"):
            print(f"This workflow already finished with exit {self.state['exit_code']}; use --new for another submission")
            return self.state["exit_code"]
        job = None
        if self.state.get("submission_started"):
            job = self.discover()
            if job is None:
                raise RuntimeError("Submission outcome unknown: no matching job yet. Rerun to look up the same tag; do not submit again blindly")
        else:
            for args in (["--help"], ["job", "--help"], ["job", "create", "--help"],
                         ["job", "list", "--help"], ["job", "sync-outputs", "--help"], ["job", "log", "--help"]):
                self.cdk(args=args)
            self.register_recipe()
            self.prepare_image(rebuild=rebuild)
            mapping = self.resume_bundle()
            resume = "/opt/jobset-resume" if mapping else ""
            self.save(submission_started=True)
            code, _ = self.cdk(args=["job", "create", self.state["recipe"],
                                     "IMAGE=" + self.state["image"], "SCRIPT=" + self.state["script"], "RESUME_DIR=" + resume,
                                     "--tags", self.state["tag"],
                                     "--mount-gcs", "--log-mode", "log-transport", "--active-deadline-seconds", "43200", *mapping],
                               check=False, timeout=300)
            self.save(submission_exit=code)
            for attempt in range(3):
                job = self.discover()
                if job:
                    break
                time.sleep(min(self.poll_seconds, 10))
            if job is None:
                raise RuntimeError("No job discovered after submission; output saved. Rerun to recover by tag")
        _, rendered = self.cdk(args=["job", "recipe", self.state["job_id"], "--no-color"])
        (self.directory / "submitted-recipe.yml").write_text(rendered)
        if self.state["image"] not in rendered or self.state["script"] not in rendered or "<no value>" in rendered:
            raise RuntimeError("Submitted recipe does not contain the requested image/script; inspect saved recipe and job")
        self.save(phase="polling")
        started = time.monotonic()
        terminal_since = None
        errors = 0
        while True:
            if job is not None:
                status, state = job.get("job_status"), job.get("state")
                print(f"{job['id']}: execution={status}; archive={state}", flush=True)
                if status in TERMINAL:
                    if terminal_since is None:
                        terminal_since = time.monotonic()
                    if state == "Complete" or state in ARCHIVE_FAILED:
                        success = self.collect()
                        code = 0 if status == "Succeeded" and state == "Complete" and success else 1
                        self.save(finished=True, phase="finished", exit_code=code)
                        return code
                    if time.monotonic() - terminal_since >= self.archive_seconds:
                        raise TimeoutError("CDK archive wait timed out; rerun to continue polling this job")
            if time.monotonic() - started >= self.wait_seconds:
                raise TimeoutError("Job wait timed out; rerun to continue polling this job")
            time.sleep(self.poll_seconds)
            try:
                job = self.discover()
                if job is None:
                    raise RuntimeError("Saved job absent from CDK list")
                errors = 0
            except Exception:
                errors += 1
                if errors >= 3:
                    raise
                job = None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new", action="store_true", help="Start a new workflow after the previous one finished")
    parser.add_argument("--rebuild", action="store_true", help="Build a fresh image before submission")
    parser.add_argument("--resume-from", type=Path, help="Previous collected attempt directory, for a new job using the same image")
    parser.add_argument("--collect-only", metavar="JOB_ID", help="Retrieve an existing job's diagnostics without submitting")
    args = parser.parse_args()
    if "NO_UPDATE_CHECK" in os.environ:
        raise SystemExit("Unset NO_UPDATE_CHECK so CDK can auto-update as required")
    base = ROOT / "tmp/vllm_logs/jobset-workflows"
    base.mkdir(parents=True, exist_ok=True)
    with (base / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another local JobSet workflow is running")
        folders = sorted(base.glob("run-*/state.json"))
        if folders and not args.new and not args.collect_only:
            if args.resume_from:
                raise SystemExit("Use --new with --resume-from; existing workflow inputs are preserved")
            directory = folders[-1].parent
            state = json.loads(folders[-1].read_text())
        else:
            if args.new and folders:
                previous = json.loads(folders[-1].read_text())
                if previous.get("submission_started") and not previous.get("finished"):
                    raise SystemExit("Previous submission still unresolved; resume it before creating another")
            if args.collect_only and not JOB_ID.fullmatch(args.collect_only):
                raise SystemExit("Invalid CDK job ID")
            directory = base / (("collect-" if args.collect_only else "run-") + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:8])
            directory.mkdir()
            user = os.environ.get("USER", "")
            if not re.fullmatch(r"[a-zA-Z0-9_.-]+", user):
                raise SystemExit("USER must identify the CDK submitting user")
            slug = user.lower().replace("_", "-").replace(".", "-")[:30]
            repository = os.environ.get("JOBSET_IMAGE_REPOSITORY", REGISTRY + "/qwen-sweep-" + slug)
            if not re.fullmatch(r"[a-z0-9.-]+-docker\.pkg\.dev/[a-z0-9._/-]+", repository):
                raise SystemExit("JOBSET_IMAGE_REPOSITORY must be an Artifact Registry image path without a tag")
            script = os.environ.get("JOBSET_SCRIPT", "tests/e2e/benchmarking/inferencex/qwen3.5/sweep.sh")
            if not re.fullmatch(r"[a-zA-Z0-9_./-]+", script) or script.startswith("/") or ".." in Path(script).parts:
                raise SystemExit("JOBSET_SCRIPT must be a relative script inside the image")
            state = {"tag": "qwen-sweep-" + uuid.uuid4().hex[:16], "user": user, "user_slug": slug,
                     "image_repository": repository, "script": script, "phase": "preparing",
                     "resume_from": str(args.resume_from.resolve()) if args.resume_from else None}
        print(f"Workflow diagnostics: {directory}", flush=True)
        workflow = Workflow(directory=directory, state=state)
        workflow.save()
        def interrupted(number: int, frame: object) -> None:
            raise KeyboardInterrupt(f"Signal {number}")
        signal.signal(signal.SIGTERM, interrupted)
        try:
            if args.collect_only:
                workflow.save(job_id=args.collect_only, collect_only=True)
                return 0 if workflow.collect() else 1
            return workflow.run(rebuild=args.rebuild)
        except (Exception, KeyboardInterrupt) as error:
            workflow.save(last_error=str(error))
            (directory / "error.txt").write_text(traceback.format_exc())
            print(f"Workflow stopped: {error}", flush=True)
            if workflow.state.get("submission_started") and not workflow.state.get("job_id"):
                try:
                    workflow.discover()
                except Exception:
                    pass
            if workflow.state.get("job_id"):
                try:
                    workflow.collect()
                except Exception:
                    (directory / "collection-error.txt").write_text(traceback.format_exc())
            return 130 if isinstance(error, KeyboardInterrupt) else 1
        finally:
            print(f"Results and errors retained at {directory}", flush=True)
            print("Review, then stage: " + shlex.join(["git", "add", "--", str(directory.relative_to(ROOT))]), flush=True)


if __name__ == "__main__":
    sys.exit(main())
