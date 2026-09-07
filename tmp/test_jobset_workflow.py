"""Local protocol/failure tests; never invokes a cloud service or Docker."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import shlex
import shutil
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import jobset_workflow as workflow

DIGEST = "example/image@sha256:" + "a" * 64


def bundle(root: Path, succeeded: bool = True) -> None:
    root.mkdir(parents=True, exist_ok=True)
    content = b'{"completed": 2560}\n'
    (root / "client").mkdir(exist_ok=True)
    (root / "client/result.json").write_bytes(content)
    (root / "manifest.json").write_text(json.dumps({
        "format": "sweep-jobset-v1", "attempt": "fixture", "image": DIGEST,
        "state": "succeeded" if succeeded else "failed", "exit_code": 0 if succeeded else 42,
        "files": [{"path": "client/result.json", "bytes": len(content),
                   "sha256": hashlib.sha256(content).hexdigest()}]}))


class Protocol(workflow.Workflow):
    def __init__(self, directory: Path, state: dict) -> None:
        super().__init__(directory=directory, state=state)
        self.calls = []
        self.visible = False
        self.create_code = 0
        self.job_status = "Succeeded"
        self.cd_state = "Complete"
        self.fail_sync = False
        self.corrupt = False
        self.interrupt = False
        self.archives = []
        self.syncs = 0

    def register_recipe(self) -> None:
        self.save(recipe="fixture-recipe")

    def prepare_image(self, rebuild: bool) -> None:
        self.save(image=DIGEST)

    def cdk(self, args: list[str], check: bool = True, timeout: int = 180) -> tuple[int, str]:
        self.calls.append(args)
        if args[-1] == "--help":
            return 0, "fixture help"
        if args[:2] == ["job", "create"]:
            self.visible = True
            if self.interrupt:
                raise KeyboardInterrupt("after remote acceptance")
            return self.create_code, "Submission acknowledged (ID recovered via JSON)"
        if args[:2] == ["job", "list"]:
            if not self.visible:
                return 0, "[]"
            state = self.archives.pop(0) if self.archives else self.cd_state
            return 0, json.dumps([{"id": "j-fixture", "user": "fixture", "recipe": "fixture-recipe",
                                   "tags": [self.state['tag']], "job_status": self.job_status, "state": state}])
        if args[:2] == ["job", "recipe"]:
            return 0, DIGEST + "\n" + self.state["script"]
        if args[:2] == ["job", "desc"]:
            return 0, "fixture scheduling diagnostics"
        if args[:2] == ["job", "log"]:
            return 0, "https://console.cloud.google.com/logs/query; fixture fallback"
        if args[:2] == ["job", "sync-outputs"]:
            self.syncs += 1
            destination = self.outputs_root / "j-fixture" / "outputs/attempts/fixture"
            bundle(root=destination, succeeded=self.job_status == "Succeeded")
            if self.corrupt:
                (destination / "client/result.json").write_text("truncated")
            return (1 if self.fail_sync else 0), "fixture sync diagnostics"
        raise AssertionError(args)


class WorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="jobset-protocol-test-")
        self.base = Path(self.temp.name)
        self.directory = self.base / "workflow"
        self.directory.mkdir()
        self.environment = patch.dict(os.environ, {"CDK_SOURCE_DIR": str(self.base / "cdk"),
                                                  "CDK_JOB_OUTPUTS_DIR": str(self.base / "outputs")})
        self.environment.start()
        self.sleep = patch.object(workflow.time, "sleep")
        self.sleep.start()
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.state = {"user": "fixture", "user_slug": "fixture", "tag": "fixture-tag", "script": "tmp/sweep.sh"}

    def tearDown(self) -> None:
        self.output.__exit__(None, None, None)
        self.sleep.stop()
        self.environment.stop()
        self.temp.cleanup()

    def test_success_waits_for_archiving_and_rerun_does_not_submit(self) -> None:
        run = Protocol(directory=self.directory, state=self.state)
        run.archives = ["ExecutionComplete", "LogArchiveSucceeded", "Complete"]
        self.assertEqual(run.run(rebuild=False), 0)
        self.assertTrue(run.state["artifacts_verified"])
        self.assertEqual(run.state["last_job"]["state"], "Complete")
        self.assertEqual(run.run(rebuild=False), 0)
        self.assertEqual(sum(a[:2] == ["job", "create"] and "--help" not in a for a in run.calls), 1)
        self.assertTrue((Path(run.state["collection"]) / "description.yml").exists())

    def test_failed_job_retains_verified_failure_bundle(self) -> None:
        run = Protocol(directory=self.directory, state=self.state)
        run.job_status = "Failed"
        self.assertEqual(run.run(rebuild=False), 1)
        self.assertTrue(run.state["finished"])
        self.assertTrue(run.state["artifacts_verified"])
        self.assertFalse(run.state["artifacts_succeeded"])
        self.assertTrue(list(self.directory.rglob("manifest.json")))

    def test_interrupted_submission_is_recovered_without_duplicate(self) -> None:
        run = Protocol(directory=self.directory, state=self.state)
        run.interrupt = True
        with self.assertRaises(KeyboardInterrupt):
            run.run(rebuild=False)
        self.assertTrue(json.loads((self.directory / "state.json").read_text())["submission_started"])
        self.assertNotIn("job_id", run.state)
        self.assertEqual(run.run(rebuild=False), 0)
        self.assertEqual(sum(a[:2] == ["job", "create"] and "--help" not in a for a in run.calls), 1)

    def test_nonzero_create_that_accepted_job_is_not_retried(self) -> None:
        run = Protocol(directory=self.directory, state=self.state)
        run.create_code = 1
        self.assertEqual(run.run(rebuild=False), 0)
        self.assertEqual(run.state["submission_exit"], 1)
        self.assertEqual(sum(a[:2] == ["job", "create"] and "--help" not in a for a in run.calls), 1)

    def test_unknown_submission_never_calls_create_again(self) -> None:
        self.state.update(submission_started=True, recipe="fixture-recipe", image=DIGEST)
        run = Protocol(directory=self.directory, state=self.state)
        with self.assertRaisesRegex(RuntimeError, "outcome unknown"):
            run.run(rebuild=False)
        self.assertFalse(any(a[:2] == ["job", "create"] and "--help" not in a for a in run.calls))

    def test_corrupt_sync_retries_and_keeps_each_partial_copy(self) -> None:
        self.state.update(job_id="j-fixture", image=DIGEST)
        run = Protocol(directory=self.directory, state=self.state)
        run.corrupt = True
        self.assertFalse(run.collect())
        self.assertEqual(run.syncs, 3)
        self.assertEqual(len(list(Path(run.state["collection"]).glob("artifacts-*"))), 3)
        self.assertFalse(run.state["artifacts_verified"])

    def test_nonzero_sync_does_not_report_success_from_cached_files(self) -> None:
        self.state.update(job_id="j-fixture", image=DIGEST)
        run = Protocol(directory=self.directory, state=self.state)
        run.fail_sync = True
        self.assertFalse(run.collect())
        self.assertEqual(run.syncs, 3)
        self.assertTrue(list(self.directory.rglob("client/result.json")))

    def test_wrong_image_and_traversal_rejected(self) -> None:
        data = self.base / "bundle"
        bundle(root=data)
        with self.assertRaisesRegex(ValueError, "different image"):
            workflow.verify_artifacts(root=data, expected_image="wrong")
        manifest = json.loads((data / "manifest.json").read_text())
        manifest["files"][0]["path"] = "../outside"
        (data / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "Invalid artifact path"):
            workflow.verify_artifacts(root=data, expected_image=DIGEST)

    def test_registration_preserves_other_recipes_and_is_idempotent(self) -> None:
        run = workflow.Workflow(directory=self.directory, state=self.state)
        root = self.base / "project"
        (root / "tmp").mkdir(parents=True)
        (root / "tmp/jobset.yml").write_text("fixture template")
        run.cdk_root.mkdir()
        registry = run.cdk_root / "recipes.yml"
        before = "# shared recipes\n- name: unrelated\n  owner: colleague\n"
        registry.write_text(before)
        def cdk(args, **kwargs):
            self.assertEqual(args, ["recipe", "list", "-o", "json"])
            rows = [{"name": "unrelated"}]
            rows.extend(json.loads(line[2:]) for line in registry.read_text().splitlines() if line.startswith('- {'))
            return 0, json.dumps(rows)
        run.cdk = cdk
        with patch.object(workflow, "ROOT", root):
            run.register_recipe()
            first = registry.read_bytes()
            run.register_recipe()
        self.assertTrue(first.startswith(before.encode()))
        self.assertEqual(registry.read_bytes(), first)

    def test_command_errors_and_timeouts_save_status_and_stderr(self) -> None:
        run = workflow.Workflow(directory=self.directory, state=self.state)
        with self.assertRaisesRegex(RuntimeError, "exited 42"):
            run.command(args=[sys.executable, "-c", "import sys; print('fixture failure',file=sys.stderr); sys.exit(42)"])
        self.assertEqual(next((self.directory / "commands").glob("*.exit")).read_text().strip(), "42")
        self.assertIn("fixture failure", next((self.directory / "commands").glob("*.stderr")).read_text())
        with self.assertRaises(subprocess.TimeoutExpired):
            run.command(args=[sys.executable, "-c", "import time; time.sleep(30)"], timeout=1)
        self.assertIn("124\n", [p.read_text() for p in (self.directory / "commands").glob("*.exit")])

    def test_changed_letter_blocks_actual_cdk_command(self) -> None:
        run = workflow.Workflow(directory=self.directory, state=self.state)
        seen = []
        def command(args, **kwargs):
            seen.append(args)
            return 0, "Changed CDK instructions"
        run.command = command
        with self.assertRaisesRegex(RuntimeError, "instructions changed"):
            run.cdk(args=["job", "create", "fixture"])
        self.assertEqual(seen, [["cdk", "agent-letter"]])

    def test_failed_publication_stops_before_submission(self) -> None:
        run = Protocol(directory=self.directory, state=self.state)
        run.cached_image = lambda: "local-fixture"
        run.state["image_repository"] = "us-central1-docker.pkg.dev/project/repo/image"
        seen = []
        def command(args, **kwargs):
            seen.append(args)
            if args[:2] == ["docker", "push"]:
                raise RuntimeError("registry denied upload")
            return 0, ""
        run.command = command
        run.prepare_image = lambda rebuild: workflow.Workflow.prepare_image(run, rebuild=rebuild)
        with self.assertRaisesRegex(RuntimeError, "registry denied"):
            run.run(rebuild=False)
        self.assertFalse(run.state.get("submission_started", False))
        self.assertFalse(any(a[:2] == ["job", "create"] and "--help" not in a for a in run.calls))


    def test_changed_source_invalidates_cached_image(self) -> None:
        root = self.base / "project"
        (root / "tmp/vllm_logs/jobset-build-fixture").mkdir(parents=True)
        (root / "tmp/prepare_jobset_image.py").write_text(
            'def application_file(path): return path.startswith("tpu_inference/") or path == "tmp/jobset_bootstrap.sh"\n')
        folder = root / "tmp/vllm_logs/jobset-build-fixture"
        metadata = {"source_revisions": {"vllm": "a" * 40, "tpu_inference": "b" * 40}}
        (folder / "image-source.json").write_text(json.dumps(metadata))
        (folder / "exit-code.txt").write_text("0\n")
        (folder / "image-tag.txt").write_text("vllm12:topk-bbbbbbbbbbbb\n")
        run = workflow.Workflow(directory=self.directory, state=self.state)
        changes = "tmp/vllm_logs/new.log\n"
        def command(args, **kwargs):
            if args[:2] == ["git", "-C"]:
                return 0, "a" * 40 if args[-2:] == ["rev-parse", "HEAD"] else ""
            if args[:2] == ["git", "diff"]:
                return 0, changes
            if args[:3] == ["docker", "image", "inspect"]:
                return 0, "sha256:fixture"
            if args[:2] == ["docker", "run"]:
                self.assertIn("--pull=never", args)
                return 0, json.dumps(metadata)
            raise AssertionError(args)
        run.command = command
        with patch.object(workflow, "ROOT", root), patch.object(workflow.importlib.util, "find_spec", return_value=SimpleNamespace(origin="/fixture/vllm/__init__.py")):
            self.assertEqual(run.cached_image(), "vllm12:topk-bbbbbbbbbbbb")
            changes = "tmp/jobset_bootstrap.sh\n"
            self.assertIsNone(run.cached_image())

    def test_startup_does_not_install_and_build_install_failure_stops_script(self) -> None:
        root = self.base / "image"
        (root / "tmp").mkdir(parents=True)
        shutil.copyfile(src=Path(__file__).parent / "jobset_bootstrap.sh", dst=root / "tmp/jobset_bootstrap.sh")
        vllm = self.base / "vllm"
        vllm.mkdir()
        for name, source in (("vllm", vllm), ("tpu_inference", root)):
            (source / "setup.py").write_text("# fixture\n")
            (source / name).mkdir()
            (source / name / "__init__.py").write_text('__version__ = "fixture"\n')
            dist = source / (name + "-1.0.dist-info")
            dist.mkdir()
            (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: " + name + "\nVersion: 1.0\n")
        (root / "tmp/bench.sh").write_text('echo "fixture benchmark started"\n')
        binary = self.base / "bin"
        binary.mkdir()
        (binary / "python3").write_text('#!/bin/bash\nif [[ "$1" == -m && "$2" == pip ]]; then echo "fixture install"; exit "${INSTALL_EXIT:-0}"; fi\nexec ' + shlex.quote(sys.executable) + ' "$@"\n')
        (binary / "python3").chmod(0o755)
        env = {**os.environ, "PATH": str(binary) + ":" + os.environ["PATH"],
               "PYTHONPATH": str(vllm) + ":" + str(root), "VLLM_SOURCE_DIR": str(vllm),
               "RUN_METADATA_DIR": str(self.directory)}
        for install, failure, expected in (("0", "0", 0), ("1", "0", 0), ("1", "42", 42)):
            result = subprocess.run(args=["bash", str(root / "tmp/jobset_bootstrap.sh"), str(root / "tmp/bench.sh")],
                                    env={**env, "JOBSET_INSTALL_SOURCES": install, "INSTALL_EXIT": failure},
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
            self.assertEqual(result.stdout.count("fixture install"), 0 if install == "0" else (1 if expected else 2))
            self.assertEqual("fixture benchmark started" in result.stdout, expected == 0)
        metadata = json.loads((self.directory / "installed-sources.json").read_text())
        self.assertTrue(metadata["packages"]["vllm"]["import_path"].startswith(str(vllm.resolve())))

    def test_fresh_letter_precedes_each_command_and_codes_keep_order(self) -> None:
        run = workflow.Workflow(directory=self.directory, state=self.state)
        letter = "Fixture rules\n" + "\n".join(f"[ack {i}/10: CODE{i}]" for i in range(1, 11))
        normal = " ".join(workflow.re.sub(r"\[ack \d+/\d+: [A-Za-z0-9]+\]", "[ack]", letter).split())
        seen = []
        def command(args, **kwargs):
            seen.append(args)
            return 0, letter if args == ["cdk", "agent-letter"] else "[]"
        run.command = command
        with patch.object(workflow, "LETTER_HASH", hashlib.sha256(normal.encode()).hexdigest()):
            run.cdk(args=["job", "list", "-o", "json"])
            run.cdk(args=["recipe", "list", "-o", "json"])
        self.assertEqual(seen[0], ["cdk", "agent-letter"])
        self.assertEqual(seen[2], ["cdk", "agent-letter"])
        self.assertEqual(seen[1][1], "--agent-code=" + "-".join(f"CODE{i}" for i in range(1, 11)))

    def test_archive_failure_collects_outputs_but_returns_nonzero(self) -> None:
        run = Protocol(directory=self.directory, state=self.state)
        run.cd_state = "LogArchiveFailed"
        self.assertEqual(run.run(rebuild=False), 1)
        self.assertTrue(run.state["artifacts_verified"])
        self.assertTrue(list(self.directory.rglob("client/result.json")))

    def test_resume_bundle_copies_only_clients_and_requires_same_image(self) -> None:
        root = self.base / "project"
        (root / "tmp").mkdir(parents=True)
        shutil.copyfile(src=Path(__file__).parent / "jobset_runner.py", dst=root / "tmp/jobset_runner.py")
        previous = self.base / "previous"
        bundle(root=previous, succeeded=False)
        (previous / "server.log").write_text("unneeded old server output")
        self.state.update(image=DIGEST, resume_from=str(previous))
        run = workflow.Workflow(directory=self.directory, state=self.state)
        with patch.object(workflow, "ROOT", root):
            mapping = run.resume_bundle()
            self.assertEqual(mapping[0], "--map-dir")
            self.assertTrue(mapping[1].endswith(":tpu-worker:/opt/jobset-resume"))
            self.assertTrue((self.directory / "resume-input/client/result.json").exists())
            self.assertFalse((self.directory / "resume-input/server.log").exists())
            run.state["image"] = "wrong"
            with self.assertRaisesRegex(ValueError, "different image"):
                run.resume_bundle()

if __name__ == "__main__":
    unittest.main()
