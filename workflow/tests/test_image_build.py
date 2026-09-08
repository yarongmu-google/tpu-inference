"""Check per-run builds, publication records, and failure retention locally."""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import unittest
from unittest.mock import patch
from contextlib import ExitStack
from types import SimpleNamespace

import test_sweep_integration

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tmp'))
import configure_profile
import controller
import core
import image_build
import prepare_runtime_image


class ImageBuildTests(unittest.TestCase):
    def setUp(self) -> None:
        fixture = test_sweep_integration.SweepIntegrationTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.base = fixture.base
        self.repository = 'us-central1-docker.pkg.dev/fixture-project/images/runtime'
        self.run_id = 'fixture-' + 'a' * 24
        self.image = self.repository + '/' + self.run_id + '@sha256:' + 'b' * 64
        self.profile = {**fixture.profile, 'runtime': {'repository': self.repository}}
        self.profile_path = fixture.workflow / 'local/environment.json'
        core.save(path=self.profile_path, value=self.profile)
        self.hook = self.base / 'image hook.py'
        self.hook.write_text('import json, os\nfrom pathlib import Path\n'
            'print("image hook diagnostics", flush=True)\n'
            'Path(os.environ["IMAGE_RESULT"]).write_text(json.dumps({"image":os.environ["IMAGE_REPOSITORY"]+"@sha256:"+"b"*64, "owner_run_id":os.environ["IMAGE_RUN_ID"], "source_revisions":{"app":"fixture"}}))\n')
        self.description = {**fixture.description, 'image_build': {
            'cwd': '..', 'argv': [sys.executable, str(self.hook)], 'timeout_seconds': 10}}
        core.save(path=fixture.description_path, value=self.description)
        controller.STOP.clear()

    def prepare(self, dry_run: bool = False) -> Path:
        with patch('sys.stdout', new=io.StringIO()):
            return controller.prepare(description_path=self.fixture.description_path, name='image check', dry_run=dry_run)

    def test_real_hook_digest_reaches_saved_jobs_and_profile_stays_reusable(self) -> None:
        before = self.profile_path.read_bytes()
        root = self.prepare()
        campaign = core.read_document(root / 'campaign.json')
        folder = root / campaign['jobs'][0]
        with patch('sys.stdout', new=io.StringIO()):
            controller.Job(folder).prepare_image()
        self.image = core.read_document(folder / 'resources.json')['image'] + '@sha256:' + 'b' * 64
        self.assertEqual(core.read_document(folder / 'state.json')['image'], self.image)
        self.assertEqual(core.read_document(folder / 'input/run.json')['image'], self.image)
        self.assertEqual(core.read_document(folder / 'image-build.json')['source_revisions'], {'app': 'fixture'})
        self.assertIn(self.image, (folder / 'recipe.json').read_text())
        self.assertIn('image hook diagnostics', (folder / 'image/command.log').read_text())
        self.assertEqual(self.profile_path.read_bytes(), before)

    def test_failed_hook_retains_logs_without_preparing_submission(self) -> None:
        self.hook.write_text('print("build broke", flush=True)\nraise SystemExit(42)\n')
        root = self.prepare()
        folder = root / core.read_document(root / 'campaign.json')['jobs'][0]
        with patch('sys.stdout', new=io.StringIO()), self.assertRaisesRegex(RuntimeError, '42'):
            controller.Job(folder).prepare_image()
        self.assertEqual(core.read_document(folder / 'image/status.json')['phase'], 'failed')
        self.assertIn('build broke', (folder / 'image/command.log').read_text())
        self.assertFalse(core.read_document(folder / 'state.json').get('submission_started'))
        self.assertTrue((folder / 'resources.json').exists())
        self.assertTrue((root / 'campaign.json').exists())

    def test_dry_run_does_not_invoke_builder_or_invent_an_image(self) -> None:
        self.hook.write_text('raise AssertionError("must not run")\n')
        root = self.prepare(dry_run=True)
        campaign = core.read_document(root / 'campaign.json')
        self.assertTrue(campaign['dry_run'])
        folder = root / campaign['jobs'][0]
        self.assertIsNone(core.read_document(folder / 'state.json')['image'])
        self.assertFalse((folder / 'image').exists())
        self.assertNotIn('image', core.read_document(root / 'profile.json')['runtime'])

    def test_digest_from_wrong_repository_blocks_job_preparation(self) -> None:
        self.hook.write_text(self.hook.read_text().replace('os.environ["IMAGE_REPOSITORY"]', '"other/repo"'))
        root = self.prepare()
        folder = root / core.read_document(root / 'campaign.json')['jobs'][0]
        with patch('sys.stdout', new=io.StringIO()), self.assertRaisesRegex(ValueError, 'configured repository'):
            controller.Job(folder).prepare_image()

    def test_stop_cancels_image_preparation_and_records_failure(self) -> None:
        self.hook.write_text('import time\ntime.sleep(60)\n')
        stop = threading.Event()
        stop.set()
        with patch('sys.stdout', new=io.StringIO()), self.assertRaisesRegex(RuntimeError, 'interrupted'):
            image_build.resolve(build={**self.description['image_build'], 'cwd': str(self.base)},
                                repository=self.repository, directory=self.base / 'cancelled', stop=stop, run_id=self.run_id)
        self.assertEqual(core.read_document(self.base / 'cancelled/status.json')['phase'], 'failed')

    def test_timeout_preserves_diagnostics_without_a_result(self) -> None:
        self.hook.write_text('import time\nprint("waiting", flush=True)\ntime.sleep(60)\n')
        with patch('sys.stdout', new=io.StringIO()), self.assertRaises(TimeoutError):
            image_build.resolve(build={**self.description['image_build'], 'cwd': str(self.base), 'timeout_seconds': 1},
                                repository=self.repository, directory=self.base / 'timeout', stop=threading.Event(), run_id=self.run_id)
        self.assertIn('waiting', (self.base / 'timeout/command.log').read_text())
        self.assertFalse((self.base / 'timeout/result.json').exists())

    def test_initial_setup_asks_for_repository_not_digest(self) -> None:
        self.profile_path.unlink()
        answers = ['fixture-project', 'us-central1', self.repository, 'fixture-worker',
                   'serviceAccount:fixture@example.invalid', '', '', '', 'yes']
        with patch('sys.stdin.isatty', return_value=True), patch('builtins.input', side_effect=answers) as prompts, patch('sys.stdout', new=io.StringIO()):
            configure_profile.configure(description=self.fixture.description_path, source=None)
        self.assertEqual(core.read_document(self.profile_path), self.profile)
        self.assertTrue(any('Runtime image repository' in call.args[0] for call in prompts.call_args_list))
        self.assertFalse(any('Published runtime image digest' in call.args[0] for call in prompts.call_args_list))

    def test_existing_fixed_image_profile_gets_reviewed_repository_migration(self) -> None:
        core.save(path=self.profile_path, value=self.fixture.profile)
        with patch('sys.stdin.isatty', return_value=True), patch('builtins.input', side_effect=['', 'yes']), patch('sys.stdout', new=io.StringIO()):
            configure_profile.configure(description=self.fixture.description_path, source=None)
        value = core.read_document(self.profile_path)
        self.assertEqual(value['runtime']['repository'], self.repository)
        self.assertEqual(value['cloud'], self.profile['cloud'])

    def test_input_hash_changes_for_native_environment_and_build_files(self) -> None:
        env = self.base / 'environment'
        env.mkdir()
        library = env / 'library.so'
        library.write_bytes(b'original')
        original = prepare_runtime_image.fingerprint({'environment': prepare_runtime_image.tree(env)})
        library.write_bytes(b'changed!')
        self.assertNotEqual(original, prepare_runtime_image.fingerprint({'environment': prepare_runtime_image.tree(env)}))
        library.write_bytes(b'original')
        self.assertEqual(original, prepare_runtime_image.fingerprint({'environment': prepare_runtime_image.tree(env)}))
        library.chmod(0o755)
        self.assertNotEqual(original, prepare_runtime_image.fingerprint({'environment': prepare_runtime_image.tree(env)}))

    def builder_fixture(self, stack: ExitStack) -> tuple[list, dict]:
        prefix = self.base / 'prefix'
        (prefix / 'conda-meta').mkdir(parents=True)
        client = self.base / 'client'
        client.mkdir()
        cache = self.base / 'cache'
        cache.mkdir()
        value = {'sources': {name: {'revision': 'fixture'} for name in ('vllm', 'tpu_inference', 'InferenceX')},
                 'environment': 'initial'}
        commands = []
        def command(argv: list[str], *, capture: bool = False, check: bool = True, env=None):
            commands.append(argv)
            output = ''
            if argv[:6] == ['gcloud', '--project', 'fixture-project', 'artifacts', 'repositories', 'describe']:
                output = 'DOCKER'
            elif argv[0] == 'bash':
                build = Path(env['JOBSET_BUILD_DIAGNOSTICS'])
                build.mkdir()
                (build / 'image-id.txt').write_text('sha256:' + 'c' * 64)
                core.save(path=build / 'image-source.json', value={'source_revisions': {
                    name: 'fixture' for name in ('vllm', 'tpu_inference', 'InferenceX')}})
            elif argv[:3] == ['docker', 'image', 'inspect']:
                output = 'sha256:' + 'c' * 64 if argv[4] == '{{.Id}}' else json.dumps([argv[-1].removesuffix(':run') + '@sha256:' + 'b' * 64])
            return subprocess.CompletedProcess(args=argv, returncode=0, stdout=output)
        for target, replacement in [
            ('platform.system', lambda: 'Linux'), ('platform.machine', lambda: 'x86_64'),
            ('sys.prefix', str(prefix)), ('sys.version_info', (3, 12)),
            ('importlib.util.find_spec', lambda name: SimpleNamespace(origin=str(self.base / 'vllm/__init__.py'))),
            ('snapshot.git', lambda *args: str(self.base / 'vllm')),
            ('inventory', lambda **kwargs: json.loads(json.dumps(value))), ('run', command), ('CACHE', cache)]:
            stack.enter_context(patch('prepare_runtime_image.' + target, replacement))
        stack.enter_context(patch.dict(os.environ, {'CONDA_DEFAULT_ENV': 'vllm12', 'INFERENCEX_REPO': str(client)}))
        stack.enter_context(patch('sys.stdout', new=io.StringIO()))
        return commands, value

    def test_builder_always_builds_identical_inputs_under_distinct_run_paths(self) -> None:
        with ExitStack() as stack:
            commands, inputs = self.builder_fixture(stack)
            for index in range(3):
                diagnostics = self.base / f'build-{index}'
                diagnostics.mkdir()
                run_id = 'fixture-' + f'{index:024x}'
                destination = self.repository + '/' + run_id
                value = prepare_runtime_image.prepare(repository=destination, diagnostics=diagnostics, run_id=run_id)
                self.assertEqual(value['image'], destination + '@sha256:' + 'b' * 64)
                self.assertEqual(value['owner_run_id'], run_id)
            self.assertEqual(sum(argv[0] == 'bash' for argv in commands), 3)
            self.assertEqual(sum(argv[:2] == ['docker', 'push'] for argv in commands), 3)

    def test_concurrent_input_change_blocks_publication(self) -> None:
        with ExitStack() as stack:
            commands, before = self.builder_fixture(stack)
            stack.enter_context(patch('prepare_runtime_image.inventory', side_effect=[before, {**before, 'environment': 'changed'}]))
            diagnostics = self.base / 'racing-build'
            diagnostics.mkdir()
            with self.assertRaisesRegex(RuntimeError, 'inputs changed'):
                prepare_runtime_image.prepare(repository=self.repository + '/' + self.run_id, diagnostics=diagnostics, run_id=self.run_id)
            self.assertFalse(any(argv[:2] == ['docker', 'push'] for argv in commands))
            self.assertFalse(list((self.base / 'cache').glob('*.json')))

    def test_build_failure_does_not_publish(self) -> None:
        with ExitStack() as stack:
            self.builder_fixture(stack)
            failed = stack.enter_context(patch('prepare_runtime_image.run', side_effect=subprocess.CalledProcessError(42, ['bash'])))
            diagnostics = self.base / 'failed-build'
            diagnostics.mkdir()
            with self.assertRaises(subprocess.CalledProcessError):
                prepare_runtime_image.prepare(repository=self.repository + '/' + self.run_id, diagnostics=diagnostics, run_id=self.run_id)
            self.assertEqual(failed.call_count, 1)
            self.assertFalse(list((self.base / 'cache').glob('*.json')))

    def test_uncommitted_and_untracked_application_code_are_rejected(self) -> None:
        repo = self.base / 'source'
        repo.mkdir()
        def git(*args: str) -> None:
            subprocess.run(args=['git', '-C', str(repo), *args], check=True, capture_output=True)
        git('init')
        (repo / 'app.py').write_text('x = 1\n')
        git('add', 'app.py')
        git('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-m', 'Initial fixture.')
        original = prepare_runtime_image.sources(root=repo, keep=lambda p: p.endswith('.py'))
        (repo / 'new.py').write_text('x = 2\n')
        with self.assertRaisesRegex(RuntimeError, 'Commit source changes'):
            prepare_runtime_image.sources(root=repo, keep=lambda p: p.endswith('.py'))
        (repo / 'new.py').unlink()
        (repo / 'app.py').write_text('x = 3\n')
        with self.assertRaisesRegex(RuntimeError, 'Commit source changes'):
            prepare_runtime_image.sources(root=repo, keep=lambda p: p.endswith('.py'))
        (repo / 'app.py').write_text('x = 1\n')
        self.assertEqual(original, prepare_runtime_image.sources(root=repo, keep=lambda p: p.endswith('.py')))


if __name__ == '__main__':
    unittest.main()
