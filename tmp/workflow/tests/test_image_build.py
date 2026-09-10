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

import fixtures

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import configure_profile
import controller
import core
import image_build


class ImageBuildTests(unittest.TestCase):
    def setUp(self) -> None:
        fixture = fixtures.DescriptionFixture()
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
