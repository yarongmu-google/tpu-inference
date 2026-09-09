"""Exercise owned-resource deletion and recovery without cloud services."""
from __future__ import annotations

from contextlib import ExitStack
import fcntl
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from urllib.parse import quote

import controller
import core
import resources
import test_image_build


class ResourceTests(unittest.TestCase):
    def setUp(self) -> None:
        fixture = test_image_build.ImageBuildTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        fixture.description['execution']['cleanup'] = 'after_collection'
        core.save(path=fixture.fixture.description_path, value=fixture.description)
        root = fixture.prepare()
        self.root = root
        self.directory = root / core.read_document(root / 'campaign.json')['jobs'][0]
        self.job = controller.Job(self.directory)
        self.image = self.job.state['owned_image']
        self.job.save(image=self.image + '@sha256:' + 'b' * 64, image_ready=True)
        with patch('sys.stdout', new=io.StringIO()):
            self.job.prepare_image()
        self.job.save(submission_started=True, bucket_creation_started=True, bucket_marked=True, bucket_created=True,
                      finished=True, exit_code=0, artifacts_verified=True, artifact_exit_code=0)
        output = self.directory / 'collected/files/diagnostics/error.txt'
        output.parent.mkdir(parents=True)
        output.write_text('saved diagnostics\n')
        self.manifest = {'format': 'run-artifacts-v1', 'run_id': self.job.state['run_id'],
            'image': self.job.state['image'], 'config_sha256': self.job.state['config_sha256'],
            'state': 'succeeded', 'exit_code': 0, 'files': [{'path': 'diagnostics/error.txt',
                'bytes': output.stat().st_size, 'sha256': core.checksum(output)}]}
        core.save(path=self.directory / 'collected/manifest.json', value=self.manifest)
        self.remote = {'job_status': 'Succeeded', 'state': 'Complete', 'id': 'j-fixture'}
        self.bucket = True
        self.image_exists = True
        self.created_at = '2026-01-01T00:00:00Z'
        self.commands = []
        self.fail_bucket_once = False
        self.fail_image_once = False
        self.deny_registry = False
        self.short_package_names = False
        self.local_tags = {'runtime:' + self.job.state['run_id'], self.image + ':run', 'unrelated:keep'}

    def gcloud(self, *, args: list[str], **kwargs) -> tuple[int, str]:
        self.commands.append(args)
        if args[:3] == ['storage', 'buckets', 'list']:
            return 0, json.dumps([{'name': self.job.state['bucket']}] if self.bucket else [])
        if args[:3] == ['storage', 'buckets', 'describe']:
            return 0, json.dumps({'name': self.job.state['bucket'], 'timeCreated': self.created_at,
                'labels': {'run_id': self.job.state['run_id']}})
        if args[:2] == ['storage', 'cat']:
            return 0, (self.directory / 'owner.json').read_text()
        if args[:2] == ['storage', 'rm']:
            self.assertEqual(args[-1], self.job.state['uri'])
            self.bucket = False
            if self.fail_bucket_once:
                self.fail_bucket_once = False
                raise RuntimeError('connection lost after bucket deletion')
            return 0, ''
        raise AssertionError(args)

    def command(self, *, args: list[str], **kwargs) -> tuple[int, str]:
        self.commands.append(args)
        if args[:6] == ['gcloud', '--project', 'fixture-project', 'artifacts', 'packages', 'list']:
            if self.deny_registry:
                raise RuntimeError('registry access denied')
            package = self.image.split('/', 3)[3]
            if self.short_package_names:
                return 0, json.dumps([{'name': 'unrelated/keep'}] +
                    ([{'name': package}] if self.image_exists else []))
            return 0, json.dumps([{'name': 'projects/fixture-project/locations/us-central1/repositories/images/packages/'
                + quote(package, safe='')}] if self.image_exists else [])
        if args[:6] == ['gcloud', '--project', 'fixture-project', 'artifacts', 'packages', 'delete']:
            self.assertTrue(args[6].endswith('/packages/' + quote(self.image.split('/', 3)[3], safe='')))
            self.assertEqual(args[7:], ['--quiet'])
            self.image_exists = False
            if self.fail_image_once:
                self.fail_image_once = False
                raise RuntimeError('connection lost after image deletion')
            return 0, ''
        if args[:3] == ['docker', 'image', 'ls']:
            return 0, '\n'.join(sorted(self.local_tags))
        if args[:3] == ['docker', 'image', 'rm']:
            self.local_tags.remove(args[3])
            return 0, ''
        raise AssertionError(args)

    def mocked(self, job=None) -> ExitStack:
        job = job or self.job
        stack = ExitStack()
        stack.enter_context(patch.object(job, 'gcloud', side_effect=self.gcloud))
        stack.enter_context(patch.object(job, 'command', side_effect=self.command))
        stack.enter_context(patch.object(job, 'discover', return_value=self.remote))
        stack.enter_context(patch('sys.stdout', new=io.StringIO()))
        return stack

    def clean(self) -> None:
        with self.mocked(), patch('builtins.input', side_effect=AssertionError('automatic cleanup must not prompt')):
            self.job.cleanup(discard=False, automatic=True)

    def test_success_deletes_owned_resources_and_keeps_results_and_other_local_images(self) -> None:
        self.clean()
        self.assertFalse(self.bucket)
        self.assertFalse(self.image_exists)
        self.assertEqual(self.local_tags, {'unrelated:keep'})
        self.assertTrue(self.job.state['resources_cleaned'])
        self.assertTrue((self.directory / 'collected/files/diagnostics/error.txt').is_file())
        self.assertEqual(core.read_document(self.directory / 'cleanup.json')['image_digest'], self.job.state['image'])
        resources.verify_collected(job=self.job)

    def test_failed_workload_is_cleaned_after_error_collection(self) -> None:
        self.remote['job_status'] = 'Failed'
        self.manifest.update(state='failed', exit_code=42)
        core.save(path=self.directory / 'collected/manifest.json', value=self.manifest)
        self.job.save(exit_code=1, artifact_exit_code=42)
        self.clean()
        self.assertTrue(self.job.state['resources_cleaned'])
        self.assertEqual(self.job.state['exit_code'], 1)
        self.assertEqual(self.job.state['artifact_exit_code'], 42)

    def test_active_or_unresolved_job_blocks_all_deletion(self) -> None:
        for status in ('Running', None):
            self.remote['job_status'] = status
            with self.assertRaisesRegex(ValueError, 'active, archiving, or unresolved'):
                self.clean()
        self.assertEqual(self.commands, [])

    def test_unverified_or_corrupted_results_block_deletion(self) -> None:
        self.job.save(artifacts_verified=False)
        with self.assertRaisesRegex(ValueError, 'not verified'):
            self.clean()
        self.job.save(artifacts_verified=True)
        (self.directory / 'collected/files/diagnostics/error.txt').write_text('corrupted')
        with self.assertRaisesRegex(ValueError, 'verification failed'):
            self.clean()
        self.assertEqual(self.commands, [])

    def test_foreign_image_path_is_never_deleted(self) -> None:
        self.job.save(owned_image='us-central1-docker.pkg.dev/fixture-project/images/other')
        with self.assertRaisesRegex(ValueError, 'ownership record differs'):
            self.clean()
        self.assertEqual(self.commands, [])

    def test_bucket_deletion_interruption_can_resume_without_owner_object(self) -> None:
        self.fail_bucket_once = True
        with self.assertRaisesRegex(RuntimeError, 'connection lost'):
            self.clean()
        self.assertTrue(self.job.state['artifacts_verified'])
        self.job = controller.Job(self.directory)
        self.clean()
        self.assertTrue(self.job.state['resources_cleaned'])
        self.assertEqual(sum(args[:2] == ['storage', 'rm'] for args in self.commands), 1)

    def test_bucket_already_removed_externally_does_not_strand_the_image(self) -> None:
        self.bucket = False
        self.clean()
        self.assertTrue(self.job.state['resources_cleaned'])

    def test_recreated_bucket_is_not_deleted_on_resume(self) -> None:
        self.job.save(cleanup_bucket_created_at=self.created_at, deletion_started=True)
        self.created_at = '2026-01-02T00:00:00Z'
        with self.assertRaisesRegex(ValueError, 'replaced'):
            self.clean()
        self.assertTrue(self.bucket)
        self.assertTrue(self.image_exists)

    def test_image_deletion_interruption_can_resume_without_recollecting(self) -> None:
        self.fail_image_once = True
        with self.assertRaisesRegex(RuntimeError, 'connection lost'):
            self.clean()
        self.assertTrue(self.job.state['deleted'])
        self.job = controller.Job(self.directory)
        with (self.mocked(), patch.object(self.job, 'collect', side_effect=AssertionError('must not recollect')),
              patch.object(self.job, 'prepare_image', side_effect=AssertionError('must not rebuild'))):
            self.assertEqual(self.job.run(), 0)
        self.assertTrue(self.job.state['resources_cleaned'])

    def test_registry_permission_failure_is_not_treated_as_absence(self) -> None:
        self.deny_registry = True
        with self.assertRaisesRegex(RuntimeError, 'access denied'):
            self.clean()
        self.assertFalse(self.job.state.get('image_deleted'))
        self.assertFalse(self.job.state.get('resources_cleaned'))

    def test_active_image_builder_lock_blocks_cleanup(self) -> None:
        directory = self.directory / 'image'
        directory.mkdir()
        with (directory / '.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                self.clean()
        self.assertEqual(self.commands, [])

    def test_manual_policy_does_not_enable_automatic_deletion(self) -> None:
        self.job.state['execution']['cleanup'] = 'manual'
        with self.assertRaisesRegex(ValueError, 'not selected'):
            self.clean()
        self.assertEqual(self.commands, [])

    def test_failed_build_without_submission_has_only_owned_images_to_clean(self) -> None:
        self.job.save(submission_started=False, bucket_creation_started=False, image=None,
                      artifacts_verified=False, finished=True, exit_code=1)
        (self.directory / 'error.txt').write_text('local build failed')
        self.bucket = False
        self.clean()
        self.assertFalse(any(args[:2] == ['storage', 'rm'] for args in self.commands))
        self.assertTrue(self.job.state['resources_cleaned'])
        self.assertEqual(self.job.state['exit_code'], 1)

    def test_multiple_cases_get_separate_resource_paths_before_building(self) -> None:
        self.fixture.description['cases'] = [{'name': 'first'}, {'name': 'second'}]
        core.save(path=self.fixture.fixture.description_path, value=self.fixture.description)
        root = self.fixture.prepare()
        folders = [root / name for name in core.read_document(root / 'campaign.json')['jobs']]
        records = [core.read_document(folder / 'resources.json') for folder in folders]
        self.assertNotEqual(records[0]['image'], records[1]['image'])
        self.assertNotEqual(records[0]['bucket'], records[1]['bucket'])
        for record in records:
            self.assertTrue(record['image'].endswith('/' + record['run_id']))
            self.assertEqual(record['bucket'], record['run_id'])
        self.assertFalse(any((folder / 'image').exists() for folder in folders))

    def test_normal_terminal_run_collects_then_cleans(self) -> None:
        self.job.save(finished=False, authorized=True, artifacts_verified=False)
        def collected() -> bool:
            self.assertTrue(self.bucket)
            self.assertTrue(self.image_exists)
            self.job.save(artifacts_verified=True, artifact_exit_code=0)
            return True
        with self.mocked(), patch.object(self.job, 'collect', side_effect=collected) as collect:
            self.assertEqual(self.job.run(), 0)
        collect.assert_called_once_with()
        self.assertTrue(self.job.state['resources_cleaned'])

    def test_collection_failure_retains_resources_then_resume_collects_and_cleans(self) -> None:
        self.job.save(finished=False, authorized=True, artifacts_verified=False)
        with self.mocked(), patch.object(self.job, 'collect', return_value=False):
            self.assertEqual(self.job.run(), 1)
        self.assertTrue(self.job.state['cleanup_pending'])
        self.assertTrue(self.bucket)
        self.assertTrue(self.image_exists)
        def collected() -> bool:
            self.job.save(artifacts_verified=True, artifact_exit_code=0)
            return True
        with self.mocked(), patch.object(self.job, 'collect', side_effect=collected):
            self.assertEqual(self.job.run(), 0)
        self.assertTrue(self.job.state['resources_cleaned'])

    def test_cleanup_failure_through_controller_preserves_verified_collection(self) -> None:
        self.fail_image_once = True
        with (patch.object(controller.Job, 'gcloud', autospec=True, side_effect=lambda job, **kwargs: self.gcloud(**kwargs)),
              patch.object(controller.Job, 'command', autospec=True, side_effect=lambda job, **kwargs: self.command(**kwargs)),
              patch.object(controller.Job, 'discover', return_value=self.remote),
              patch.object(controller.Job, 'collect', side_effect=AssertionError('must not recollect')) as collect,
              patch('sys.stdout', new=io.StringIO())):
            self.assertEqual(controller.locked_job(directory=self.directory), 1)
            state = core.read_document(self.directory / 'state.json')
            self.assertTrue(state['artifacts_verified'])
            self.assertTrue(state['deleted'])
            self.assertEqual(controller.locked_job(directory=self.directory), 0)
        collect.assert_not_called()
        self.assertTrue(core.read_document(self.directory / 'state.json')['resources_cleaned'])

    def test_build_failure_is_recorded_before_automatic_cleanup(self) -> None:
        self.fixture.hook.write_text('print("build failed", flush=True)\nraise SystemExit(42)\n')
        root = self.fixture.prepare()
        directory = root / core.read_document(root / 'campaign.json')['jobs'][0]
        def cleaned(*, job, discard, automatic):
            self.assertFalse(job.state.get('submission_started'))
            self.assertTrue((directory / 'error.txt').is_file())
            self.assertIn('build failed', (directory / 'image/command.log').read_text())
            self.assertFalse(discard)
            self.assertTrue(automatic)
            return 0
        with patch('resources.cleanup', side_effect=cleaned) as cleanup, patch('sys.stdout', new=io.StringIO()):
            self.assertEqual(controller.locked_job(directory=directory), 1)
        cleanup.assert_called_once()

    def test_published_result_recovers_without_a_second_build(self) -> None:
        self.job.save(image_build_started=True, image_ready=False)
        directory = self.directory / 'image'
        directory.mkdir()
        core.save(path=directory / 'result.json', value={'image': self.job.state['image'], 'owner_run_id': self.job.state['run_id']})
        with patch('image_build.resolve', side_effect=AssertionError('must not rebuild')):
            self.job.prepare_image()
        self.assertTrue(self.job.state['image_ready'])

    def test_completed_cleanup_is_idempotent(self) -> None:
        self.clean()
        self.commands.clear()
        self.clean()
        self.assertEqual(self.commands, [])


    def test_cleanup_accepts_short_package_names_and_deletes_only_owned_package(self) -> None:
        self.short_package_names = True
        self.clean()
        self.assertTrue(self.job.state['resources_cleaned'])
        deletes = [args for args in self.commands if args[:6] ==
                   ['gcloud', '--project', 'fixture-project', 'artifacts', 'packages', 'delete']]
        self.assertEqual(len(deletes), 1)
        self.assertEqual(deletes[0][6],
            'projects/fixture-project/locations/us-central1/repositories/images/packages/' +
            quote(self.image.split('/', 3)[3], safe=''))
        self.assertEqual(self.local_tags, {'unrelated:keep'})

    def test_package_name_forms_match_exactly_in_configured_repository(self) -> None:
        package = self.image.split('/', 3)[3]
        prefix = 'projects/fixture-project/locations/us-central1/repositories/images/packages/'
        for name in (package, quote(package, safe=''), prefix + package, prefix + quote(package, safe='')):
            with self.subTest(name=name), patch.object(self.job, 'command', return_value=(0,
                    json.dumps([{'name': package + '-other'}, {'name': name}]))):
                self.assertEqual(resources.image_package(job=self.job, image=self.image),
                                 prefix + quote(package, safe=''))

    def test_missing_owned_package_in_short_listing_is_absent(self) -> None:
        with patch.object(self.job, 'command', return_value=(0, json.dumps([{'name': 'unrelated/keep'}]))):
            self.assertIsNone(resources.image_package(job=self.job, image=self.image))

    def test_foreign_duplicate_and_malformed_package_listings_are_rejected(self) -> None:
        package = self.image.split('/', 3)[3]
        prefix = 'projects/fixture-project/locations/us-central1/repositories/images/packages/'
        listings = [
            [{'name': prefix.replace('fixture-project', 'another-project') + quote(package, safe='')}],
            [{'name': package}, {'name': prefix + quote(package, safe='')}],
            [{'name': None}], [None], {},
        ]
        for rows in listings:
            with self.subTest(rows=rows), patch.object(self.job, 'command', return_value=(0, json.dumps(rows))):
                with self.assertRaises(ValueError):
                    resources.image_package(job=self.job, image=self.image)


if __name__ == '__main__':
    unittest.main()
