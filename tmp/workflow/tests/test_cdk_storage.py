"""Exercise CDK-mounted storage without contacting cloud services."""
from __future__ import annotations

from contextlib import ExitStack
import copy
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import controller
import core
import test_resources


class CdkStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = test_resources.ResourceTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.job = self.fixture.job
        self.directory = self.fixture.directory
        profile = self.job.state['profile']
        profile['cloud'] = {'project': 'fixture-project'}
        profile['storage'] = {'mode': 'cdk', 'outputs_root': 'gs://fixture-jobs/jobs'}
        self.job.save(profile=profile, bucket=None, job_id='j-fixture', cdk_upload_started=True)
        self.job.save(uri=core.cdk_storage_uri(self.job.state))
        record = core.read_document(self.directory / 'resources.json')
        record['bucket'] = None
        core.save(path=self.directory / 'resources.json', value=record)
        self.commands = []
        self.fail_once = False

    def gcloud(self, *, args: list[str], **kwargs) -> tuple[int, str]:
        self.commands.append(args)
        if args[:2] == ['storage', 'cat']:
            self.assertEqual(args[2], self.job.state['uri'] + '/owner.json')
            return 0, (self.directory / 'owner.json').read_text()
        if args[:2] == ['storage', 'rsync']:
            self.assertEqual(args[-1], self.job.state['uri'] + '/')
            self.assertEqual(args[2:4], ['--recursive', '--delete-unmatched-destination-objects'])
            self.assertEqual(list(Path(args[-2]).iterdir()), [])
            if self.fail_once:
                self.fail_once = False
                raise RuntimeError('connection lost after prefix deletion')
            return 0, ''
        raise AssertionError(args)

    def mocked(self) -> ExitStack:
        stack = ExitStack()
        stack.enter_context(patch.object(self.job, 'gcloud', side_effect=self.gcloud))
        stack.enter_context(patch.object(self.job, 'command', side_effect=self.fixture.command))
        stack.enter_context(patch.object(self.job, 'discover', return_value=self.fixture.remote))
        stack.enter_context(patch('sys.stdout', new=io.StringIO()))
        return stack

    def test_sweep_preset_has_no_identity_or_bucket_questionnaire(self) -> None:
        profile = core.read_document(controller.ROOT / 'sweep-environment.yml')
        self.assertEqual(set(profile['cloud']), {'project'})
        self.assertTrue(core.uses_cdk_storage(profile))
        self.assertEqual(profile['runtime']['repository'],
            'us-central1-docker.pkg.dev/cloud-tpu-inference-test/vllm-tpu-rdna/runtime')
        fixture = self.fixture.fixture.fixture
        fixture.profile = profile
        fixture.saved_profile()
        core.load_description(fixture.description_path)

    def test_recipe_accepts_assigned_account_but_rejects_changed_image(self) -> None:
        expected = controller.recipe(state=self.job.state, payload={'runtime.py': '# fixture'})
        actual = copy.deepcopy(expected)
        pod = actual['spec']['replicatedJobs'][0]['template']['spec']['template']['spec']
        self.assertNotIn('serviceAccountName', pod)
        self.assertEqual([v['name'] for v in pod['volumes']], ['shared-memory'])
        pod['serviceAccountName'] = 'cdk-assigned-worker'
        controller.validate_recipe(actual=actual, expected=expected, service_account=None)
        pod['containers'][0]['image'] = 'foreign-image'
        with self.assertRaisesRegex(ValueError, 'image differs'):
            controller.validate_recipe(actual=actual, expected=expected, service_account=None)

    def test_string_resource_counts_allow_existing_job_authorization(self) -> None:
        expected = controller.recipe(state=self.job.state, payload={'runtime.py': '# fixture'})
        core.save(path=self.directory / 'recipe.json', value=expected)
        actual = copy.deepcopy(expected)
        pod = actual['spec']['replicatedJobs'][0]['template']['spec']['template']['spec']
        for quantities in pod['containers'][0]['resources'].values():
            for key, count in quantities.items():
                quantities[key] = str(count)
        from test_end_to_end import mounted_recipe
        core.save(path=self.directory / 'recipe.json', value=actual)
        actual = mounted_recipe(job=self.job, mount=core.CDK_MOUNT_ROOT)
        with patch.object(self.job, 'cdk', return_value=(0, json.dumps(actual))) as cdk, \
                patch.object(self.job, 'gcloud', return_value=(0, '')) as cloud:
            self.job.authorize()
        self.assertTrue(self.job.state['authorized'])
        cdk.assert_called_once_with(args=['job', 'recipe', 'j-fixture', '--no-color'])
        cloud.assert_called_once_with(args=['storage', 'cp', str(self.directory / 'start.json'),
                                           self.job.state['uri'] + '/control/start.json'])

    def test_resource_changes_and_malformed_quantities_are_rejected(self) -> None:
        expected = controller.recipe(state=self.job.state, payload={'runtime.py': '# fixture'})
        original = expected['spec']['replicatedJobs'][0]['template']['spec']['template']['spec']['containers'][0]['resources']
        variants = [None, {}, {'limits': original['limits']},
                    {**original, 'claims': []}, {**original, 'requests': None},
                    {**original, 'requests': {**original['requests'], 'cpu': '1'}}]
        for kind in ('requests', 'limits'):
            for count in (99, '99', 0, 'four', None, True, 4.0, {}, []):
                variants.append({**original, kind: {'google.com/tpu': count}})
        for resources in variants:
            with self.subTest(resources=resources):
                actual = copy.deepcopy(expected)
                pod = actual['spec']['replicatedJobs'][0]['template']['spec']['template']['spec']
                pod['containers'][0]['resources'] = resources
                with self.assertRaisesRegex(ValueError, 'Rendered runner'):
                    controller.validate_recipe(actual=actual, expected=expected, service_account=None)

    def test_discovery_resolves_only_confirmed_job_prefix(self) -> None:
        row = {**self.fixture.remote, 'user': self.job.state['user'],
            'recipe': self.job.state['recipe'], 'tags': [self.job.state['run_id']]}
        self.job.state.pop('job_id')
        self.job.save(uri=None)
        with patch.object(self.job, 'cdk', return_value=(0, json.dumps([row]))):
            self.job.discover()
        self.assertEqual(self.job.state['uri'],
            'gs://fixture-jobs/jobs/j-fixture/outputs/workflow-' + self.job.state['run_id'])

    def test_submission_precedes_upload_and_authorization(self) -> None:
        self.job.save(finished=False, submission_started=False, authorized=False, uploaded=False)
        events = []
        def submit(*, args: list[str], **kwargs) -> tuple[int, str]:
            self.assertEqual(args[:2], ['job', 'create'])
            self.assertIn('--mount-gcs=true', args)
            events.append('submit')
            return 0, ''
        def upload(*, args: list[str], **kwargs) -> tuple[int, str]:
            self.assertIn('submit', events)
            self.assertTrue(args[-1].startswith(self.job.state['uri'] + '/'))
            self.assertNotIn('buckets', args)
            events.append(args[1])
            return 0, ''
        with self.mocked(), patch.object(self.job, 'gcloud', side_effect=upload), \
                patch.object(self.job, 'cdk', side_effect=submit), \
                patch.object(self.job, 'ensure_bucket', side_effect=AssertionError('no bucket creation')), \
                patch.object(self.job, 'register'), \
                patch.object(self.job, 'discover', side_effect=[{'job_status': 'Pending', 'id': 'j-fixture'}, self.fixture.remote]), \
                patch.object(controller.STOP, 'wait'), \
                patch.object(self.job, 'authorize', side_effect=lambda: events.append('authorize')), \
                patch.object(self.job, 'collect', return_value=True), \
                patch.object(self.job, 'finish_cleanup'):
            self.assertEqual(self.job.run(), 0)
        self.assertEqual(events, ['submit', 'cp', 'rsync', 'authorize'])

    def test_cleanup_deletes_only_owned_prefix_and_image(self) -> None:
        with self.mocked(), patch('builtins.input', side_effect=AssertionError('no prompt')):
            self.job.cleanup(discard=False, automatic=True)
        self.assertTrue(self.job.state['resources_cleaned'])
        self.assertEqual(len(self.commands), 2)
        self.assertFalse(self.fixture.image_exists)
        self.assertEqual(self.fixture.local_tags, {'unrelated:keep'})
        record = core.read_document(self.directory / 'cleanup.json')
        self.assertEqual(record['storage_uri'], self.job.state['uri'])
        self.assertTrue((self.directory / 'collected/manifest.json').is_file())

    def test_interrupted_prefix_deletion_retries_without_owner(self) -> None:
        self.fail_once = True
        with self.mocked(), self.assertRaisesRegex(RuntimeError, 'connection lost'):
            self.job.cleanup(discard=False, automatic=True)
        self.assertFalse(self.job.state.get('deleted'))
        self.job = controller.Job(self.directory)
        with self.mocked():
            self.job.cleanup(discard=False, automatic=True)
        self.assertEqual(sum(args[1] == 'cat' for args in self.commands), 1)
        self.assertEqual(sum(args[1] == 'rsync' for args in self.commands), 2)
        self.assertTrue(self.job.state['resources_cleaned'])

    def test_foreign_prefix_cannot_be_cleaned(self) -> None:
        self.job.save(uri='gs://fixture-jobs/jobs/j-other/outputs/')
        with self.mocked(), self.assertRaisesRegex(ValueError, 'ownership record differs'):
            self.job.cleanup(discard=False, automatic=True)
        self.assertEqual(self.commands, [])

    def test_fixed_image_cleanup_preserves_image(self) -> None:
        self.job.state.pop('owned_image')
        with self.mocked(), patch('sys.stdin.isatty', return_value=True), \
                patch('builtins.input', return_value='DELETE ' + self.job.state['run_id']):
            self.job.cleanup(discard=False)
        self.assertTrue(self.job.state['deleted'])
        self.assertTrue(self.fixture.image_exists)

    def test_storage_permission_failure_retains_resources(self) -> None:
        with self.mocked(), patch.object(self.job, 'gcloud', side_effect=RuntimeError('access denied')), \
                self.assertRaisesRegex(RuntimeError, 'access denied'):
            self.job.cleanup(discard=False, automatic=True)
        self.assertFalse(self.job.state.get('deleted'))
        self.assertTrue(self.fixture.image_exists)

    def test_runtime_waits_for_inputs_and_preserves_error(self) -> None:
        bucket = self.directory / 'fake-cdk-output'
        source = self.directory / 'input'
        config = core.read_document(source / 'run.json')
        target = self.directory / 'runtime-code'
        config['bundles'][0]['destination'] = str(target)
        config['run']['cwd'] = str(target)
        config['run']['verify_imports'] = {}
        config['run']['argv'] = [sys.executable, '-c', 'import sys; print("workload error", file=sys.stderr); sys.exit(42)']
        core.save(path=source / 'run.json', value=config)
        digest = core.checksum(source / 'run.json')
        delivered = threading.Event()
        def deliver() -> None:
            shutil.copytree(src=source, dst=bucket / 'input', dirs_exist_ok=True)
            shutil.copyfile(src=self.directory / 'owner.json', dst=bucket / 'owner.json')
            core.save(path=bucket / 'control/start.json', value={'run_id': config['run_id'], 'config_sha256': digest})
            delivered.set()
        timer = threading.Timer(interval=0.5, function=deliver)
        timer.start()
        try:
            script = 'from pathlib import Path; import runtime,sys; sys.exit(runtime.execute(bucket=Path(sys.argv[1]),local=Path(sys.argv[2]),expected=sys.argv[3],require_mount=False,input_wait_seconds=5))'
            result = subprocess.run(args=[sys.executable, '-c', script, str(bucket), str(self.directory / 'runtime-local'), digest],
                env={**os.environ, 'PYTHONPATH': str(controller.ROOT), 'RUN_ID': config['run_id']},
                capture_output=True, text=True, timeout=15)
        finally:
            timer.cancel()
            timer.join()
        self.assertTrue(delivered.is_set())
        self.assertIn('WAITING_FOR_INPUTS', result.stdout)
        self.assertEqual(result.returncode, 42, result.stderr)
        self.assertEqual(core.read_document(bucket / 'manifest.json')['exit_code'], 42)
        self.assertIn('workload error', (self.directory / 'runtime-local/stderr.log').read_text())


if __name__ == '__main__':
    unittest.main()
