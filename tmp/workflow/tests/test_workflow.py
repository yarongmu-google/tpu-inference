"""Local protocol tests; cloud commands are replaced with deterministic fixtures."""
from __future__ import annotations

import copy
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import controller
import core
import runtime


class WorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix='description-workflow-test-')
        self.base = Path(self.temp.name).resolve()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(controller.STOP.clear)
        controller.STOP.clear()
        (self.base / 'code').mkdir()
        (self.base / 'code/main.py').write_text('print("fixture")\n')
        self.profile = {'version': 1, 'cloud': {'project': 'fixture-project', 'region': 'us-central1',
                       'service_account': 'fixture-worker', 'workload_iam_member': 'serviceAccount:fixture@example.invalid'},
                       'runtime': {'image': 'registry.invalid/runtime@sha256:' + 'a' * 64},
                       'hardware': {'accelerator': 'tpu7x', 'topology': '2x2x1', 'chips_per_host': 4},
                       'storage': {'dedicated_bucket': True, 'deletion': 'manual', 'soft_delete_days': 0}}
        self.description = {'version': 1, 'profile': 'profile.json', 'name': 'Fixture Run',
                            'code': {'directory': 'code', 'delivery': 'snapshot', 'destination': '/workspace/code'},
                            'run': {'cwd': '/workspace/code', 'argv': ['python3', 'main.py']},
                            'outputs': {'directory': 'results', 'snapshot_seconds': 1},
                            'execution': {'poll_seconds': 1, 'archive_seconds': 1, 'wait_seconds': 5}}
        self.write_descriptions()

    def write_descriptions(self) -> None:
        core.save(path=self.base / 'profile.json', value=self.profile)
        core.save(path=self.base / 'experiment.json', value=self.description)

    def prepare(self, dry_run: bool = False) -> Path:
        self.write_descriptions()
        with patch('sys.stdout', new=io.StringIO()):
            root = controller.prepare(description_path=self.base / 'experiment.json', name=None, dry_run=dry_run)
        names = json.loads((root / 'campaign.json').read_text())['jobs']
        self.campaign = root
        return root / names[0]

    def test_name_normalization_uniqueness_and_prompt(self) -> None:
        self.assertEqual(core.slug(' Long Name / TEST! '), 'long-name-test')
        self.assertEqual(core.slug('google-internal'), 'run')
        self.assertNotEqual(core.identity('same'), core.identity('same'))
        self.assertLessEqual(len(core.identity('x' * 100)), 63)
        self.description['name'] = None
        self.write_descriptions()
        with patch('sys.stdin.isatty', return_value=True), patch('builtins.input', return_value='Chosen Name') as prompt, patch('sys.stdout', new=io.StringIO()):
            root = controller.prepare(description_path=self.base / 'experiment.json', name=None, dry_run=True)
        self.assertEqual(prompt.call_count, 1)
        self.assertEqual(json.loads((root / 'campaign.json').read_text())['name'], 'Chosen Name')

    def test_noninteractive_name_required(self) -> None:
        self.description['name'] = None
        self.write_descriptions()
        with patch('sys.stdin.isatty', return_value=False), self.assertRaisesRegex(ValueError, 'Supply --name'):
            controller.prepare(description_path=self.base / 'experiment.json', name=None)

    def test_yaml_and_duplicate_keys(self) -> None:
        import yaml
        yaml_path = self.base / 'experiment.yml'
        yaml_path.write_text(yaml.safe_dump(self.description))
        data, _ = core.load_description(yaml_path)
        self.assertEqual(data['name'], 'Fixture Run')
        for name, content in [('a.json', '{"a":1,"a":2}'), ('a.yml', 'a: 1\na: 2\n')]:
            path = self.base / name
            path.write_text(content)
            with self.assertRaises(ValueError):
                core.read_document(path)

    def test_unknown_keys_and_mutable_image_rejected(self) -> None:
        self.description['outptus'] = {}
        self.write_descriptions()
        with self.assertRaises(ValueError):
            core.load_description(self.base / 'experiment.json')
        del self.description['outptus']
        self.profile['runtime']['image'] = 'registry.invalid/runtime:latest'
        self.write_descriptions()
        with self.assertRaisesRegex(ValueError, 'digest'):
            core.load_description(self.base / 'experiment.json')

    def test_provisioned_scratch_requires_valid_class_and_capacity(self) -> None:
        for execution in ({'scratch_storage_class': 'premium-rwo'},
                          {'scratch_gib': 1024, 'scratch_storage_class': ''},
                          {'scratch_gib': 1024, 'scratch_storage_class': 1},
                          {'scratch_gib': 1024, 'scratch_storage_class': '../disk'},
                          {'scratch_gib': 1024, 'scratch_storage_class': 'disk..class'},
                          {'scratch_gib': 0, 'scratch_storage_class': 'premium-rwo'}):
            self.description['execution'] = execution
            self.write_descriptions()
            with self.subTest(execution=execution), self.assertRaises(ValueError):
                core.load_description(self.base / 'experiment.json')
        self.description['execution'] = {'scratch_gib': 1024, 'scratch_storage_class': 'premium-rwo'}
        self.write_descriptions()
        data, _ = core.load_description(self.base / 'experiment.json')
        self.assertEqual(data['execution']['scratch_storage_class'], 'premium-rwo')

    def test_snapshot_and_input_verification(self) -> None:
        source = self.base / 'code'
        entries = core.snapshot(source=source, target=self.base / 'copy', include=['**'])
        core.verify(root=self.base / 'copy', entries=entries)
        (source / 'main.py').write_text('changed')
        self.assertEqual((self.base / 'copy/main.py').read_text(), 'print("fixture")\n')
        (self.base / 'copy/main.py').write_text('corrupt')
        with self.assertRaises(ValueError):
            core.verify(root=self.base / 'copy', entries=entries)

    def test_symlinks_traversal_and_recursive_snapshot_rejected(self) -> None:
        (self.base / 'code/link').symlink_to(self.base / 'profile.json')
        with self.assertRaises(ValueError):
            core.snapshot(source=self.base / 'code', target=self.base / 'copy', include=['**'])
        with self.assertRaises(ValueError):
            core.safe_path(root=self.base, relative='../elsewhere')
        with self.assertRaises(ValueError):
            core.snapshot(source=self.base / 'code', target=self.base / 'code/recursive', include=['**'])

    def test_cases_have_distinct_buckets_same_image_and_pinned_sources(self) -> None:
        self.description['cases'] = [{'name': 'first', 'env': {'SIZE': 128}}, {'name': 'second', 'env': {'SIZE': 256}}]
        first = self.prepare()
        campaign = json.loads((self.campaign / 'campaign.json').read_text())
        jobs = [json.loads((self.campaign / name / 'state.json').read_text()) for name in campaign['jobs']]
        self.assertEqual(len({j['bucket'] for j in jobs}), 2)
        self.assertEqual(len({j['image'] for j in jobs}), 1)
        payload = json.loads((first / 'input/run.json').read_text())
        self.assertEqual(payload['run']['env']['SIZE'], '128')
        self.assertNotIn(str(self.base), json.dumps(payload))

    def test_recipe_uses_dedicated_mount_and_checks_assigned_identity(self) -> None:
        folder = self.prepare()
        state = json.loads((folder / 'state.json').read_text())
        expected = json.loads((folder / 'recipe.json').read_text())
        actual = copy.deepcopy(expected)
        pod = actual['spec']['replicatedJobs'][0]['template']['spec']['template']['spec']
        pod['serviceAccountName'] = 'fixture-worker'
        controller.validate_recipe(actual=actual, expected=expected, service_account='fixture-worker')
        self.assertEqual(pod['volumes'][0]['csi']['volumeAttributes']['bucketName'], state['bucket'])
        pod['volumes'][0]['csi']['volumeAttributes']['bucketName'] = 'wrong-bucket'
        with self.assertRaisesRegex(ValueError, 'bucket'):
            controller.validate_recipe(actual=actual, expected=expected, service_account='fixture-worker')
        pod['volumes'] = copy.deepcopy(expected['spec']['replicatedJobs'][0]['template']['spec']['template']['spec']['volumes'])
        with self.assertRaisesRegex(ValueError, 'service account'):
            controller.validate_recipe(actual=actual, expected=expected, service_account='wrong-account')

    def test_arguments_are_not_shell_evaluated(self) -> None:
        value = runtime.expand(value='${OUTPUT_DIR}/$(touch unsafe)', variables={'OUTPUT_DIR': '/out with spaces'})
        self.assertEqual(value, '/out with spaces/$(touch unsafe)')
        with self.assertRaises(ValueError):
            runtime.expand(value='${MISSING}', variables={})

    def test_real_command_failure_and_timeout_retained(self) -> None:
        job = controller.Job(self.prepare())
        with patch('sys.stdout', new=io.StringIO()):
            code, _ = job.command(args=[sys.executable, '-c', 'import sys; print("failure",file=sys.stderr); sys.exit(42)'], check=False)
            self.assertEqual(code, 42)
            with self.assertRaises(subprocess.TimeoutExpired):
                job.command(args=[sys.executable, '-c', 'import time; time.sleep(30)'], timeout=0.1)
        codes = [p.read_text().strip() for p in (job.directory / 'commands').glob('*.exit')]
        self.assertEqual(codes, ['42', '124'])
        self.assertIn('failure', ''.join(p.read_text() for p in (job.directory / 'commands').glob('*.stderr')))

    def test_unknown_submission_does_not_create_again(self) -> None:
        job = controller.Job(self.prepare())
        job.save(submission_started=True)
        with patch.object(job, 'discover', return_value=None), patch.object(job, 'ensure_bucket') as bucket:
            with self.assertRaisesRegex(RuntimeError, 'unresolved'):
                job.run()
        bucket.assert_not_called()

    def test_bucket_creation_uncertainty_does_not_adopt_existing_bucket(self) -> None:
        job = controller.Job(self.prepare())
        job.save(bucket_creation_started=True)
        with patch.object(job, 'gcloud') as cloud, patch('sys.stdout', new=io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'uncertain'):
                job.ensure_bucket()
        cloud.assert_not_called()

    def test_bucket_creation_and_ownership_settings(self) -> None:
        job = controller.Job(self.prepare())
        calls = []
        def cloud(args, **kwargs):
            calls.append(args)
            if args[:3] == ['storage', 'buckets', 'describe']:
                return 0, json.dumps({'name': job.state['bucket'], 'labels': {'run_id': job.state['run_id']}})
            if args[:2] == ['storage', 'cat']:
                return 0, (job.directory / 'owner.json').read_text()
            return 0, ''
        with patch.object(job, 'gcloud', side_effect=cloud), patch('sys.stdout', new=io.StringIO()):
            job.ensure_bucket()
            job.ensure_bucket()
        creates = [c for c in calls if c[:3] == ['storage', 'buckets', 'create']]
        self.assertEqual(len(creates), 1)
        self.assertIn('--soft-delete-duration=0', creates[0])
        self.assertIn('--public-access-prevention', creates[0])
        grants = [c for c in calls if c[:3] == ['storage', 'buckets', 'add-iam-policy-binding']]
        self.assertEqual(len(grants), 1)
        self.assertIn('--role=roles/storage.objectUser', grants[0])

    def test_artifact_roundtrip_and_corruption(self) -> None:
        folder = self.prepare()
        job = controller.Job(folder)
        bucket = self.base / 'bucket'
        bucket.mkdir()
        local = self.base / 'runtime'
        local.mkdir()
        (local / 'stderr.log').write_text('error detail')
        status = {'run_id': job.state['run_id'], 'image': job.state['image'], 'config_sha256': job.state['config_sha256'],
                  'state': 'failed', 'exit_code': 42, 'updated': time.time()}
        runtime.publish(local=local, bucket=bucket, status=status, extras={})
        manifest = json.loads((bucket / 'manifest.json').read_text())
        def cloud(args, **kwargs):
            if args[:2] == ['storage', 'cat']:
                return 0, (bucket / 'manifest.json').read_text()
            source = bucket / args[2].split(job.state['uri'] + '/', 1)[1]
            shutil.copyfile(src=source, dst=args[3])
            return 0, ''
        with patch.object(job, 'gcloud', side_effect=cloud), patch('sys.stdout', new=io.StringIO()), patch.object(controller.STOP, 'wait'):
            self.assertFalse(job.collect())
            self.assertTrue(job.state['artifacts_verified'])
            self.assertEqual((folder / 'collected/files/diagnostics/stderr.log').read_text(), 'error detail')
            entry = manifest['files'][0]
            (bucket / 'objects' / entry['sha256']).write_text('corrupt')
            (folder / 'collected/objects' / entry['sha256']).unlink()
            self.assertFalse(job.collect())
            self.assertFalse(job.state['artifacts_verified'])

    def test_malicious_artifact_manifest_rejected_before_download(self) -> None:
        job = controller.Job(self.prepare())
        manifest = {'format': 'run-artifacts-v1', 'run_id': job.state['run_id'], 'image': job.state['image'],
                    'config_sha256': job.state['config_sha256'], 'state': 'succeeded', 'exit_code': 0,
                    'files': [{'path': '../escape', 'sha256': 'a' * 64, 'bytes': 1}]}
        with patch.object(job, 'gcloud', return_value=(0, json.dumps(manifest))) as cloud, patch('sys.stdout', new=io.StringIO()), patch.object(controller.STOP, 'wait'):
            self.assertFalse(job.collect())
        self.assertTrue(all(c.kwargs['args'][:2] == ['storage', 'cat'] for c in cloud.call_args_list))

    def test_cleanup_blocks_active_job_and_requires_exact_confirmation(self) -> None:
        job = controller.Job(self.prepare())
        job.save(submission_started=True, bucket_marked=True)
        with patch.object(job, 'discover', return_value={'job_status': 'Running'}), patch.object(job, 'gcloud') as cloud:
            with self.assertRaisesRegex(ValueError, 'active'):
                job.cleanup(discard=True)
        cloud.assert_not_called()
        def cloud(args, **kwargs):
            if args[:2] == ['storage', 'cat']:
                return 0, (job.directory / 'owner.json').read_text()
            return 0, ''
        with patch.object(job, 'discover', return_value={'job_status': 'Failed', 'state': 'Complete'}), \
             patch.object(job, 'bucket_identity', return_value={'labels': {'run_id': job.state['run_id']}}), \
             patch.object(job, 'gcloud', side_effect=cloud) as calls, patch('sys.stdin.isatty', return_value=True), \
             patch('builtins.input', return_value='no'), patch('sys.stdout', new=io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'confirmed'):
                job.cleanup(discard=True)
            self.assertFalse(any(c.kwargs['args'][:2] == ['storage', 'rm'] for c in calls.call_args_list))
        with patch.object(job, 'discover', return_value={'job_status': 'Failed', 'state': 'Complete'}), \
             patch.object(job, 'bucket_identity', return_value={'labels': {'run_id': job.state['run_id']}}), \
             patch.object(job, 'gcloud', side_effect=cloud) as calls, patch('sys.stdin.isatty', return_value=True), \
             patch('builtins.input', return_value='DELETE ' + job.state['run_id']), patch('sys.stdout', new=io.StringIO()):
            self.assertEqual(job.cleanup(discard=True), 0)
            deletes = [c.kwargs['args'] for c in calls.call_args_list if c.kwargs['args'][:2] == ['storage', 'rm']]
            self.assertEqual(deletes, [['storage', 'rm', '--recursive', '--quiet', job.state['uri']]])

    def test_completed_run_does_not_submit_again(self) -> None:
        job = controller.Job(self.prepare())
        job.save(finished=True, exit_code=0)
        with patch.object(job, 'cdk') as cdk:
            self.assertEqual(job.run(), 0)
        cdk.assert_not_called()

    def test_no_image_build_or_extra_tags_during_submission(self) -> None:
        job = controller.Job(self.prepare())
        calls = []
        row = {'id': 'j-fixture', 'job_status': 'Succeeded', 'state': 'Complete'}
        def cdk(args, **kwargs):
            calls.append(args)
            return 1, ''
        def discover():
            job.save(job_id='j-fixture')
            return row
        with patch.object(job, 'ensure_bucket'), patch.object(job, 'register'), patch.object(job, 'gcloud', return_value=(0, '')), \
             patch.object(job, 'cdk', side_effect=cdk), patch.object(job, 'discover', side_effect=discover), \
             patch.object(job, 'authorize'), patch.object(job, 'collect', return_value=True), patch('sys.stdout', new=io.StringIO()):
            self.assertEqual(job.run(), 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][calls[0].index('--tags') + 1], job.state['label'] + ',' + job.state['run_id'])
        self.assertIn('--mount-gcs=false', calls[0])

    def test_runtime_real_success_failure_and_output_capture(self) -> None:
        folder = self.prepare()
        bucket = self.base / 'bucket'
        bucket.mkdir()
        shutil.copytree(src=folder / 'input', dst=bucket / 'input')
        shutil.copyfile(src=folder / 'owner.json', dst=bucket / 'owner.json')
        config = json.loads((bucket / 'input/run.json').read_text())
        target = self.base / 'runtime-code'
        config['bundles'][0]['destination'] = str(target)
        config['run']['cwd'] = str(target)
        config['run']['argv'] = [sys.executable, '-c', 'import os,pathlib,sys; pathlib.Path(os.environ["OUTPUT_DIR"],"result.txt").write_text("ok"); print("stdout fixture"); print("stderr fixture",file=sys.stderr); sys.exit(int(sys.argv[1]))', '0']
        for exit_code in (0, 42):
            config['run']['argv'][-1] = str(exit_code)
            core.save(path=bucket / 'input/run.json', value=config)
            digest = core.checksum(bucket / 'input/run.json')
            core.save(path=bucket / 'control/start.json', value={'run_id': config['run_id'], 'config_sha256': digest})
            local = self.base / f'run-{exit_code}'
            script = 'from pathlib import Path; import runtime,sys; sys.exit(runtime.execute(bucket=Path(sys.argv[1]),local=Path(sys.argv[2]),expected=sys.argv[3],require_mount=False))'
            result = subprocess.run(args=[sys.executable, '-c', script, str(bucket), str(local), digest],
                                    env={**os.environ, 'PYTHONPATH': str(controller.ROOT), 'RUN_ID': config['run_id']},
                                    capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, exit_code, result.stderr)
            manifest = json.loads((bucket / 'manifest.json').read_text())
            self.assertEqual(manifest['exit_code'], exit_code)
            self.assertIn('stdout fixture', (local / 'stdout.log').read_text())
            self.assertEqual((local / 'artifacts/result.txt').read_text(), 'ok')
            paths = [entry['path'] for entry in manifest['files']]
            self.assertIn('artifacts/output/result.txt', paths)
            self.assertNotIn('diagnostics/artifacts/result.txt', paths)

    def test_runtime_missing_mount_prevents_workload(self) -> None:
        script = 'from pathlib import Path; import runtime,sys; sys.exit(runtime.execute(bucket=Path(sys.argv[1]),local=Path(sys.argv[2]),expected="unused"))'
        result = subprocess.run(args=[sys.executable, '-c', script, str(self.base / 'no-mount'), str(self.base / 'local')],
                                env={**os.environ, 'PYTHONPATH': str(controller.ROOT), 'RUN_ID': 'fixture'},
                                capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue((self.base / 'local/error.txt').exists())

    def test_dry_run_is_local_only_and_not_resumable(self) -> None:
        folder = self.prepare(dry_run=True)
        with patch.object(controller.Job, 'gcloud') as cloud, patch.object(controller.Job, 'cdk') as cdk, patch('sys.stdout', new=io.StringIO()), patch('sys.stderr', new=io.StringIO()):
            self.assertEqual(controller.locked_job(directory=folder), 1)
        cloud.assert_not_called()
        cdk.assert_not_called()

    def test_scheduler_reserves_uncertain_submissions(self) -> None:
        self.description['cases'] = [{'name': 'first'}, {'name': 'second'}, {'name': 'third'}]
        self.description['execution']['max_in_flight'] = 1
        self.prepare()
        campaign = json.loads((self.campaign / 'campaign.json').read_text())
        first, second, third = [self.campaign / n for n in campaign['jobs']]
        state = json.loads((second / 'state.json').read_text())
        state['submission_started'] = True
        core.save(path=second / 'state.json', value=state)
        seen = []
        def work(directory, *args, **kwargs):
            seen.append(directory)
            return 1
        with patch.object(controller, 'locked_job', side_effect=work), patch('sys.stdout', new=io.StringIO()):
            self.assertEqual(controller.run_campaign(self.campaign), 1)
        self.assertEqual(seen, [second])


    def test_fresh_acknowledgement_and_changed_rules(self) -> None:
        job = controller.Job(self.prepare())
        letter = 'Fixture rules\n[ack 1/2: AAAA]\n[ack 2/2: BBBB]'
        normal = ' '.join(controller.re.sub(r'\[ack \d+/\d+: [A-Za-z0-9]+\]', '[ack]', letter).split())
        seen = []
        def command(args, **kwargs):
            seen.append(args)
            return 0, letter if args == ['cdk', 'agent-letter'] else '[]'
        digest = controller.hashlib.sha256(normal.encode()).hexdigest()
        with patch.object(job, 'command', side_effect=command), patch.object(controller, 'LETTER_HASH', digest):
            job.cdk(args=['job', 'list', '-o', 'json'])
            job.cdk(args=['recipe', 'list', '-o', 'json'])
        self.assertEqual(seen[0], ['cdk', 'agent-letter'])
        self.assertEqual(seen[2], ['cdk', 'agent-letter'])
        self.assertEqual(seen[1][1], '--agent-code=AAAA-BBBB')
        with patch.object(job, 'command', return_value=(0, 'changed')) as calls:
            with self.assertRaises(ValueError):
                job.cdk(args=['job', 'create', 'unused'])
        self.assertEqual(calls.call_count, 1)

    def test_authorization_not_published_for_wrong_rendered_bucket(self) -> None:
        job = controller.Job(self.prepare())
        job.save(job_id='j-fixture')
        rendered = json.loads((job.directory / 'recipe.json').read_text())
        pod = rendered['spec']['replicatedJobs'][0]['template']['spec']['template']['spec']
        pod['serviceAccountName'] = 'fixture-worker'
        pod['volumes'][0]['csi']['volumeAttributes']['bucketName'] = 'other'
        with patch.object(job, 'cdk', return_value=(0, json.dumps(rendered))), patch.object(job, 'gcloud') as cloud:
            with self.assertRaises(ValueError):
                job.authorize()
        cloud.assert_not_called()

    def test_live_collection_omits_large_artifacts(self) -> None:
        job = controller.Job(self.prepare())
        manifest = {'format': 'run-artifacts-v1', 'run_id': job.state['run_id'], 'image': job.state['image'],
                    'config_sha256': job.state['config_sha256'], 'state': 'running', 'exit_code': None,
                    'updated': time.time(), 'files': [{'path': 'artifacts/output/large.bin', 'sha256': 'a' * 64, 'bytes': 10000000}]}
        with patch.object(job, 'gcloud', return_value=(0, json.dumps(manifest))) as calls, patch('sys.stdout', new=io.StringIO()):
            job.collect(live=True)
        self.assertEqual(calls.call_count, 1)
        self.assertEqual(job.state['workload_state'], 'running')

    def test_runtime_termination_preserves_error_and_exit(self) -> None:
        folder = self.prepare()
        bucket = self.base / 'bucket'
        bucket.mkdir()
        shutil.copytree(src=folder / 'input', dst=bucket / 'input')
        shutil.copyfile(src=folder / 'owner.json', dst=bucket / 'owner.json')
        config = json.loads((bucket / 'input/run.json').read_text())
        config['bundles'][0]['destination'] = str(self.base / 'runtime-code')
        config['run']['cwd'] = str(self.base / 'runtime-code')
        config['run']['argv'] = [sys.executable, '-u', '-c', 'import time; print("started"); time.sleep(60)']
        core.save(path=bucket / 'input/run.json', value=config)
        digest = core.checksum(bucket / 'input/run.json')
        core.save(path=bucket / 'control/start.json', value={'run_id': config['run_id'], 'config_sha256': digest})
        local = self.base / 'runtime'
        script = 'from pathlib import Path; import runtime,sys; sys.exit(runtime.execute(bucket=Path(sys.argv[1]),local=Path(sys.argv[2]),expected=sys.argv[3],require_mount=False))'
        child = subprocess.Popen(args=[sys.executable, '-c', script, str(bucket), str(local), digest],
                                 env={**os.environ, 'PYTHONPATH': str(controller.ROOT), 'RUN_ID': config['run_id']},
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            deadline = time.monotonic() + 5
            log = local / 'stdout.log'
            while not (log.exists() and 'started' in log.read_text()):
                if child.poll() is not None or time.monotonic() > deadline:
                    self.fail('Runtime did not start fixture')
                time.sleep(0.05)
            child.send_signal(signal.SIGTERM)
            stdout, stderr = child.communicate(timeout=10)
            self.assertEqual(child.returncode, 143, stdout + stderr)
            self.assertEqual(json.loads((bucket / 'manifest.json').read_text())['exit_code'], 143)
            self.assertIn('started', log.read_text())
        finally:
            if child.poll() is None:
                child.kill()
                child.communicate()


if __name__ == '__main__':
    unittest.main()
