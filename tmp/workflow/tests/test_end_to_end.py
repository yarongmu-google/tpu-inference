"""Replay CDK storage through workload execution, collection and cleanup locally."""
from __future__ import annotations

import copy
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import unittest
from unittest.mock import patch

import controller
import core
import pack_results
import runtime
import test_cdk_storage


def mounted_recipe(job, mount: Path) -> dict:
    actual = core.read_document(job.directory / 'recipe.json')
    pod = actual['spec']['replicatedJobs'][0]['template']['spec']['template']['spec']
    runner = next(c for c in pod['containers'] if c['name'] == 'runner')
    for quantities in runner['resources'].values():
        for name, value in quantities.items():
            quantities[name] = str(value)
    runner['volumeMounts'].append({'name': 'cdk-outputs', 'mountPath': str(mount)})
    root = job.state['profile']['storage']['outputs_root'].removeprefix('gs://')
    bucket, prefix = root.split('/', 1)
    pod['volumes'].append({'name': 'cdk-outputs', 'csi': {'driver': 'gcsfuse.csi.storage.gke.io',
        'volumeAttributes': {'bucketName': bucket, 'mountOptions': 'only-dir=' + prefix + '/' + job.state['job_id']}}})
    return actual


class EndToEndTests(unittest.TestCase):
    def setUp(self) -> None:
        fixture = test_cdk_storage.CdkStorageTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.job = fixture.job
        self.base = fixture.directory / 'simulation'
        self.base.mkdir()
        self.mount = self.base / 'cdk-outputs'
        self.mount.mkdir()
        self.remote_root = self.job.state['profile']['storage']['outputs_root'] + '/' + self.job.state['job_id']
        self.job.save(finished=False, authorized=False, uploaded=False, artifacts_verified=False)

    def run_pipeline(self, code: int) -> None:
        job = self.job
        template = core.read_document(job.directory / 'run-template.json')
        target = self.base / 'code'
        template['bundles'][0]['destination'] = str(target)
        template['run'].update(cwd=str(target), verify_imports={}, argv=[sys.executable, '-c',
            'import os,sys,json; from pathlib import Path; p=Path(os.environ["OUTPUT_DIR"]); '
            '(p/"summary.json").write_text(json.dumps({"fixture":True})); '
            '(p/"client.log").write_text("saved client result"); print("fixture output"); '
            'print("fixture diagnostic",file=sys.stderr); sys.exit(' + str(code) + ')'])
        core.save(path=job.directory / 'run-template.json', value=template)
        job.save(config_sha256=None)
        with patch('sys.stdout', new=io.StringIO()):
            job.prepare_image()
        actual = mounted_recipe(job=job, mount=self.mount)
        unrelated = self.mount / 'outputs/unrelated/keep.txt'
        unrelated.parent.mkdir(parents=True)
        unrelated.write_text('keep')
        calls = []
        def remote(value: str) -> Path:
            self.assertTrue(value.startswith(self.remote_root + '/'), value)
            return self.mount / value[len(self.remote_root) + 1:]
        def cloud(*, args: list[str], **kwargs):
            calls.append(args)
            if args[:2] == ['storage', 'cat']:
                return 0, remote(args[2]).read_text()
            if args[:2] == ['storage', 'cp']:
                source = remote(args[2]) if args[2].startswith('gs://') else Path(args[2])
                destination = remote(args[3]) if args[3].startswith('gs://') else Path(args[3])
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src=source, dst=destination)
                return 0, ''
            self.assertEqual(args[:2], ['storage', 'rsync'])
            source, destination = Path(args[-2]), remote(args[-1])
            if '--delete-unmatched-destination-objects' in args:
                self.assertEqual(list(source.iterdir()), [])
                self.assertEqual(destination, self.mount / 'outputs' / ('workflow-' + job.state['run_id']))
                for child in destination.iterdir():
                    shutil.rmtree(child) if child.is_dir() else child.unlink()
            else:
                shutil.copytree(src=source, dst=destination, dirs_exist_ok=True)
            return 0, ''
        def cdk(*, args: list[str], **kwargs):
            if args[1] == 'recipe':
                return 0, json.dumps(actual)
            if args[1] == 'desc':
                return 0, 'fixture terminal job'
            if args[1] == 'log':
                return 0, 'fixture container diagnostic'
            raise AssertionError(args)
        read_text = Path.read_text
        make_directory = Path.mkdir
        is_file = Path.is_file
        markers = {self.mount}
        checking_mount = [False]
        def mkdir(path, *args, **kwargs):
            result = make_directory(path, *args, **kwargs)
            if checking_mount[0] and path.is_relative_to(self.mount):
                markers.add(path)
                markers.update(parent for parent in path.parents if parent.is_relative_to(self.mount))
            return result
        def visible_file(path):
            if checking_mount[0] and path.is_relative_to(self.mount) and path.parent not in markers:
                return False
            return is_file(path)
        def mountinfo(path, *args, **kwargs):
            if str(path) == '/proc/self/mountinfo':
                return f'24 0 0:1 / {self.mount} rw - fuse.gcsfuse fixture rw\n'
            return read_text(path, *args, **kwargs)
        terminal = {'job_status': 'Succeeded' if code == 0 else 'Failed', 'state': 'Complete', 'id': 'j-fixture'}
        with patch.object(job, 'gcloud', side_effect=cloud), patch.object(job, 'cdk', side_effect=cdk), \
                patch.object(job, 'command', side_effect=self.fixture.fixture.command), \
                patch.object(job, 'discover', return_value=terminal), \
                patch.object(core, 'CDK_MOUNT_ROOT', self.mount), \
                patch.object(runtime, 'CDK_MOUNT_ROOT', self.mount), \
                patch.object(controller, 'CDK_MOUNT_ROOT', self.mount), \
                patch.object(Path, 'read_text', mountinfo), \
                patch.object(Path, 'mkdir', mkdir), patch.object(Path, 'is_file', visible_file), \
                patch.dict(os.environ, {'RUN_ID': job.state['run_id'], 'RUN_IMAGE': job.state['image'],
                    'RUN_CONFIG_SHA256': job.state['config_sha256'], 'WORKFLOW_STORAGE_MODE': 'cdk',
                    'CDK_OUTPUT_DIR': str(self.mount / 'outputs/worker/pod/runner')}), \
                patch('sys.stdout', new=io.StringIO()), patch('sys.stderr', new=io.StringIO()):
            job.upload_cdk_inputs()
            job.authorize()
            checking_mount[0] = True
            self.assertEqual(runtime.main(local=self.base / 'runtime-local'), code)
            checking_mount[0] = False
            self.assertEqual(job.run(), int(bool(code)))
            archive = pack_results.export_run(directory=job.directory)
        self.assertTrue(job.state['resources_cleaned'])
        self.assertTrue(job.state['artifacts_verified'])
        self.assertEqual(job.state['artifact_exit_code'], code)
        self.assertEqual(unrelated.read_text(), 'keep')
        self.assertFalse(self.fixture.fixture.image_exists)
        self.assertEqual(sum(args[:2] == ['storage', 'cp'] and args[2].endswith('.tar.gz') for args in calls), 1)
        with tarfile.open(name=archive, mode='r:gz') as packed:
            self.assertIn('run/collected/files/artifacts/output/summary.json', packed.getnames())
            self.assertIn('run/collected/files/diagnostics/stderr.log', packed.getnames())

    def test_successful_workload_crosses_all_storage_boundaries(self) -> None:
        self.run_pipeline(code=0)

    def test_failed_workload_preserves_error_then_cleans_owned_resources(self) -> None:
        self.run_pipeline(code=42)

    def test_terminal_job_is_collected_without_upload_or_authorization(self) -> None:
        with patch.object(self.job, 'prepare_image'), patch.object(self.job, 'discover',
                return_value={'job_status': 'Failed', 'state': 'Complete', 'id': 'j-fixture'}), \
                patch.object(self.job, 'upload_cdk_inputs') as upload, \
                patch.object(self.job, 'authorize') as authorize, \
                patch.object(self.job, 'collect', return_value=False), \
                patch.object(self.job, 'cleanup') as cleanup, patch('sys.stdout', new=io.StringIO()):
            self.assertEqual(self.job.run(), 1)
        upload.assert_not_called()
        authorize.assert_not_called()
        cleanup.assert_not_called()
        self.assertTrue(self.job.state['cleanup_pending'])
        self.assertTrue(self.job.state['finished'])

    def test_missing_mount_keeps_console_error_without_writing_fake_remote_output(self) -> None:
        bucket = self.base / 'not-mounted'
        local = self.base / 'local-diagnostics'
        read_text = Path.read_text
        def mounts(path, *args, **kwargs):
            return '' if str(path) == '/proc/self/mountinfo' else read_text(path, *args, **kwargs)
        errors = io.StringIO()
        with patch.object(Path, 'read_text', mounts), \
                patch.dict(os.environ, {'RUN_ID': self.job.state['run_id'], 'RUN_IMAGE': self.job.state['image']}), \
                patch('sys.stdout', new=io.StringIO()), patch('sys.stderr', new=errors):
            self.assertEqual(runtime.execute(bucket=bucket, local=local, expected='a' * 64), 1)
        self.assertFalse(bucket.exists())
        self.assertIn('Storage mount is missing', (local / 'error.txt').read_text())
        self.assertNotIn('Final artifact publication failed', errors.getvalue())

    def test_failure_before_config_load_publishes_verifiable_identity(self) -> None:
        bucket = self.base / 'storage'
        bucket.mkdir()
        expected = self.job.state['config_sha256']
        with patch.dict(os.environ, {'RUN_ID': self.job.state['run_id'], 'RUN_IMAGE': self.job.state['image']}), \
                patch('sys.stdout', new=io.StringIO()), patch('sys.stderr', new=io.StringIO()):
            self.assertEqual(runtime.execute(bucket=bucket, local=self.base / 'error-local',
                                             expected=expected, require_mount=False), 1)
        manifest = core.read_document(bucket / 'manifest.json')
        self.assertEqual(manifest['image'], self.job.state['image'])
        self.assertEqual(manifest['config_sha256'], expected)
        self.assertEqual(manifest['state'], 'failed')
        self.assertIn('diagnostics/error.txt', {entry['path'] for entry in manifest['files']})

    def test_wrong_cdk_bucket_prefix_and_mount_are_rejected(self) -> None:
        with patch('sys.stdout', new=io.StringIO()):
            self.job.prepare_image()
        original = mounted_recipe(job=self.job, mount=core.CDK_MOUNT_ROOT)
        controller.validate_cdk_storage(actual=original, state=self.job.state)
        for changed in ('bucket', 'prefix', 'mount', 'readonly'):
            actual = copy.deepcopy(original)
            pod = actual['spec']['replicatedJobs'][0]['template']['spec']['template']['spec']
            csi = pod['volumes'][-1]['csi']
            if changed == 'bucket':
                csi['volumeAttributes']['bucketName'] = 'foreign'
            elif changed == 'prefix':
                csi['volumeAttributes']['mountOptions'] = 'only-dir=jobs/j-other'
            elif changed == 'mount':
                pod['containers'][0]['volumeMounts'][-1]['mountPath'] += '/outputs/worker/pod/runner'
            else:
                csi['readOnly'] = True
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                controller.validate_cdk_storage(actual=actual, state=self.job.state)


    def test_scratch_volume_does_not_reserve_disk_and_survives_rendering(self) -> None:
        self.job.state['execution']['scratch_gib'] = 600
        expected = controller.recipe(state=self.job.state, payload={})
        controller.validate_recipe(actual=expected, expected=expected, service_account=None)
        pod = expected['spec']['replicatedJobs'][0]['template']['spec']['template']['spec']
        for quantities in pod['containers'][0]['resources'].values():
            self.assertNotIn('ephemeral-storage', quantities)
        self.assertEqual(pod['volumes'][-1], {'name': 'run-scratch', 'emptyDir': {'sizeLimit': '600Gi'}})
        for target in ('volume', 'mount', 'request', 'limit'):
            actual = copy.deepcopy(expected)
            changed = actual['spec']['replicatedJobs'][0]['template']['spec']['template']['spec']
            if target == 'volume':
                changed['volumes'].pop()
            elif target == 'mount':
                changed['containers'][0]['volumeMounts'].pop()
            else:
                section = 'requests' if target == 'request' else 'limits'
                changed['containers'][0]['resources'][section]['ephemeral-storage'] = '1Gi'
            with self.subTest(target=target), self.assertRaises(ValueError):
                controller.validate_recipe(actual=actual, expected=expected, service_account=None)
