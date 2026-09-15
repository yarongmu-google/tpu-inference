"""Incremental candidate bundles are immutable, verified and available while running."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import controller
import core
import runtime
import streams
import test_workflow


class StreamTests(unittest.TestCase):
    def test_parts_round_trip_and_repeat_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'input'
            source.mkdir()
            (source / 'a.txt').write_text('a' * 16)
            (source / 'b.txt').write_text('b' * 16)
            with patch.object(streams, 'PART_BYTES', 20):
                record = streams.seal(source=source, destination=root / 'sealed/case',
                    names=['a.txt', 'b.txt'], metadata={'status': 'failed'})
            self.assertEqual(len(record['parts']), 2)
            self.assertTrue(streams.materialize(source=root / 'sealed/case', destination=root / 'received/case'))
            self.assertEqual((root / 'received/case/files/b.txt').read_text(), 'b' * 16)
            self.assertFalse(streams.materialize(source=root / 'sealed/case', destination=root / 'received/case'))
            self.assertTrue((source / 'a.txt').exists())

    def test_corruption_never_publishes_a_received_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'input'
            source.mkdir()
            (source / 'log').write_text('worker failed')
            streams.seal(source=source, destination=root / 'sealed/case', names=['log'], metadata={})
            (root / 'sealed/case/part-0000.tar.gz').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                streams.materialize(source=root / 'sealed/case', destination=root / 'received/case')
            self.assertFalse((root / 'received/case').exists())
            self.assertTrue((source / 'log').exists())

    def test_unsafe_source_never_publishes_a_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(ValueError):
                streams.seal(source=root, destination=root / 'out/case', names=['../secret'], metadata={})
            self.assertFalse((root / 'out/case').exists())

    def test_publish_omits_private_work_and_reuses_uploaded_objects(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / 'local/artifacts'
            (output / '.work').mkdir(parents=True)
            (output / '.work/raw.txt').write_text('private raw dump')
            (output / 'result.json').write_text('{}')
            (root / 'remote').mkdir()
            cache = {}
            runtime.publish(local=root / 'local', bucket=root / 'remote', status={},
                            extras={'output': str(output)}, cache=cache)
            with patch.object(runtime, 'checksum', side_effect=AssertionError('must not reread remote objects')):
                runtime.publish(local=root / 'local', bucket=root / 'remote', status={},
                                extras={'output': str(output)}, cache=cache)
            record = json.loads((root / 'remote/manifest.json').read_text())
            self.assertEqual([entry['path'] for entry in record['files']], ['artifacts/output/result.json'])
            runtime.publish(local=root / 'local', bucket=root / 'remote', status={},
                            extras={'output': str(output)}, cache=cache, include_private=True)
            record = json.loads((root / 'remote/manifest.json').read_text())
            self.assertIn('artifacts/output/.work/raw.txt', [entry['path'] for entry in record['files']])


class IncrementalControllerTests(unittest.TestCase):
    setUp = test_workflow.WorkflowTests.setUp
    write_descriptions = test_workflow.WorkflowTests.write_descriptions
    prepare = test_workflow.WorkflowTests.prepare

    def test_live_candidate_is_available_and_downloaded_only_once(self):
        self.description['outputs']['delivery'] = 'incremental'
        job = controller.Job(self.prepare())
        source = self.base / 'candidate'
        source.mkdir()
        (source / 'worker.log').write_text('candidate finished')
        ready = self.base / 'ready/case'
        streams.seal(source=source, destination=ready, names=['worker.log'], metadata={})
        entries = [{'path': 'artifacts/output/candidates/case/' + p.name,
                    'sha256': core.checksum(p), 'bytes': p.stat().st_size} for p in ready.iterdir()]
        objects = {core.checksum(p): p for p in ready.iterdir()}
        manifest = {'format': 'run-artifacts-v1', 'run_id': job.state['run_id'], 'image': job.state['image'],
                    'config_sha256': job.state['config_sha256'], 'state': 'running', 'exit_code': None,
                    'updated': time.time(), 'files': entries}
        copies = []
        def cloud(*, args, **kwargs):
            if args[:2] == ['storage', 'cat']:
                return 0, json.dumps(manifest)
            self.assertEqual(args[:2], ['storage', 'cp'])
            shutil.copyfile(src=objects[args[-2].rsplit('/', 1)[-1]], dst=args[-1])
            copies.append(args)
            return 0, ''
        with patch.object(job, 'gcloud', side_effect=cloud):
            job.collect(live=True)
            target = job.directory.parents[1] / 'candidates' / job.state['run_id'] / 'case'
            self.assertEqual((target / 'files/worker.log').read_text(), 'candidate finished')
            count = len(copies)
            job.collect(live=True)
            self.assertEqual(len(copies), count)
            self.assertFalse(job.state.get('artifacts_verified', False))
            self.assertFalse(job.state.get('resources_cleaned', False))

    def test_final_collection_with_missing_candidate_part_is_not_verified(self):
        self.description['outputs']['delivery'] = 'incremental'
        job = controller.Job(self.prepare())
        source = self.base / 'candidate'
        source.mkdir()
        (source / 'worker.log').write_text('candidate finished')
        ready = self.base / 'ready/case'
        streams.seal(source=source, destination=ready, names=['worker.log'], metadata={})
        path = ready / 'candidate.json'
        manifest = {'format': 'run-artifacts-v1', 'run_id': job.state['run_id'], 'image': job.state['image'],
                    'config_sha256': job.state['config_sha256'], 'state': 'succeeded', 'exit_code': 0,
                    'updated': time.time(), 'files': [{'path': 'artifacts/output/candidates/case/candidate.json',
                    'sha256': core.checksum(path), 'bytes': path.stat().st_size}]}
        def cloud(*, args, **kwargs):
            if args[:2] == ['storage', 'cat']:
                return 0, json.dumps(manifest)
            shutil.copyfile(src=path, dst=args[-1])
            return 0, ''
        with patch.object(job, 'gcloud', side_effect=cloud), patch.object(job, 'cdk', return_value=(0, '')), \
                patch.object(controller.STOP, 'wait'):
            self.assertFalse(job.collect())
        self.assertFalse(job.state['artifacts_verified'])
        self.assertIn('omits a candidate part', job.state['collection_error'])

    def test_runtime_incremental_final_manifest_has_no_outer_archive(self):
        import subprocess
        self.description['outputs']['delivery'] = 'incremental'
        folder = self.prepare()
        bucket = self.base / 'bucket'
        bucket.mkdir()
        shutil.copytree(src=folder / 'input', dst=bucket / 'input')
        shutil.copyfile(src=folder / 'owner.json', dst=bucket / 'owner.json')
        config = json.loads((bucket / 'input/run.json').read_text())
        target = self.base / 'runtime-code'
        config['bundles'][0]['destination'] = str(target)
        config['run']['cwd'] = str(target)
        config['run']['argv'] = [sys.executable, '-c',
            'import os,pathlib; p=pathlib.Path(os.environ["OUTPUT_DIR"]); '
            '(p/".work").mkdir(); (p/".work/raw").write_text("private"); '
            '(p/"result.txt").write_text("ok")']
        for fallback in (False, True):
            if fallback:
                config['run']['argv'][-1] += '; (p/".incomplete-artifacts").write_text("failed packing")'
            core.save(path=bucket / 'input/run.json', value=config)
            digest = core.checksum(bucket / 'input/run.json')
            core.save(path=bucket / 'control/start.json', value={'run_id': config['run_id'], 'config_sha256': digest})
            local = self.base / f'runtime-{fallback}'
            script = 'from pathlib import Path; import runtime,sys; sys.exit(runtime.execute(bucket=Path(sys.argv[1]),local=Path(sys.argv[2]),expected=sys.argv[3],require_mount=False))'
            result = subprocess.run(args=[sys.executable, '-c', script, str(bucket), str(local), digest],
                env={**os.environ, 'PYTHONPATH': str(controller.ROOT), 'RUN_ID': config['run_id']},
                capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((bucket / 'manifest.json').read_text())
            self.assertNotIn('archive', manifest)
            paths = {entry['path'] for entry in manifest['files']}
            self.assertIn('artifacts/output/result.txt', paths)
            self.assertEqual('artifacts/output/.work/raw' in paths, fallback)

    def test_delivery_schema_rejects_unknown_mode(self):
        self.description['outputs']['delivery'] = 'unknown'
        self.write_descriptions()
        with self.assertRaisesRegex(ValueError, 'outputs.delivery'):
            core.load_description(self.base / 'experiment.json')


if __name__ == '__main__':
    unittest.main()
