"""Avoid duplicate storage without weakening final artifact verification."""
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import controller
import core
import pack_results
import streams
import test_workflow


class FlowEfficiencyTests(unittest.TestCase):
    def test_materialized_parts_share_existing_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'source'
            source.mkdir()
            (source / 'log').write_text('diagnostics')
            sealed = root / 'sealed/case'
            streams.seal(source=source, destination=sealed, names=['log'], metadata={})
            streams.materialize(source=sealed, destination=root / 'received')
            self.assertTrue(os.path.samefile(sealed / 'part-0000.tar.gz', root / 'received/part-0000.tar.gz'))

    def test_final_export_excludes_already_compressed_parts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / 'runs/campaign/job'
            part = directory / 'collected/files/artifacts/output/candidates/case/part-0000.tar.gz'
            part.parent.mkdir(parents=True)
            part.write_bytes(b'preserve separately')
            (part.parent / 'candidate.json').write_text('{}')
            core.save(path=directory / 'state.json', value={'run_id': 'fixture', 'phase': 'VERIFIED',
                 'artifact_delivery': 'incremental', 'artifacts_verified': True})
            archive = pack_results.export_run(directory=directory)
            with tarfile.open(archive) as bundle:
                names = bundle.getnames()
                self.assertFalse(any(name.endswith('part-0000.tar.gz') for name in names))
                self.assertIn('run/separate-artifacts.json', names)
                self.assertTrue(any(name.endswith('candidate.json') for name in names))
            self.assertEqual(part.read_bytes(), b'preserve separately')

    def test_final_receipt_avoids_recopy_but_corruption_still_fails(self):
        fixture = test_workflow.WorkflowTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.description['outputs']['delivery'] = 'incremental'
        job = controller.Job(fixture.prepare())
        raw = fixture.base / 'source'
        raw.write_text('result')
        entry = {'path': 'artifacts/output/result.txt', 'bytes': raw.stat().st_size, 'sha256': core.checksum(raw)}
        manifest = {'format': 'run-artifacts-v1', 'run_id': job.state['run_id'], 'image': job.state['image'],
                    'config_sha256': job.state['config_sha256'], 'state': 'running', 'exit_code': None,
                    'updated': time.time(), 'files': [entry]}
        def cloud(*, args, **kwargs):
            if args[:2] == ['storage', 'cat']:
                return 0, json.dumps(manifest)
            shutil.copyfile(src=raw, dst=args[-1])
            return 0, ''
        with patch.object(job, 'gcloud', side_effect=cloud), patch.object(job, 'cdk', return_value=(0, '')):
            job.collect(live=True)
            manifest.update(state='succeeded', exit_code=0)
            with patch.object(core, 'link_or_copy', side_effect=AssertionError('Should reuse receipt')):
                self.assertTrue(job.collect())
            target = job.directory / 'collected/files' / entry['path']
            target.write_text('corrupt')
            with patch.object(job, 'gcloud', side_effect=lambda **kwargs: (0, json.dumps(manifest))), \
                 patch.object(controller.STOP, 'wait'):
                self.assertFalse(job.collect())
            self.assertFalse(job.state['artifacts_verified'])

class PublisherTests(unittest.TestCase):
    def test_slow_upload_does_not_delay_workload_timeout(self):
        import subprocess
        fixture = test_workflow.WorkflowTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.description['outputs']['delivery'] = 'incremental'
        directory = fixture.prepare()
        bucket = fixture.base / 'bucket'
        bucket.mkdir()
        shutil.copytree(directory / 'input', bucket / 'input')
        shutil.copyfile(directory / 'owner.json', bucket / 'owner.json')
        config = json.loads((bucket / 'input/run.json').read_text())
        target = fixture.base / 'runtime-code'
        config['bundles'][0]['destination'] = str(target)
        config['run']['cwd'] = str(target)
        marker = fixture.base / 'terminated.txt'
        config['run']['argv'] = [sys.executable, '-c',
            'import time,signal,sys; from pathlib import Path; start=time.monotonic(); '
            'signal.signal(signal.SIGTERM,lambda *_: (Path(sys.argv[1]).write_text(str(time.monotonic()-start)),sys.exit(0))); '
            'time.sleep(30)', str(marker)]
        config['timeout_seconds'] = 1
        core.save(path=bucket / 'input/run.json', value=config)
        digest = core.checksum(bucket / 'input/run.json')
        core.save(path=bucket / 'control/start.json', value={'run_id': config['run_id'], 'config_sha256': digest})
        script = '''import time,sys
from pathlib import Path
import runtime
original = runtime.publish
def slow(**kwargs):
    if kwargs['status']['state'] == 'running':
        time.sleep(3)
    return original(**kwargs)
runtime.publish = slow
sys.exit(runtime.execute(bucket=Path(sys.argv[1]), local=Path(sys.argv[2]), expected=sys.argv[3], require_mount=False))
'''
        result = subprocess.run(args=[sys.executable, '-c', script, str(bucket),
            str(fixture.base / 'runtime'), digest], env={**os.environ, 'PYTHONPATH': str(controller.ROOT),
            'RUN_ID': config['run_id']}, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 124, result.stderr)
        self.assertLess(float(marker.read_text()), 2.5)
        manifest = json.loads((bucket / 'manifest.json').read_text())
        self.assertEqual(manifest['state'], 'failed')
        self.assertEqual(manifest['exit_code'], 124)


if __name__ == '__main__':
    unittest.main()
