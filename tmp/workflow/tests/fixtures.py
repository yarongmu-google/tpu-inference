"""Local controller fixture shared by image and resource tests."""
import tempfile
import unittest
from pathlib import Path
import core

class DescriptionFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix='sweep-description-test-')
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.workflow = self.base / 'workflow'
        self.workflow.mkdir()
        self.profile = {'version': 1,
            'cloud': {'project': 'fixture-project', 'region': 'us-central1', 'service_account': 'fixture-worker',
                      'workload_iam_member': 'serviceAccount:fixture@example.invalid'},
            'runtime': {'image': 'us-central1-docker.pkg.dev/fixture-project/images/runtime@sha256:' + 'a' * 64},
            'hardware': {'accelerator': 'tpu7x', 'topology': '2x2x1', 'chips_per_host': 4},
            'storage': {'dedicated_bucket': True, 'deletion': 'manual', 'soft_delete_days': 0}}
        self.description = {
            'version': 1, 'profile': './local/environment.json',
            'code': {'directory': '../scripts', 'delivery': 'snapshot', 'destination': '/workspace/code',
                     'include': ['sweep.sh', 'server.sh', 'bench.sh']},
            'run': {'cwd': '/workspace/code', 'argv': ['bash', 'sweep.sh']},
            'outputs': {'directory': './local/results'},
            'execution': {'timeout_seconds': 43200, 'max_in_flight': 1}}
        # Existing snapshot-only checks also cover compatibility with fixed images.
        self.description['profile'] = './local/environment.json'
        self.description.pop('image_build', None)
        self.description['execution'].pop('cleanup', None)
        self.description['code']['directory'] = '../scripts'
        self.description_path = self.workflow / 'sweep.json'
        core.save(path=self.description_path, value=self.description)
        source = self.base / 'scripts'
        source.mkdir()
        for name in ('sweep.sh', 'server.sh', 'bench.sh'):
            (source / name).write_text('#!/bin/bash\nexit 0\n')
        (source / 'private-unselected.txt').write_text('not included')

    def saved_profile(self) -> Path:
        path = self.workflow / 'local/environment.json'
        core.save(path=path, value=self.profile)
        return path
