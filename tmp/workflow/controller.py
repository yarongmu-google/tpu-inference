"""Submit, resume, monitor, collect, and clean described jobs."""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import configparser
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid

from core import cdk_storage_uri, uses_cdk_storage, checksum, identity, json_bytes, load_description, read_document, safe_path, save, slug, snapshot, verify

ROOT = Path(__file__).resolve().parent
LETTER_HASH = '4093b4cb6404558219be50b15b409be1485fd7f64405eb20030c0e9b062ce31f'
TERMINAL = {'Succeeded', 'Failed', 'Deleted'}
ARCHIVE_FAILED = {'LogArchiveFailed', 'TraceGenFailed', 'FinalizingFailed', 'K8sJobDeleted'}
TRANSFERS = threading.BoundedSemaphore(4)
REQUESTS = threading.BoundedSemaphore(8)
STOP = threading.Event()


def recipe(state: dict, payload: dict[str, str]) -> dict:
    encoded = base64.b64encode(gzip.compress(json_bytes(payload), mtime=0)).decode()
    bootstrap = ('import base64,gzip,json,os,sys; from pathlib import Path; '
                 'p=Path("/run-bundle"); p.mkdir(exist_ok=True); '
                 'files=json.loads(gzip.decompress(base64.b64decode(sys.argv[1]))); '
                 '[(p/k).write_text(v) for k,v in files.items()]; '
                 'os.execv(sys.executable,[sys.executable,"-u",str(p/"runtime.py")])')
    hardware = state['profile']['hardware']
    labels = {'run-name': state['label'], 'run-id': state['run_id']}
    value = {'apiVersion': 'jobset.x-k8s.io/v1alpha2', 'kind': 'JobSet',
            'metadata': {'name': state['run_id'], 'labels': labels},
            'spec': {'failurePolicy': {'maxRestarts': 0}, 'replicatedJobs': [{
                'name': 'worker', 'replicas': 1, 'template': {'spec': {
                    'backoffLimit': 0, 'parallelism': 1, 'completions': 1,
                    'template': {'metadata': {'labels': labels, 'annotations': {'gke-gcsfuse/volumes': 'true'}},
                                 'spec': {'restartPolicy': 'Never',
                                          'nodeSelector': {'cloud.google.com/gke-tpu-accelerator': hardware['accelerator'],
                                                           'cloud.google.com/gke-tpu-topology': hardware['topology']},
                                          'containers': [{'name': 'runner', 'image': state['image'],
                                                          'command': ['python3', '-u', '-c', bootstrap, encoded],
                                                          'env': [{'name': 'RUN_ID', 'value': state['run_id']},
                                                                  {'name': 'RUN_CONFIG_SHA256', 'value': state['config_sha256']}],
                                                          'resources': {'requests': {'google.com/tpu': hardware['chips_per_host']},
                                                                        'limits': {'google.com/tpu': hardware['chips_per_host']}},
                                                          'volumeMounts': [{'name': 'run-storage', 'mountPath': '/run-storage'},
                                                                           {'name': 'shared-memory', 'mountPath': '/dev/shm'}]}],
                                          'volumes': [{'name': 'run-storage', 'csi': {'driver': 'gcsfuse.csi.storage.gke.io',
                                                       'readOnly': False, 'volumeAttributes': {'bucketName': state['bucket'], 'mountOptions': 'implicit-dirs'}}},
                                                      {'name': 'shared-memory', 'emptyDir': {'medium': 'Memory'}}]}}}}}]}}
    if uses_cdk_storage(state['profile']):
        template = value['spec']['replicatedJobs'][0]['template']['spec']['template']
        template['metadata'].pop('annotations')
        pod = template['spec']
        pod['volumes'] = [v for v in pod['volumes'] if v['name'] != 'run-storage']
        runner = pod['containers'][0]
        runner['volumeMounts'] = [v for v in runner['volumeMounts'] if v['name'] != 'run-storage']
        runner['env'].append({'name': 'WORKFLOW_STORAGE_MODE', 'value': 'cdk'})
    return value


def validate_recipe(actual: dict, expected: dict, service_account: str | None) -> None:
    jobs = actual['spec']['replicatedJobs']
    if len(jobs) != 1 or jobs[0]['name'] != 'worker' or jobs[0]['replicas'] != 1:
        raise ValueError('Rendered job topology differs from the description')
    a = jobs[0]['template']['spec']
    e = expected['spec']['replicatedJobs'][0]['template']['spec']
    for key in ('parallelism', 'completions', 'backoffLimit'):
        if a.get(key) != e[key]:
            raise ValueError(f'Rendered {key} differs')
    pod = a['template']['spec']
    epod = e['template']['spec']
    if (service_account is not None and pod.get('serviceAccountName') != service_account) or pod.get('restartPolicy') != 'Never':
        raise ValueError('CDK assigned an unexpected service account or restart policy')
    if pod.get('nodeSelector') != epod['nodeSelector']:
        raise ValueError('Rendered hardware selectors differ')
    cdk_storage = {'name': 'WORKFLOW_STORAGE_MODE', 'value': 'cdk'} in epod['containers'][0]['env']
    if not cdk_storage and a['template']['metadata'].get('annotations', {}).get('gke-gcsfuse/volumes') != 'true':
        raise ValueError('Dedicated storage driver annotation was removed')
    containers = [c for c in pod['containers'] if c['name'] == 'runner']
    if len(containers) != 1:
        raise ValueError('Rendered runner container missing or duplicated')
    for key in ('image', 'command'):
        if containers[0].get(key) != epod['containers'][0][key]:
            raise ValueError(f'Rendered runner {key} differs')
    resources = containers[0].get('resources')
    wanted = epod['containers'][0]['resources']
    if not isinstance(resources, dict) or resources.keys() != wanted.keys():
        raise ValueError('Rendered runner resource sections differ')
    for kind, quantities in wanted.items():
        values = resources[kind]
        if not isinstance(values, dict) or values.keys() != quantities.keys():
            raise ValueError(f'Rendered runner {kind} resource names differ')
        for resource, count in quantities.items():
            value = values[resource]
            # CDK serializes integer resource quantities as decimal strings.
            if type(value) not in (int, str) or str(value) != str(count):
                raise ValueError(f'Rendered runner {kind}.{resource} differs: expected {count}, got {value!r}')
    for item in epod['containers'][0]['env']:
        if sum(v == item for v in containers[0].get('env', [])) != 1:
            raise ValueError('Rendered execution identity differs')
    expected_mount = epod['containers'][0]['volumeMounts'][0]
    if sum(v == expected_mount for v in containers[0].get('volumeMounts', [])) != 1:
        raise ValueError('Dedicated output mount missing or changed')
    if sum(v == epod['volumes'][0] for v in pod.get('volumes', [])) != 1:
        raise ValueError('Dedicated output bucket missing or changed')


class Job:
    def __init__(self, directory: Path):
        self.directory = directory
        self.state = json.loads((directory / 'state.json').read_text())
        (directory / 'commands').mkdir(exist_ok=True)

    def save(self, **updates: object) -> None:
        self.state.update(updates)
        save(path=self.directory / 'state.json', value=self.state)

    def phase(self, value: str) -> None:
        self.save(phase=value)
        print(f'[{self.state["run_id"]}] {value}', flush=True)

    def command(self, args: list[str], timeout: int = 180, check: bool = True, transfer: bool = False) -> tuple[int, str]:
        number = self.state.get('command_number', 0) + 1
        self.save(command_number=number)
        base = self.directory / 'commands' / f'{number:05d}-{Path(args[0]).name}'
        print(f'[{self.state["run_id"]}] + {shlex.join(args)}\n  output: {base}.stdout / .stderr', flush=True)
        semaphore = TRANSFERS if transfer else REQUESTS
        with semaphore, base.with_suffix('.stdout').open('wb') as stdout, base.with_suffix('.stderr').open('wb') as stderr:
            child = subprocess.Popen(args=args, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, start_new_session=True)
            code = 1
            try:
                deadline = time.monotonic() + timeout
                while True:
                    if STOP.is_set():
                        code = 130
                        raise InterruptedError('Controller stopped; remote job remains available')
                    try:
                        code = child.wait(timeout=min(10, max(0, deadline - time.monotonic())))
                        break
                    except subprocess.TimeoutExpired:
                        if time.monotonic() >= deadline:
                            code = 124
                            raise
                        print(f'[{self.state["run_id"]}] {self.state["phase"]}: command still running', flush=True)
            finally:
                if child.poll() is None:
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(child.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        child.wait()
                base.with_suffix('.exit').write_text(str(code) + '\n')
        text = base.with_suffix('.stdout').read_text(errors='replace')
        if code:
            error_log = base.with_suffix('.stderr')
            source = error_log if error_log.stat().st_size else base.with_suffix('.stdout')
            with source.open('rb') as stream:
                stream.seek(max(0, source.stat().st_size - 4096))
                detail = stream.read().decode(errors='replace').strip()
            print(f'[{self.state["run_id"]}] Command FAILED (exit {code}); {source}', file=sys.stderr, flush=True)
            if detail:
                print(detail, file=sys.stderr, flush=True)
        if check and code:
            raise RuntimeError(f'Command exit {code}; see {base}.stderr')
        return code, text

    def gcloud(self, args: list[str], **kwargs) -> tuple[int, str]:
        return self.command(args=['gcloud', '--project', self.state['profile']['cloud']['project'], *args], **kwargs)

    def cdk(self, args: list[str], **kwargs) -> tuple[int, str]:
        if 'NO_UPDATE_CHECK' in os.environ:
            raise ValueError('Unset NO_UPDATE_CHECK')
        _, letter = self.command(args=['cdk', 'agent-letter'])
        normal = ' '.join(re.sub(r'\[ack \d+/\d+: [A-Za-z0-9]+\]', '[ack]', letter).split())
        if hashlib.sha256(normal.encode()).hexdigest() != LETTER_HASH:
            raise ValueError('CDK instructions changed; review the saved command output')
        codes = re.findall(r'\[ack (\d+)/(\d+): ([A-Za-z0-9]+)\]', letter)
        if not codes or any(int(i) != n or int(total) != len(codes) for n, (i, total, _) in enumerate(codes, 1)):
            raise ValueError('Incomplete CDK acknowledgement')
        return self.command(args=['cdk', '--agent-code=' + '-'.join(c for _, _, c in codes), *args], **kwargs)

    def discover(self) -> dict | None:
        _, text = self.cdk(args=['job', 'list', '-H', '-n', '100', '-t', self.state['run_id'], '-o', 'json'])
        rows = json.loads(text)
        if not isinstance(rows, list) or len(rows) > 1:
            raise ValueError('Expected at most one job for this run')
        if not rows:
            return None
        row = rows[0]
        tags = row.get('tags', [])
        tags = tags.split(',') if isinstance(tags, str) else tags
        if (row.get('user') != self.state['user'] or row.get('recipe') != self.state['recipe']
                or self.state['run_id'] not in tags or not re.fullmatch('j-[A-Za-z0-9-]+', row.get('id', ''))
                or self.state.get('job_id', row['id']) != row['id']):
            raise ValueError('Job identity does not match saved execution')
        self.save(job_id=row['id'], last_job=row)
        if uses_cdk_storage(self.state['profile']):
            uri = cdk_storage_uri(self.state)
            if self.state.get('uri') not in {None, uri}:
                raise ValueError('Saved CDK storage prefix differs')
            self.save(uri=uri)
        return row

    def bucket_identity(self) -> dict:
        _, text = self.gcloud(args=['storage', 'buckets', 'describe', self.state['uri'], '--raw', '--format=json'])
        metadata = json.loads(text)
        if metadata.get('name') != self.state['bucket']:
            raise ValueError('Unexpected bucket identity')
        return metadata

    def ensure_bucket(self) -> None:
        self.phase('PREPARING_STORAGE')
        if not self.state.get('bucket_created'):
            if self.state.get('bucket_creation_started'):
                raise RuntimeError('Bucket creation outcome is uncertain; inspect it before recording ownership or starting a new run')
            self.save(bucket_creation_started=True)
            self.gcloud(args=['storage', 'buckets', 'create', self.state['uri'],
                              '--location', self.state['profile']['cloud']['region'], '--default-storage-class=STANDARD',
                              '--uniform-bucket-level-access', '--public-access-prevention', '--soft-delete-duration=0'])
            self.save(bucket_created=True)
            self.gcloud(args=['storage', 'buckets', 'update', self.state['uri'],
                              '--update-labels=run_name=' + self.state['label'] + ',run_id=' + self.state['run_id']])
            self.gcloud(args=['storage', 'cp', str(self.directory / 'owner.json'), self.state['uri'] + '/owner.json'])
            self.save(bucket_marked=True)
        if not self.state.get('bucket_marked'):
            raise RuntimeError('Bucket exists but its ownership marker is incomplete; manual inspection required')
        metadata = self.bucket_identity()
        if metadata.get('labels', {}).get('run_id') != self.state['run_id']:
            raise ValueError('Bucket ownership label differs')
        seconds = metadata.get('softDeletePolicy', {}).get('retentionDurationSeconds', '0')
        if int(seconds) != 0 or metadata.get('versioning', {}).get('enabled') or metadata.get('retentionPolicy'):
            raise ValueError('Bucket retention differs from the manual-cleanup profile')
        _, text = self.gcloud(args=['storage', 'cat', self.state['uri'] + '/owner.json'])
        if json.loads(text) != json.loads((self.directory / 'owner.json').read_text()):
            raise ValueError('Bucket ownership marker differs')
        if not self.state.get('access_configured'):
            self.gcloud(args=['storage', 'buckets', 'add-iam-policy-binding', self.state['uri'],
                              '--member', self.state['profile']['cloud']['workload_iam_member'],
                              '--role=roles/storage.objectUser', '--condition=None'])
            self.save(access_configured=True)

    def register(self) -> None:
        settings = configparser.ConfigParser(interpolation=None)
        ini = Path.home() / '.cdk.ini'
        root = Path.home() / 'cloud-devkit'
        if ini.exists():
            content = ini.read_text()
            settings.read_string(content if content.lstrip().startswith('[') else '[DEFAULT]\n' + content)
            for section in [settings.defaults(), *(settings[s] for s in settings.sections())]:
                if section.get('cdk_root_dir'):
                    root = Path(section['cdk_root_dir']).expanduser()
        root = Path(os.environ.get('CDK_SOURCE_DIR', str(root))).resolve()
        registry = root / 'recipes.yml'
        if not registry.is_file():
            raise ValueError('CDK recipe registry missing; configure CDK_SOURCE_DIR')
        relative = 'recipes/experimental/' + self.state['recipe'] + '/jobset.yml'
        line = '- ' + json.dumps({'name': self.state['recipe'], 'owner': self.state['user'],
                                 'k8s_file': relative, 'require_gcs_mount': uses_cdk_storage(self.state['profile'])})
        with (root / '.description-workflow.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            before = registry.read_text()
            if before.lstrip().startswith('['):
                raise ValueError('Expected block-style CDK recipe registry')
            _, text = self.cdk(args=['recipe', 'list', '-o', 'json'])
            matches = [r for r in json.loads(text) if r['name'] == self.state['recipe']]
            if matches and (len(matches) != 1 or line not in before.splitlines()):
                raise ValueError('Recipe name collision')
            target = root / relative
            wanted = (self.directory / 'recipe.json').read_bytes()
            if target.exists() and target.read_bytes() != wanted:
                raise ValueError('Existing recipe differs')
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(wanted)
            if not matches:
                temporary = registry.with_name('.workflow-recipes.new')
                temporary.write_text(before.rstrip() + '\n' + line + '\n')
                shutil.copymode(src=registry, dst=temporary)
                if registry.read_text() != before:
                    raise ValueError('Recipe registry changed concurrently')
                temporary.replace(registry)
        _, text = self.cdk(args=['recipe', 'list', '-o', 'json'])
        if sum(r['name'] == self.state['recipe'] for r in json.loads(text)) != 1:
            raise ValueError('Registered recipe was not discovered')

    def authorize(self) -> None:
        _, text = self.cdk(args=['job', 'recipe', self.state['job_id'], '--no-color'])
        path = self.directory / 'rendered.yml'
        path.write_text(text)
        validate_recipe(actual=read_document(path), expected=json.loads((self.directory / 'recipe.json').read_text()),
                        service_account=self.state['profile']['cloud'].get('service_account'))
        rendered = read_document(path)
        account = rendered['spec']['replicatedJobs'][0]['template']['spec']['template']['spec'].get('serviceAccountName')
        self.save(assigned_service_account=account)
        self.gcloud(args=['storage', 'cp', str(self.directory / 'start.json'), self.state['uri'] + '/control/start.json'])
        self.save(authorized=True)

    def collect(self, live: bool = False) -> bool:
        destination = self.directory / ('live' if live else 'collected')
        destination.mkdir(exist_ok=True)
        if not live:
            self.save(artifacts_verified=False)
            self.phase('COLLECTING')
            if self.state.get('job_id'):
                for args, filename in [(['job', 'desc', self.state['job_id'], '--no-color'], 'job.yml'),
                                       (['job', 'log', self.state['job_id'], '-c', 'runner'], 'container.log')]:
                    try:
                        _, text = self.cdk(args=args, check=False)
                        (destination / filename).write_text(text)
                    except Exception:
                        (destination / (filename + '.error')).write_text(traceback.format_exc())
        for attempt in range(1 if live else 3):
            try:
                _, text = self.gcloud(args=['storage', 'cat', self.state['uri'] + '/manifest.json'])
                manifest = json.loads(text)
                save(path=destination / 'manifest.json', value=manifest)
                if (manifest.get('format') != 'run-artifacts-v1' or manifest.get('run_id') != self.state['run_id']
                        or manifest.get('image') != self.state['image'] or manifest.get('config_sha256') != self.state['config_sha256']):
                    raise ValueError('Artifact manifest identity differs')
                if live and 'archive' in manifest:
                    self.save(last_output=manifest.get('updated'), workload_state=manifest['state'])
                    print(f'[{self.state["run_id"]}] final results compressed; waiting for CDK archival', flush=True)
                    return False
                if not live and 'archive' in manifest:
                    from artifacts import verify_bundle
                    archive = manifest['archive']
                    digest = archive.get('sha256', '')
                    if (not re.fullmatch('[a-f0-9]{64}', digest)
                            or archive.get('path') != 'archives/' + digest + '.tar.gz'
                            or type(archive.get('bytes')) is not int or archive['bytes'] <= 0):
                        raise ValueError('Invalid compressed artifact reference')
                    packed = destination / 'artifacts.tar.gz'
                    if not packed.exists() or checksum(packed) != digest:
                        self.gcloud(args=['storage', 'cp', self.state['uri'] + '/' + archive['path'], str(packed)],
                                    timeout=1800, transfer=True)
                    if packed.stat().st_size != archive['bytes'] or checksum(packed) != digest:
                        raise ValueError('Downloaded archive checksum mismatch')
                    verify_bundle(path=packed, manifest={key: value for key, value in manifest.items() if key != 'archive'},
                                  target=destination / 'files')
                else:
                    objects = destination / 'objects'
                    objects.mkdir(exist_ok=True)
                    paths = set()
                    selected = [entry for entry in manifest['files'] if not live or entry['path'] in
                                {'diagnostics/stdout.log', 'diagnostics/stderr.log', 'diagnostics/error.txt', 'diagnostics/status.json'}]
                    for entry in selected:
                        if not re.fullmatch('[a-f0-9]{64}', entry['sha256']) or entry['path'] in paths:
                            raise ValueError('Invalid or repeated manifest entry')
                        paths.add(entry['path'])
                        target = safe_path(root=destination / 'files', relative=entry['path'])
                        blob = objects / entry['sha256']
                        if not blob.exists() or checksum(blob) != entry['sha256']:
                            self.gcloud(args=['storage', 'cp', self.state['uri'] + '/objects/' + entry['sha256'], str(blob)],
                                        timeout=1800, transfer=True)
                        if blob.stat().st_size != entry['bytes'] or checksum(blob) != entry['sha256']:
                            raise ValueError('Downloaded artifact checksum mismatch')
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(src=blob, dst=target)
                    verify(root=destination / 'files', entries=selected)
                if live:
                    self.save(last_output=manifest.get('updated'), workload_state=manifest['state'])
                    age = max(0, time.time() - manifest.get('updated', time.time()))
                    print(f'[{self.state["run_id"]}] workload={manifest["state"]}; snapshot age={age:.0f}s', flush=True)
                    for filename in ('stdout.log', 'stderr.log'):
                        path = destination / 'files/diagnostics' / filename
                        if path.exists():
                            with path.open('rb') as stream:
                                stream.seek(max(0, path.stat().st_size - 512))
                                print(stream.read().decode(errors='replace').rstrip(), flush=True)
                    return False
                if manifest['state'] not in {'succeeded', 'failed'} or type(manifest['exit_code']) is not int:
                    raise ValueError('Final manifest not yet available')
                self.save(artifacts_verified=True, artifact_exit_code=manifest['exit_code'])
                return manifest['exit_code'] == 0 and manifest['state'] == 'succeeded'
            except Exception as error:
                self.save(collection_error=str(error))
                if attempt < (0 if live else 2):
                    STOP.wait(5)
        return False

    def upload_cdk_inputs(self) -> None:
        if self.state.get('uploaded'):
            return
        if self.state.get('uri') != cdk_storage_uri(self.state) or not self.state.get('job_id'):
            raise ValueError('CDK job storage has not been resolved')
        self.phase('UPLOADING')
        bundle = self.directory / 'input'
        for item in read_document(bundle / 'run.json')['bundles']:
            verify(root=bundle / item['source'], entries=item['files'])
        if checksum(bundle / 'run.json') != self.state['config_sha256']:
            raise ValueError('Saved configuration changed')
        self.save(cdk_upload_started=True)
        self.gcloud(args=['storage', 'cp', str(self.directory / 'owner.json'), self.state['uri'] + '/owner.json'])
        self.gcloud(args=['storage', 'rsync', '--recursive', str(bundle), self.state['uri'] + '/input'],
                    timeout=14400, transfer=True)
        self.save(uploaded=True, bucket_marked=True)

    def prepare_image(self) -> None:
        if not self.state.get('owned_image'):
            return
        if self.state.get('resources_cleaned'):
            raise ValueError('Run resources were cleaned; start a new run to rebuild')
        from image_build import IMAGE, resolve
        if not self.state.get('image_ready'):
            if self.state.get('image_build_started'):
                result = self.directory / 'image/result.json'
                if not result.is_file():
                    raise RuntimeError('Previous image preparation did not finish; clean this run and start a new one')
                value = read_document(result)
            else:
                self.phase('BUILDING_IMAGE')
                self.save(image_build_started=True)
                value = resolve(build=self.state['image_build'], repository=self.state['owned_image'],
                    directory=self.directory / 'image', stop=STOP, run_id=self.state['run_id'])
            image = value.get('image', '')
            if (not isinstance(image, str) or not IMAGE.fullmatch(image)
                    or not image.startswith(self.state['owned_image'] + '@')
                    or value.get('owner_run_id') != self.state['run_id']):
                raise ValueError('Image result does not belong to this run')
            save(path=self.directory / 'image-build.json', value=value)
            self.save(image=image, image_ready=True)
        config = read_document(self.directory / 'run-template.json')
        payload = read_document(self.directory / 'runtime-payload.json')
        if hashlib.sha256(json_bytes(payload)).hexdigest() != config['runtime_sha256']:
            raise ValueError('Saved runtime payload changed')
        config['image'] = self.state['image']
        expected = hashlib.sha256(json_bytes(config)).hexdigest()
        if self.state.get('config_sha256') not in {None, expected}:
            raise ValueError('Saved configuration changed after image preparation')
        save(path=self.directory / 'input/run.json', value=config)
        self.save(config_sha256=checksum(self.directory / 'input/run.json'), phase='PREPARED')
        save(path=self.directory / 'start.json', value={'run_id': self.state['run_id'],
                                                     'config_sha256': self.state['config_sha256']})
        save(path=self.directory / 'recipe.json', value=recipe(state=self.state, payload=payload))

    def finish_cleanup(self) -> None:
        if self.state['execution'].get('cleanup') == 'after_collection' and not self.state.get('resources_cleaned'):
            self.cleanup(discard=False, automatic=True)

    def run(self) -> int:
        if self.state.get('finished'):
            if not self.state.get('artifacts_verified') and self.state.get('submission_started'):
                self.collect()
            self.finish_cleanup()
            return self.state['exit_code']
        self.prepare_image()
        config = self.state['execution']
        wait_started = time.monotonic()
        job = None
        if self.state.get('submission_started'):
            job = self.discover()
            if job is None:
                raise RuntimeError('Submission outcome unresolved; not submitting again')
        else:
            if not uses_cdk_storage(self.state['profile']):
                self.ensure_bucket()
                self.phase('UPLOADING')
                if not self.state.get('uploaded'):
                    bundle = self.directory / 'input'
                    for item in json.loads((bundle / 'run.json').read_text())['bundles']:
                        verify(root=bundle / item['source'], entries=item['files'])
                    if checksum(bundle / 'run.json') != self.state['config_sha256']:
                        raise ValueError('Saved configuration changed')
                    self.gcloud(args=['storage', 'rsync', '--recursive', str(bundle), self.state['uri'] + '/input'],
                                timeout=14400, transfer=True)
                    self.save(uploaded=True)
            self.register()
            self.phase('SUBMITTING')
            self.save(submission_started=True, submitted_at=time.time())
            code, _ = self.cdk(args=['job', 'create', self.state['recipe'],
                                     '--tags', self.state['label'] + ',' + self.state['run_id'],
                                     '--mount-gcs=true' if uses_cdk_storage(self.state['profile']) else '--mount-gcs=false', '--log-mode=log-transport',
                                     '--active-deadline-seconds', str(config['timeout_seconds'] + 1800)],
                               check=False, timeout=300)
            self.save(submission_exit=code)
            for _ in range(3):
                job = self.discover()
                if job:
                    break
                STOP.wait(5)
            if job is None:
                raise RuntimeError('No confirmed job ID; resume to reconcile submission')
        if uses_cdk_storage(self.state['profile']):
            self.upload_cdk_inputs()
        if not self.state.get('authorized'):
            self.authorize()
        terminal_since = None
        errors = 0
        while not STOP.is_set():
            status, archive = job.get('job_status'), job.get('state')
            self.phase(status.upper() if isinstance(status, str) else 'UNKNOWN')
            print(f'[{self.state["run_id"]}] job={job["id"]}; execution={status}; archive={archive}', flush=True)
            if status in TERMINAL:
                if terminal_since is None:
                    terminal_since = time.monotonic()
                if archive == 'Complete' or archive in ARCHIVE_FAILED:
                    success = self.collect()
                    code = 0 if status == 'Succeeded' and archive == 'Complete' and success else 1
                    self.save(finished=True, exit_code=code, phase='VERIFIED' if code == 0 else 'FAILED')
                    self.finish_cleanup()
                    return code
                if time.monotonic() - terminal_since > config['archive_seconds']:
                    raise TimeoutError('Archive wait expired; resume to continue collection')
            elif status == 'Running':
                self.collect(live=True)
            if time.monotonic() - wait_started > config['wait_seconds']:
                raise TimeoutError('Job wait expired; remote job remains available')
            STOP.wait(config['poll_seconds'])
            try:
                job = self.discover()
                if job is None:
                    raise ValueError('Saved job not found')
                errors = 0
            except Exception:
                errors += 1
                if errors >= 3:
                    raise
        raise InterruptedError('Controller interrupted; resume the saved run')

    def cleanup(self, discard: bool, automatic: bool = False) -> int:
        if self.state.get('owned_image'):
            from resources import cleanup
            return cleanup(job=self, discard=discard, automatic=automatic)
        if automatic:
            raise ValueError('Automatic cleanup cannot adopt resources from an older run')
        if self.state.get('deleted'):
            print('Bucket already recorded as deleted')
            return 0
        if self.state.get('submission_started'):
            job = self.discover()
            if job is None or job.get('job_status') not in TERMINAL or job.get('state') not in ({'Complete'} | ARCHIVE_FAILED):
                raise ValueError('Job is active, archiving, or unresolved; cleanup is blocked')
        if uses_cdk_storage(self.state['profile']):
            from resources import clean_cdk_storage, verify_collected
            if self.state.get('uri') != cdk_storage_uri(self.state):
                raise ValueError('CDK storage ownership differs')
            if not discard:
                verify_collected(job=self)
            print(f'Delete {self.state["uri"]}; keep the supplied image and local files', flush=True)
            expected = 'DELETE ' + self.state['run_id']
            if not sys.stdin.isatty() or input(f'Type {expected} to confirm: ') != expected:
                raise ValueError('Deletion was not confirmed')
            clean_cdk_storage(job=self)
            return 0
        if not self.state.get('bucket_marked'):
            raise ValueError('Bucket ownership is unconfirmed; cleanup is blocked')
        # Check exact ownership immediately before offering deletion.
        metadata = self.bucket_identity()
        _, text = self.gcloud(args=['storage', 'cat', self.state['uri'] + '/owner.json'])
        if metadata.get('labels', {}).get('run_id') != self.state['run_id'] or json.loads(text) != json.loads((self.directory / 'owner.json').read_text()):
            raise ValueError('Bucket ownership mismatch')
        if not discard:
            if not self.state.get('artifacts_verified'):
                raise ValueError('Results are not verified; collect first or explicitly use --discard-incomplete')
            manifest = json.loads((self.directory / 'collected/manifest.json').read_text())
            if (manifest.get('run_id') != self.state['run_id'] or manifest.get('config_sha256') != self.state['config_sha256']
                    or manifest.get('image') != self.state['image'] or manifest.get('state') not in {'succeeded', 'failed'}):
                raise ValueError('Collected manifest does not describe this completed run')
            verify(root=self.directory / 'collected/files', entries=manifest['files'])
        print(f'Delete {self.state["uri"]}\nKeep local files at {self.directory}', flush=True)
        expected = 'DELETE ' + self.state['run_id']
        if not sys.stdin.isatty() or input(f'Type {expected} to confirm: ') != expected:
            raise ValueError('Deletion was not confirmed')
        self.save(deletion_started=True)
        self.gcloud(args=['storage', 'rm', '--recursive', '--quiet', self.state['uri']], timeout=1800, transfer=True)
        self.save(deleted=True)
        return 0


def prepare(description_path: Path, name: str | None, dry_run: bool = False) -> Path:
    data, profile = load_description(description_path)
    name = name or data.get('name')
    if not name:
        if not sys.stdin.isatty():
            raise ValueError('Supply --name or name in the description for noninteractive execution')
        name = input('Run name (used in cloud labels and resource names): ').strip()
    label = slug(name)
    campaign_id = identity(name)
    root = Path(data['outputs']['directory']) / campaign_id
    # Freeze experiment files before image preparation or job submission.
    for source in [Path(data['code']['directory']), *(Path(item['source']) for item in data['inputs'].values())]:
        if root.is_relative_to(source):
            raise ValueError('Results directory must be outside all uploaded source/input directories')
    root.mkdir(parents=True, exist_ok=False)
    payload = {key: (ROOT / key).read_text() for key in ('core.py', 'runtime.py', 'artifacts.py')}
    save(path=root / 'description.json', value=data)
    shared = root / 'snapshot'
    bundles = [{'source': 'code', 'destination': data['code']['destination'],
                'files': snapshot(source=Path(data['code']['directory']), target=shared / 'code', include=data['code']['include'])}]
    for key, item in data['inputs'].items():
        bundles.append({'source': 'inputs/' + key, 'destination': item['destination'],
                        'files': snapshot(source=Path(item['source']), target=shared / 'inputs' / key, include=['**'])})
    save(path=root / 'profile.json', value=profile)
    user = os.environ.get('USER', '')
    if not re.fullmatch('[a-zA-Z0-9_.-]+', user):
        raise ValueError('USER must identify the submitting CDK user')
    jobs = []
    for case in data['cases']:
        run_id = identity(name + '-' + case['name'])
        folder = root / run_id
        folder.mkdir()
        shutil.copytree(src=shared, dst=folder / 'input', copy_function=os.link)
        nonce = uuid.uuid4().hex
        run = {**data['run'], 'env': {**data['run']['env'], **case['env']}}
        config = {'format': 'run-description-v1', 'run_id': run_id, 'name': name, 'nonce': nonce,
                  'image': None if 'image_build' in data else profile['runtime']['image'], 'run': run,
                  'inputs': {key: {'destination': item['destination']} for key, item in data['inputs'].items()},
                  'outputs': {key: data['outputs'][key] for key in ('extra', 'snapshot_seconds')},
                  'bundles': bundles, 'runtime_sha256': hashlib.sha256(json_bytes(payload)).hexdigest(), 'timeout_seconds': data['execution']['timeout_seconds']}
        save(path=folder / 'run-template.json', value=config)
        save(path=folder / 'runtime-payload.json', value=payload)
        if 'image_build' not in data:
            save(path=folder / 'input/run.json', value=config)
        digest = None if 'image_build' in data else checksum(folder / 'input/run.json')
        state = {'run_id': run_id, 'name': name, 'label': label, 'case': case['name'],
                 'bucket': None if uses_cdk_storage(profile) else run_id,
                 'uri': None if uses_cdk_storage(profile) else 'gs://' + run_id, 'recipe': run_id, 'user': user,
                 'image': config['image'], 'profile': profile, 'execution': data['execution'],
                 'config_sha256': digest, 'phase': 'PREPARED', 'dry_run': dry_run}
        save(path=folder / 'owner.json', value={'run_id': run_id, 'nonce': nonce, 'project': profile['cloud']['project']})
        if 'image_build' in data:
            state.update(image_build=data['image_build'],
                         owned_image=profile['runtime']['repository'].rstrip('/') + '/' + run_id)
            save(path=folder / 'resources.json', value={'run_id': run_id, 'bucket': state['bucket'],
                'image': state['owned_image'], 'local_tags': ['runtime:' + run_id, state['owned_image'] + ':run']})
        else:
            save(path=folder / 'start.json', value={'run_id': run_id, 'config_sha256': digest})
            save(path=folder / 'recipe.json', value=recipe(state=state, payload=payload))
        save(path=folder / 'state.json', value=state)
        jobs.append(run_id)
    save(path=root / 'campaign.json', value={'name': name, 'jobs': jobs, 'max_in_flight': data['execution']['max_in_flight'], 'dry_run': dry_run})
    print(f'Prepared {len(jobs)} jobs at {root}', flush=True)
    return root


def job_action(job: Job, action: str, discard: bool) -> int:
    directory = job.directory
    try:
        if job.state.get('dry_run') and action != 'status':
            raise ValueError('Dry-run artifacts cannot be submitted; start a new run from the description')
        if action == 'cleanup':
            return job.cleanup(discard=discard)
        if action == 'collect':
            if job.state.get('owned_image') and job.state.get('deleted'):
                from resources import verify_collected
                verify_collected(job=job)
                job.finish_cleanup()
                return 0 if job.state.get('artifact_exit_code') == 0 else 1
            success = job.collect()
            if job.state.get('artifacts_verified'):
                job.finish_cleanup()
            return 0 if success else 1
        return job.run()
    except Exception as error:
        job.save(last_error=str(error))
        (directory / 'error.txt').write_text(traceback.format_exc())
        print(f'[{job.state["run_id"]}] stopped: {error}', file=sys.stderr, flush=True)
        if (action == 'run' and not STOP.is_set() and job.state.get('bucket_marked')
                and not job.state.get('artifacts_verified') and not job.state.get('deleted')):
            try:
                job.collect()
            except Exception:
                (directory / 'collection-error.txt').write_text(traceback.format_exc())
        if (action == 'run' and job.state.get('owned_image') and not STOP.is_set()
                and not job.state.get('submission_started') and not job.state.get('bucket_creation_started')):
            job.save(finished=True, exit_code=1)
            try:
                job.finish_cleanup()
            except Exception as cleanup_error:
                job.save(cleanup_error=str(cleanup_error))
        return 1


def locked_job(directory: Path, action: str = 'run', discard: bool = False) -> int:
    with (directory / '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f'Another controller owns {directory}', file=sys.stderr)
            return 1
        job = Job(directory)
        result = job_action(job=job, action=action, discard=discard)
        submission = job.state.get('job_id') or ('unresolved' if job.state.get('submission_started') else 'not submitted')
        outcome = 'FAILED' if result else 'completed'
        print(f'[{job.state["run_id"]}] {action} {outcome} (exit {result}); '
              f'phase={job.state["phase"]}; CDK job={submission}', flush=True)
        try:
            from pack_results import export_run
            export_run(directory=directory)
        except Exception as archive_error:
            (directory / 'archive-error.txt').write_text(traceback.format_exc())
            print(f'Result compression failed: {archive_error}; raw files retained at {directory}', file=sys.stderr)
            result = 1
        print(f'Results and diagnostics: {directory}', flush=True)
        print('Manual cleanup: bash tmp/workflow/run.sh cleanup ' + shlex.quote(str(directory)), flush=True)
        return result


def run_campaign(path: Path) -> int:
    campaign = json.loads((path / 'campaign.json').read_text())
    if campaign.get('dry_run'):
        raise ValueError('Dry-run campaigns cannot be submitted; start a new run from the description')
    with (path / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Count unresolved submissions as occupying a slot, even if a worker exits.
        pending = [path / name for name in campaign['jobs']]
        def already_submitted(directory: Path) -> bool:
            state = json.loads((directory / 'state.json').read_text())
            return bool(state.get('submission_started') and not state.get('finished'))
        pending.sort(key=lambda directory: not already_submitted(directory))
        if sum(already_submitted(directory) for directory in pending) > campaign['max_in_flight']:
            raise ValueError('Saved active jobs exceed the configured limit')
        failures = 0
        with ThreadPoolExecutor(max_workers=campaign['max_in_flight']) as pool:
            active = {}
            unresolved = 0
            while pending or active:
                while pending and len(active) + unresolved < campaign['max_in_flight'] and not STOP.is_set():
                    directory = pending.pop(0)
                    active[pool.submit(locked_job, directory)] = directory
                if not active:
                    if pending:
                        print('Unresolved jobs occupy the submission limit; resume this campaign after inspection', flush=True)
                        failures += 1
                    break
                future = next(as_completed(active))
                directory = active.pop(future)
                failures += future.result() != 0
                state = json.loads((directory / 'state.json').read_text())
                if state.get('submission_started') and not state.get('finished'):
                    unresolved += 1
        return 130 if STOP.is_set() else int(bool(failures))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target', help='Description file, or resume/status/collect/cleanup/archive')
    parser.add_argument('directory', nargs='?', type=Path)
    parser.add_argument('--name')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--discard-incomplete', action='store_true')
    parser.add_argument('--configure-profile', action='store_true', help='Prompt once if the description profile is missing')
    parser.add_argument('--profile-source', type=Path, help='Optional saved state file or run directory supplying profile suggestions')
    args = parser.parse_args()
    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, lambda *_: STOP.set())
    if args.target == 'archive':
        if args.name or args.dry_run or args.configure_profile or args.profile_source or args.discard_incomplete:
            parser.error('archive accepts only an optional saved campaign or job directory')
        from pack_results import archive_saved
        return archive_saved(workflow_root=ROOT, target=args.directory.resolve() if args.directory else None)
    if args.target != 'status' and not args.dry_run:
        for executable in ('gcloud', 'cdk'):
            if shutil.which(executable) is None:
                raise ValueError(f'Required CPU-VM command not found: {executable}')
    if args.target in {'resume', 'status', 'collect', 'cleanup'}:
        if args.directory is None:
            parser.error('Saved campaign or job directory is required')
        path = args.directory.resolve()
        if args.name or args.dry_run or args.configure_profile or args.profile_source:
            parser.error('Saved runs keep their original description and name')
        if args.target == 'status':
            jobs = [path / n for n in json.loads((path / 'campaign.json').read_text())['jobs']] if (path / 'campaign.json').exists() else [path]
            for directory in jobs:
                state = json.loads((directory / 'state.json').read_text())
                print(f'{state["run_id"]}: {state["phase"]}; job={state.get("job_id", "unassigned")}; {state["uri"]}; resources_cleaned={state.get("resources_cleaned", False)}')
            return 0
        if (path / 'campaign.json').exists():
            if args.target != 'resume':
                parser.error('collect/cleanup require one exact job directory')
            return run_campaign(path)
        return locked_job(directory=path, action='run' if args.target == 'resume' else args.target, discard=args.discard_incomplete)
    if args.directory or args.discard_incomplete:
        parser.error('Unexpected argument for a new run')
    if args.profile_source and not args.configure_profile:
        parser.error('--profile-source requires --configure-profile')
    if args.configure_profile:
        from configure_profile import configure
        configure(description=Path(args.target).resolve(), source=args.profile_source)
    path = prepare(description_path=Path(args.target).resolve(), name=args.name, dry_run=args.dry_run)
    return 0 if args.dry_run else run_campaign(path)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as error:
        print(f'Workflow stopped: {error}', file=sys.stderr)
        sys.exit(1)
