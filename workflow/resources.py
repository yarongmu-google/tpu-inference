"""Delete only the image and bucket recorded as belonging to one completed run."""
from __future__ import annotations

import fcntl
import json
from pathlib import Path
import re
import sys
import time
from typing import TYPE_CHECKING
from urllib.parse import unquote

from core import read_document, save, verify

if TYPE_CHECKING:
    from controller import Job


def verify_collected(job: Job) -> None:
    state = job.state
    if not state.get('artifacts_verified'):
        raise ValueError('Results are not verified; collect before cleanup')
    manifest = read_document(job.directory / 'collected/manifest.json')
    if (manifest.get('format') != 'run-artifacts-v1' or manifest.get('run_id') != state['run_id']
            or manifest.get('image') != state['image'] or manifest.get('config_sha256') != state['config_sha256']
            or manifest.get('state') not in {'succeeded', 'failed'} or type(manifest.get('exit_code')) is not int):
        raise ValueError('Collected manifest does not describe this completed run')
    verify(root=job.directory / 'collected/files', entries=manifest['files'])


def owned(job: Job) -> dict:
    state = job.state
    run_id = state['run_id']
    if not re.fullmatch(r'[a-z0-9-]+-[a-f0-9]{24}', run_id):
        raise ValueError('Invalid run identity')
    image = state['profile']['runtime']['repository'].rstrip('/') + '/' + run_id
    value = {'run_id': run_id, 'bucket': run_id, 'image': image,
             'local_tags': ['runtime:' + run_id, image + ':run']}
    if (state.get('owned_image') != image or state['bucket'] != run_id or state['uri'] != 'gs://' + run_id
            or read_document(job.directory / 'resources.json') != value):
        raise ValueError('Resource ownership record differs; cleanup is blocked')
    if state.get('image') and not state['image'].startswith(image + '@sha256:'):
        raise ValueError('Image digest belongs to another run')
    return value


def image_package(job: Job, image: str) -> str | None:
    host, project, repository, package = image.split('/', 3)
    location = host.removesuffix('-docker.pkg.dev')
    _, text = job.command(args=['gcloud', '--project', project, 'artifacts', 'packages', 'list',
        '--location', location, '--repository', repository, '--format=json'])
    rows = json.loads(text)
    if not isinstance(rows, list) or any(not isinstance(row.get('name'), str) or '/packages/' not in row['name'] for row in rows):
        raise ValueError('Unexpected registry package listing')
    return next((row['name'] for row in rows if unquote(row['name'].split('/packages/', 1)[1]) == package), None)


def bucket_present(job: Job) -> bool:
    _, text = job.gcloud(args=['storage', 'buckets', 'list', '--raw',
        '--filter=name=' + job.state['bucket'], '--format=json'])
    rows = json.loads(text)
    if not isinstance(rows, list) or any(row.get('name') != job.state['bucket'] for row in rows) or len(rows) > 1:
        raise ValueError('Unexpected bucket listing')
    return bool(rows)


def cleanup(job: Job, discard: bool, automatic: bool) -> int:
    if job.state.get('resources_cleaned'):
        return 0
    resources = owned(job=job)
    if automatic and (discard or job.state['execution'].get('cleanup') != 'after_collection'):
        raise ValueError('Automatic cleanup was not selected by this run description')
    image_dir = job.directory / 'image'
    image_dir.mkdir(exist_ok=True)
    with (image_dir / '.lock').open('a') as lock:
        # The image subprocess inherits this lock; an interrupted controller cannot
        # clean a run whose builder is still alive.
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if job.state.get('submission_started'):
            remote = job.discover()
            if (remote is None or remote.get('job_status') not in {'Succeeded', 'Failed', 'Deleted'}
                    or remote.get('state') not in {'Complete', 'LogArchiveFailed', 'TraceGenFailed', 'FinalizingFailed', 'K8sJobDeleted'}):
                raise ValueError('Job is active, archiving, or unresolved; resources retained')
            if not discard:
                verify_collected(job=job)
                code = 0 if remote.get('job_status') == 'Succeeded' and remote.get('state') == 'Complete' and job.state.get('artifact_exit_code') == 0 else 1
                job.save(finished=True, exit_code=code)
        elif not discard and not (job.directory / 'error.txt').is_file():
            raise ValueError('Unsubmitted run has no saved failure; use explicit cleanup with --discard-incomplete')
        if not automatic:
            print(f'Delete image {resources["image"]} and bucket {job.state["uri"]}; keep local results', flush=True)
            if not sys.stdin.isatty() or input('Type DELETE ' + job.state['run_id'] + ' to confirm: ') != 'DELETE ' + job.state['run_id']:
                raise ValueError('Deletion was not confirmed')
        job.phase('CLEANING_RESOURCES')
        job.save(resource_cleanup_started=True)
        # Remove the bucket first so collection failures cannot strand an image-less run.
        if not job.state.get('deleted'):
            if job.state.get('bucket_creation_started'):
                present = bucket_present(job=job)
                if present:
                    metadata = job.bucket_identity()
                    previous = job.state.get('cleanup_bucket_created_at')
                    if previous:
                        if metadata.get('timeCreated') != previous:
                            raise ValueError('Bucket was replaced after cleanup began')
                    else:
                        if not job.state.get('bucket_marked') or metadata.get('labels', {}).get('run_id') != job.state['run_id']:
                            raise ValueError('Bucket ownership is unconfirmed')
                        _, text = job.gcloud(args=['storage', 'cat', job.state['uri'] + '/owner.json'])
                        if json.loads(text) != read_document(job.directory / 'owner.json') or not metadata.get('timeCreated'):
                            raise ValueError('Bucket ownership marker or creation time is missing')
                        job.save(cleanup_bucket_created_at=metadata['timeCreated'])
                    job.save(deletion_started=True)
                    job.gcloud(args=['storage', 'rm', '--recursive', '--quiet', job.state['uri']], timeout=1800, transfer=True)
                    if bucket_present(job=job):
                        raise ValueError('Bucket deletion has not completed')
            job.save(deleted=True, bucket_deleted_at=time.time())
        if not job.state.get('image_deleted'):
            package_name = image_package(job=job, image=resources['image'])
            if package_name:
                job.save(image_deletion_started=True)
                project = resources['image'].split('/')[1]
                job.command(args=['gcloud', '--project', project, 'artifacts', 'packages', 'delete',
                    package_name, '--quiet'], timeout=1800)
                package_name = image_package(job=job, image=resources['image'])
            if package_name:
                    raise ValueError('Image deletion has not completed')
            job.save(image_deleted=True, image_deleted_at=time.time())
        if not job.state.get('local_image_tags_removed'):
            # Remove exact run tags only. Never force-remove an image ID or prune shared layers.
            _, text = job.command(args=['docker', 'image', 'ls', '--format', '{{.Repository}}:{{.Tag}}'])
            tags = set(text.splitlines())
            for tag in resources['local_tags']:
                if tag in tags:
                    job.command(args=['docker', 'image', 'rm', tag])
            job.save(local_image_tags_removed=True)
        cleaned_at = time.time()
        save(path=job.directory / 'cleanup.json', value={**resources, 'image_digest': job.state.get('image'),
            'cleaned_at': cleaned_at, 'exit_code': job.state.get('exit_code'),
            'artifact_exit_code': job.state.get('artifact_exit_code'), 'local_results': str(job.directory / 'collected')})
        job.save(resources_cleaned=True, cleaned_at=cleaned_at, phase='CLEANED')
        print(f'[{job.state["run_id"]}] image and bucket cleaned; local record: {job.directory}', flush=True)
        return 0
