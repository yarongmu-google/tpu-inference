"""Create a local environment profile from reviewed settings and saved execution metadata."""
from __future__ import annotations

import fcntl
import json
from pathlib import Path
import re
import sys
from collections.abc import Callable

from core import read_document, save
from image_build import REPOSITORY

IMAGE = re.compile(r'[a-z0-9._:/-]+@sha256:[a-f0-9]{64}')
IDENTIFIER = re.compile(r'[a-z][a-z0-9-]{1,62}')
MEMBER = re.compile(r'(serviceAccount:[^\s]+|principal://iam\.googleapis\.com/[^\s]+)')


def source_defaults(source: Path | None) -> dict[str, str]:
    if source is None or not source.exists():
        return {}
    candidates = [source] if source.is_file() else sorted(source.glob('run-*/state.json'), reverse=True)
    for path in candidates:
        try:
            state = json.loads(path.read_text())
            image = state.get('image')
            if not isinstance(image, str) or not IMAGE.fullmatch(image):
                continue
            defaults = {'image': image}
            cloud = state.get('profile', {}).get('cloud', {})
            for key in ('project', 'region', 'service_account', 'workload_iam_member'):
                if isinstance(cloud.get(key), str):
                    defaults[key] = cloud[key]
            # Registry location/project are suggestions, never silently chosen for storage.
            registry = re.fullmatch(r'([a-z0-9-]+)-docker\.pkg\.dev/([^/]+)/.+', image)
            if registry:
                defaults.setdefault('region', registry[1])
                defaults.setdefault('project', registry[2])
            for name in ('rendered.yml', 'submitted-recipe.yml'):
                rendered = path.parent / name
                if not rendered.is_file():
                    continue
                data = read_document(rendered)
                jobs = data.get('spec', {}).get('replicatedJobs', [])
                accounts = {job['template']['spec']['template']['spec'].get('serviceAccountName') for job in jobs}
                accounts.discard(None)
                if len(accounts) == 1:
                    defaults.setdefault('service_account', accounts.pop())
            print(f'Profile suggestions from saved execution: {path}', flush=True)
            return defaults
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return {}


def ask(label: str, default: str, valid: Callable[[str], object]) -> str:
    while True:
        suffix = f' [{default}]' if default else ''
        answer = input(label + suffix + ': ').strip() or default
        if valid(answer):
            return answer
        print('Invalid value; enter the explicit setting shown by your cloud/cluster configuration.', flush=True)


def configure(description: Path, source: Path | None) -> Path:
    data = read_document(description)
    profile = (description.parent / data['profile']).resolve()
    building = 'image_build' in data
    if profile.exists() and (not building or read_document(profile).get('runtime', {}).get('repository')):
        print(f'Using saved environment profile: {profile}', flush=True)
        return profile
    if not sys.stdin.isatty():
        raise ValueError(f'Environment profile is missing or needs an image repository: {profile}; run once interactively or provide the profile file')
    profile.parent.mkdir(parents=True, exist_ok=True)
    with profile.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if profile.exists():
            existing = read_document(profile)
            if not building or existing.get('runtime', {}).get('repository'):
                return profile
            repository = ask(label='Runtime image repository (no tag or digest)',
                default=existing.get('runtime', {}).get('image', '').split('@')[0], valid=REPOSITORY.fullmatch)
            if input('Save image repository in this local profile? [y/N]: ').strip().lower() not in {'y', 'yes'}:
                raise ValueError('Profile was not saved')
            existing['runtime']['repository'] = repository
            save(path=profile, value=existing)
            profile.chmod(0o600)
            return profile
        defaults = source_defaults(source)
        print('One-time environment setup. Confirm defaults with Enter; these settings are saved locally.', flush=True)
        print('Bucket project/region suggestions may come from the image registry; choose the intended storage project and region.', flush=True)
        project = ask(label='Bucket project ID', default=defaults.get('project', ''), valid=IDENTIFIER.fullmatch)
        region = ask(label='Bucket region', default=defaults.get('region', ''), valid=IDENTIFIER.fullmatch)
        if building:
            runtime = {'repository': ask(label='Runtime image repository (no tag or digest)',
                default=defaults.get('image', '').split('@')[0], valid=REPOSITORY.fullmatch)}
        else:
            runtime = {'image': ask(label='Published runtime image digest', default=defaults.get('image', ''), valid=IMAGE.fullmatch)}
        account = ask(label='CDK-assigned Kubernetes service account', default=defaults.get('service_account', ''), valid=IDENTIFIER.fullmatch)
        print('Bucket access member: the exact serviceAccount:... or principal://iam.googleapis.com/... identity for that workload.', flush=True)
        print('This identity cannot be inferred safely from the Kubernetes account name alone.', flush=True)
        member = ask(label='Bucket access IAM member', default=defaults.get('workload_iam_member', ''), valid=MEMBER.fullmatch)
        accelerator = ask(label='TPU accelerator', default='tpu7x', valid=lambda v: re.fullmatch('[a-z0-9-]+', v))
        topology = ask(label='Single-host topology', default='2x2x1', valid=lambda v: re.fullmatch(r'\d+x\d+x\d+', v))
        chips = ask(label='Physical chips per host', default='4', valid=lambda v: v.isdigit() and 1 <= int(v) <= 64)
        value = {'version': 1, 'cloud': {'project': project, 'region': region, 'service_account': account,
                 'workload_iam_member': member}, 'runtime': runtime,
                 'hardware': {'accelerator': accelerator, 'topology': topology, 'chips_per_host': int(chips)},
                 'storage': {'dedicated_bucket': True, 'deletion': 'manual', 'soft_delete_days': 0}}
        print(f'Profile: {profile}\nStorage: project={project}, region={region}, dedicated bucket, manual cleanup, soft delete disabled', flush=True)
        if input('Save this local profile? [y/N]: ').strip().lower() not in {'y', 'yes'}:
            raise ValueError('Profile was not saved')
        save(path=profile, value=value)
        profile.chmod(0o600)
        return profile
