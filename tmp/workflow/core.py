"""Validated experiment descriptions and immutable file manifests."""
from __future__ import annotations

import fnmatch
import hashlib
import itertools
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import unicodedata
import uuid


def read_document(path: Path) -> dict:
    text = path.read_text()
    if path.suffix == '.json':
        def pairs(items: list) -> dict:
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError(f'Duplicate key: {key}')
                result[key] = value
            return result
        data = json.loads(text, object_pairs_hook=pairs)
    else:
        import yaml
        class UniqueLoader(yaml.SafeLoader):
            pass
        def mapping(loader, node, deep=False):
            result = {}
            for key_node, value_node in node.value:
                key = loader.construct_object(key_node, deep=deep)
                if not isinstance(key, str) or key in result:
                    raise ValueError('Document keys must be unique strings')
                result[key] = loader.construct_object(value_node, deep=deep)
            return result
        UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
        data = yaml.load(text, Loader=UniqueLoader)
    if not isinstance(data, dict):
        raise ValueError('Expected a mapping at document root')
    return data


def keys(data: dict, allowed: set[str], required: set[str] = frozenset()) -> None:
    if not isinstance(data, dict) or set(data) - allowed or required - set(data):
        raise ValueError(f'Expected keys {sorted(allowed)}, required {sorted(required)}')


def positive(value: object, maximum: int = 86400) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f'Expected integer in 1..{maximum}: {value}')
    return value


def absolute(value: str) -> str:
    if not isinstance(value, str) or not value.startswith('/') or '..' in PurePosixPath(value).parts:
        raise ValueError(f'Expected absolute container path: {value}')
    if value == '/' or value.startswith(('/proc/', '/sys/', '/dev/')):
        raise ValueError(f'Invalid data destination: {value}')
    return str(PurePosixPath(value))


def slug(name: str) -> str:
    if not isinstance(name, str) or not name.strip() or len(name) > 200:
        raise ValueError('Run name must contain 1..200 characters')
    text = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode().lower()
    text = re.sub('[^a-z0-9]+', '-', text).strip('-')[:28].strip('-') or 'run'
    if text.startswith('goog') or 'google' in text or 'g00gle' in text:
        text = 'run'
    return text


def identity(name: str) -> str:
    # A random suffix avoids both cross-project bucket collisions and repeated labels.
    return slug(name) + '-' + uuid.uuid4().hex[:24]


def checksum(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.new')
    with temporary.open('wb') as stream:
        stream.write(json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def safe_path(root: Path, relative: str) -> Path:
    p = PurePosixPath(relative)
    if not relative or p.is_absolute() or '..' in p.parts or '\\' in relative:
        raise ValueError(f'Invalid artifact path: {relative}')
    result = root / relative
    for parent in [result, *result.parents]:
        if parent == root.parent:
            break
        if parent.is_symlink():
            raise ValueError(f'Symlink in artifact path: {relative}')
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Artifact escapes root: {relative}')
    return result


def snapshot(source: Path, target: Path, include: list[str]) -> list[dict]:
    if not source.is_dir() or source.is_symlink():
        raise ValueError(f'Source must be a regular directory: {source}')
    entries = []
    if target.resolve().is_relative_to(source.resolve()):
        raise ValueError('Snapshot destination must be outside its source directory')
    target.mkdir(parents=True, exist_ok=True)
    for base, dirs, files in os.walk(source, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in {'.git', '.venv', '__pycache__'})
        for name in dirs + sorted(files):
            path = Path(base) / name
            if path.is_symlink():
                raise ValueError(f'Snapshot does not follow symlinks: {path}')
        for name in sorted(files):
            path = Path(base) / name
            relative = path.relative_to(source).as_posix()
            if not any(fnmatch.fnmatchcase(relative, pattern) for pattern in include):
                continue
            if not stat.S_ISREG(path.stat().st_mode):
                raise ValueError(f'Not a regular file: {path}')
            dest = safe_path(root=target, relative=relative)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src=path, dst=dest)
            mode = 0o755 if path.stat().st_mode & 0o111 else 0o644
            dest.chmod(mode)
            entries.append({'path': relative, 'bytes': dest.stat().st_size,
                            'sha256': checksum(dest), 'mode': mode})
    return entries


def verify(root: Path, entries: list[dict]) -> None:
    seen = set()
    for item in entries:
        if item['path'] in seen:
            raise ValueError('Duplicate manifest path')
        seen.add(item['path'])
        path = safe_path(root=root, relative=item['path'])
        if not path.is_file() or path.stat().st_size != item['bytes'] or checksum(path) != item['sha256']:
            raise ValueError(f'Artifact verification failed: {item["path"]}')


def environment(data: dict) -> dict[str, str]:
    if not isinstance(data, dict):
        raise ValueError('Expected environment mapping')
    result = {}
    for key, value in data.items():
        if not re.fullmatch('[A-Z_][A-Z0-9_]*', key) or key in {'RUN_ID', 'RUN_NAME', 'OUTPUT_DIR'} or key.startswith('INPUT_'):
            raise ValueError(f'Invalid or reserved environment key: {key}')
        if type(value) not in {str, int, float, bool}:
            raise ValueError('Environment values must be scalars')
        result[key] = str(value)
    return result


def uses_cdk_storage(profile: dict) -> bool:
    return profile['storage'].get('mode') == 'cdk'


CDK_MOUNT_ROOT = Path('/cdk-outputs')


def cdk_storage_directory(run_id: str) -> Path:
    if not re.fullmatch(r'[a-z0-9-]+-[a-f0-9]{24}', run_id):
        raise ValueError('Invalid run ID')
    return CDK_MOUNT_ROOT / 'outputs' / ('workflow-' + run_id)


def cdk_storage_uri(state: dict) -> str | None:
    if not state.get('job_id'):
        return None
    if not re.fullmatch('j-[A-Za-z0-9-]+', state['job_id']) or not re.fullmatch(r'[a-z0-9-]+-[a-f0-9]{24}', state['run_id']):
        raise ValueError('Invalid CDK storage identity')
    return state['profile']['storage']['outputs_root'] + '/' + state['job_id'] + '/outputs/workflow-' + state['run_id']


def load_description(path: Path) -> tuple[dict, dict]:
    data = read_document(path)
    keys(data, {'version', 'profile', 'name', 'description', 'code', 'inputs', 'run', 'outputs', 'execution', 'cases', 'image_build'},
         {'version', 'profile', 'code', 'run', 'outputs'})
    if data['version'] != 1:
        raise ValueError('Unsupported description version')
    profile_path = (path.parent / data['profile']).resolve()
    profile = read_document(profile_path)
    keys(profile, {'version', 'cloud', 'runtime', 'hardware', 'storage'}, {'version', 'cloud', 'runtime', 'hardware', 'storage'})
    if profile['version'] != 1:
        raise ValueError('Unsupported profile version')
    cloud = profile['cloud']
    required_cloud = {'project'} if uses_cdk_storage(profile) else {'project', 'region', 'workload_iam_member', 'service_account'}
    keys(cloud, {'project', 'region', 'workload_iam_member', 'service_account'}, required_cloud)
    for key in required_cloud - {'workload_iam_member'}:
        if not isinstance(cloud[key], str) or not re.fullmatch('[a-z][a-z0-9-]{1,62}', cloud[key]):
            raise ValueError(f'Set cloud.{key} explicitly in {profile_path}')
    if not uses_cdk_storage(profile):
        member = cloud['workload_iam_member']
        if not isinstance(member, str) or not (member.startswith('serviceAccount:') or member.startswith('principal://iam.googleapis.com/')) or any(c.isspace() for c in member):
            raise ValueError('Set the exact workload IAM member; broad or public principals are not accepted')
    keys(profile['runtime'], {'image', 'repository'})
    if 'image_build' in data:
        build = data['image_build']
        keys(build, {'cwd', 'argv', 'timeout_seconds'}, {'cwd', 'argv'})
        build['cwd'] = str((path.parent / build['cwd']).resolve())
        if not Path(build['cwd']).is_dir():
            raise ValueError('image_build.cwd must be an existing CPU-VM directory')
        if not isinstance(build['argv'], list) or not build['argv'] or not all(isinstance(a, str) and '\0' not in a for a in build['argv']):
            raise ValueError('image_build.argv must be a nonempty string list')
        build['timeout_seconds'] = positive(build.get('timeout_seconds', 28800), maximum=86400)
        from image_build import REPOSITORY
        if not REPOSITORY.fullmatch(profile['runtime'].get('repository', '')):
            raise ValueError('runtime.repository must be an Artifact Registry image path without a tag')
    elif not re.fullmatch(r'[a-z0-9._:/-]+@sha256:[a-f0-9]{64}', profile['runtime'].get('image', '')):
        raise ValueError('runtime.image must be an immutable registry digest')
    hardware = profile['hardware']
    keys(hardware, {'accelerator', 'topology', 'chips_per_host'}, {'accelerator', 'topology', 'chips_per_host'})
    if not re.fullmatch('[a-z0-9-]+', hardware['accelerator']) or not re.fullmatch(r'\d+x\d+x\d+', hardware['topology']):
        raise ValueError('Invalid hardware selectors')
    positive(hardware['chips_per_host'], maximum=64)
    storage = profile['storage']
    if uses_cdk_storage(profile):
        keys(storage, {'mode', 'outputs_root'}, {'mode', 'outputs_root'})
        if not isinstance(storage['outputs_root'], str) or not re.fullmatch(r'gs://[a-z0-9][a-z0-9.-]+/[a-zA-Z0-9/_-]+', storage['outputs_root']) or storage['outputs_root'].endswith('/'):
            raise ValueError('storage.outputs_root must be the configured CDK jobs prefix')
    else:
        keys(storage, {'dedicated_bucket', 'deletion', 'soft_delete_days'}, {'dedicated_bucket', 'deletion', 'soft_delete_days'})
        if storage != {'dedicated_bucket': True, 'deletion': 'manual', 'soft_delete_days': 0}:
            raise ValueError('This version requires dedicated, manually deleted buckets with soft delete disabled')
    code = data['code']
    keys(code, {'directory', 'delivery', 'destination', 'include'}, {'directory', 'delivery', 'destination'})
    if code['delivery'] != 'snapshot':
        raise ValueError('Only snapshot delivery is supported')
    code['destination'] = absolute(code['destination'])
    code['directory'] = str((path.parent / code['directory']).resolve())
    code.setdefault('include', ['**'])
    if not isinstance(code['include'], list) or not all(isinstance(p, str) for p in code['include']):
        raise ValueError('code.include must be a list of file patterns')
    inputs = data.setdefault('inputs', {})
    if not isinstance(inputs, dict):
        raise ValueError('inputs must be a mapping')
    destinations = [code['destination'], '/run-storage', '/run-work', '/run-bundle', '/cdk-outputs']
    for key, item in inputs.items():
        if not re.fullmatch('[a-z][a-z0-9_]*', key):
            raise ValueError('Input names must be lowercase identifiers')
        keys(item, {'source', 'destination'}, {'source', 'destination'})
        item['source'] = str((path.parent / item['source']).resolve())
        item['destination'] = absolute(item['destination'])
        destinations.append(item['destination'])
    for a, b in itertools.combinations(destinations, 2):
        if PurePosixPath(a).is_relative_to(b) or PurePosixPath(b).is_relative_to(a):
            raise ValueError(f'Overlapping container destinations: {a}, {b}')
    run = data['run']
    keys(run, {'cwd', 'argv', 'env', 'verify_imports'}, {'cwd', 'argv'})
    run['cwd'] = absolute(run['cwd'])
    if not isinstance(run['argv'], list) or not run['argv'] or not all(isinstance(a, str) and '\0' not in a for a in run['argv']):
        raise ValueError('run.argv must be a nonempty string list')
    run['env'] = environment(run.get('env', {}))
    imports = run.setdefault('verify_imports', {})
    if not isinstance(imports, dict):
        raise ValueError('verify_imports must map module names to source paths')
    for key, value in imports.items():
        if not re.fullmatch(r'[a-zA-Z_][a-zA-Z0-9_.]*', key):
            raise ValueError('Invalid module name')
        absolute(value)
    keys(data['outputs'], {'directory', 'snapshot_seconds', 'extra'}, {'directory'})
    data['outputs']['directory'] = str((path.parent / data['outputs']['directory']).resolve())
    data['outputs']['snapshot_seconds'] = positive(data['outputs'].get('snapshot_seconds', 30), maximum=3600)
    extra = data['outputs'].setdefault('extra', {})
    if not isinstance(extra, dict):
        raise ValueError('outputs.extra must be a mapping')
    for key, value in extra.items():
        if not re.fullmatch('[a-z][a-z0-9_]*', key) or key == 'output':
            raise ValueError('Invalid or reserved extra-output name')
        absolute(value)
    execution = data.setdefault('execution', {})
    keys(execution, {'timeout_seconds', 'max_in_flight', 'poll_seconds', 'archive_seconds', 'wait_seconds', 'cleanup'})
    cleanup = execution.setdefault('cleanup', 'manual')
    if cleanup not in {'manual', 'after_collection'}:
        raise ValueError('execution.cleanup must be manual or after_collection')
    if cleanup == 'after_collection' and 'image_build' not in data:
        raise ValueError('Automatic resource cleanup requires a run-owned image build')
    for key, default, maximum in [('timeout_seconds', 43200, 604800), ('max_in_flight', 50, 50),
                                  ('poll_seconds', 30, 600), ('archive_seconds', 1200, 86400), ('wait_seconds', 86400, 604800)]:
        execution[key] = positive(execution.get(key, default), maximum=maximum)
    cases = data.setdefault('cases', [{'name': 'default', 'env': {}}])
    if not isinstance(cases, list) or not cases or len(cases) > 10000:
        raise ValueError('cases must contain 1..10000 cases')
    seen = set()
    for case in cases:
        keys(case, {'name', 'env'}, {'name'})
        case['name'] = slug(case['name'])
        if case['name'] in seen:
            raise ValueError('Case names collide after normalization')
        seen.add(case['name'])
        case['env'] = environment(case.get('env', {}))
    return data, profile
