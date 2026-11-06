"""Candidate AutoDL environment: preserve GPU packages, record/check dependency closure.

This lock records an installation, not a claim of full-model validation.
The ordinary cloud preflight still verifies SAM2, assets and real inference.
"""
import argparse
import importlib.metadata as metadata
import json
import platform
from pathlib import Path
import sys

PROFILE = 'autodl5090'
LOCK_NAME = 'earthquake-v9-autodl5090.json'
ROOTS = ('torch', 'torchvision', 'sam-2', 'numpy', 'scipy', 'h5py',
         'matplotlib', 'pyshp', 'pillow', 'timm', 'numba', 'hydra-core',
         'iopath', 'tqdm', 'packaging')


def core_check():
    if sys.platform != 'linux' or platform.machine() != 'x86_64' or sys.version_info[:2] != (3, 12):
        raise RuntimeError('AutoDL profile requires Linux x86_64 / Python 3.12')
    import torch
    import torchvision
    if str(torch.__version__) != '2.8.0+cu128' or torch.version.cuda != '12.8':
        raise RuntimeError('Expected image torch 2.8.0+cu128; do not install the cu126 lock')
    if str(torchvision.__version__) not in ('0.23.0', '0.23.0+cu128'):
        raise RuntimeError('Expected matching torchvision 0.23.0 (CUDA 12.8)')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    if torch.cuda.get_device_capability(0) != (12, 0):
        raise RuntimeError('This profile targets RTX 5090 / sm_120')
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError('Native BF16 unavailable')
    return {'python': platform.python_version(), 'torch': str(torch.__version__),
            'torchvision': str(torchvision.__version__), 'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0)}


def gpu_constraints():
    from packaging.utils import canonicalize_name
    names = {canonicalize_name(d.metadata['Name']) for d in metadata.distributions()
             if d.metadata.get('Name')}
    selected = sorted(n for n in names if n in ('torch', 'torchvision', 'torchaudio', 'triton')
                      or n.startswith('nvidia-'))
    return ''.join(f'{n}=={metadata.version(n)}\n' for n in selected)


def resolved_versions():
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    pending = list(ROOTS)
    versions = {}
    while pending:
        name = canonicalize_name(pending.pop())
        if name in versions:
            continue
        dist = metadata.distribution(name)
        versions[name] = dist.version
        for raw in dist.requires or ():
            requirement = Requirement(raw)
            if requirement.marker and not requirement.marker.evaluate({'extra': ''}):
                continue
            actual = metadata.version(requirement.name)
            if requirement.specifier and actual not in requirement.specifier:
                raise RuntimeError(f'Unsatisfied dependency of {name}: {raw}; installed {actual}')
            pending.append(requirement.name)
    return dict(sorted(versions.items()))


def check_lock(lock):
    if lock.get('profile') != PROFILE or lock.get('python') != platform.python_version():
        raise RuntimeError('AutoDL lock profile/Python mismatch; run setup in a NEW venv')
    current = resolved_versions()
    expected = lock.get('versions', {})
    changes = sorted(k for k in current.keys() | expected.keys()
                     if current.get(k) != expected.get(k))
    if changes:
        raise RuntimeError('AutoDL environment changed: ' + ', '.join(changes))
    return current


def environment_versions():
    core_check()
    path = Path(sys.prefix) / LOCK_NAME
    if not path.is_file():
        raise RuntimeError('Missing AutoDL environment lock; run cloud/autodl5090.sh setup first')
    return check_lock(json.loads(path.read_text(encoding='utf-8')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('base', 'constraints', 'record'))
    args = parser.parse_args()
    if args.mode == 'constraints':
        print(gpu_constraints(), end='')
        return
    core = core_check()
    if args.mode == 'record':
        if sys.prefix == sys.base_prefix:
            raise RuntimeError('Refusing to record in the base image; use a new venv')
        path = Path(sys.prefix) / LOCK_NAME
        lock = dict(profile=PROFILE, python=platform.python_version(), core=core,
                    versions=resolved_versions(), full_model_validated=False)
        with path.open('x', encoding='utf-8') as stream:
            json.dump(lock, stream, indent=2)
        print('Recorded installation (full-model preflight still required):', path)
    print(json.dumps(core, indent=2))


if __name__ == '__main__':
    main()
