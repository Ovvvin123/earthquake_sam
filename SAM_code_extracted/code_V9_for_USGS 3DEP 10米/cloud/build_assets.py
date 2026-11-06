"""Maintainer utility: snapshot the verified local runtime; not run by setup.sh."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import tarfile
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from verify_vendor import sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sam2-source',type=Path,required=True)
    parser.add_argument('--data-dir',type=Path,required=True)
    args=parser.parse_args()
    here=Path(__file__).resolve().parent
    vendor=here/'vendor';vendor.mkdir(exist_ok=True)
    import sam2
    source=args.sam2_source.resolve();installed=Path(sam2.__file__).parent
    runtime={}
    for path in (source/'sam2').rglob('*'):
        if path.suffix not in ('.py','.yaml','.yml') or '__pycache__' in path.parts:continue
        name=path.relative_to(source/'sam2').as_posix()
        digest=sha256(path)
        if not (installed/name).is_file() or sha256(installed/name)!=digest:
            raise RuntimeError(f'SAM2 installed/source mismatch: {name}')
        runtime[name]=digest
    archive=vendor/'sam2-runtime.tar.gz'
    if archive.exists():raise FileExistsError(archive)
    included=[]
    with tarfile.open(archive,'w:gz') as tar:
        for path in sorted(source.rglob('*')):
            rel=path.relative_to(source)
            if not path.is_file() or any(p in ('build','__pycache__','.git') or p.endswith('.egg-info') for p in rel.parts):continue
            if path.suffix in ('.pyc','.so'):continue
            if path.is_symlink():raise ValueError(f'Symlink in source archive: {rel}')
            tar.add(path,arcname='sam2-runtime/'+rel.as_posix(),recursive=False)
            included.append(rel.as_posix())
    (vendor/'manifest.json').write_text(json.dumps(dict(archive=archive.name,sha256=sha256(archive),
        source_commit=(source/'SOURCE_COMMIT.txt').read_text().strip(),runtime_files=runtime,
        included_files=included,installed_source_match=True,cuda_extension_built=False),indent=2))
    pending=['torch','torchvision','numpy','scipy','h5py','timm','numba','matplotlib','pyshp','Pillow','SAM-2']
    versions={}
    while pending:
        name=canonicalize_name(pending.pop())
        if name in versions:continue
        dist=metadata.distribution(name);versions[name]=dist.version
        for requirement in dist.requires or []:
            req=Requirement(requirement)
            if req.marker is None or req.marker.evaluate({'extra':''}):pending.append(req.name)
    lines=['# Locked from the verified Python 3.10 / Linux x86_64 environment.',
           '# Install torch/torchvision from the cu126 index first (setup.sh).',
           '# SAM-2 is installed from vendor/sam2-runtime.tar.gz, not PyPI.']
    lines += [f'{n}=={v}' for n,v in sorted(versions.items()) if n!='sam-2']
    (here/'requirements.txt').write_text('\n'.join(lines)+'\n')
    (here/'environment_baseline.json').write_text(json.dumps(dict(python='3.10',platform='Linux x86_64',versions=versions),indent=2))
    assets={'data':{},'weights':{}}
    for name in ('faults.h5','key_train.lst','key_val.lst','key_test.lst','manifest.json'):
        path=args.data_dir/name
        assets['data'][name]=dict(bytes=path.stat().st_size,sha256=sha256(path))
    path=here.parent/'sam2_hiera_small.pt'
    assets['weights'][path.name]=dict(bytes=path.stat().st_size,sha256=sha256(path))
    assets.update(dataset='Earth_dataset_v7',resolution_m=30,tile_size=1024)
    (here/'assets_manifest.json').write_text(json.dumps(assets,indent=2))
    print(json.dumps(dict(sam2_runtime_files=len(runtime),locked_packages=len(versions),archive_bytes=archive.stat().st_size)))


if __name__=='__main__':main()
