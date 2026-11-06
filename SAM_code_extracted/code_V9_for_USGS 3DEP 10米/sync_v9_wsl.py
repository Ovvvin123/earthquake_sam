"""Explicit V9-only copy/sync; refuses independently edited destination files."""
from pathlib import Path
import hashlib
import json
import shutil
import sys


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    if sys.platform!='linux':raise RuntimeError('Run using WSL Python')
    source=Path(__file__).resolve().parent
    dest=Path('/home/ovvvin/projects/earthquake-sam-v9')
    if source.name!='code_V9_for_USGS 3DEP 10米':raise ValueError('Wrong source')
    if dest.resolve()!=dest:raise ValueError('Destination is redirected')
    mf=dest/'v9_sync_manifest.json'
    previous=json.loads(mf.read_text()) if mf.exists() else None
    if dest.exists() and not previous:raise FileExistsError(dest)
    if previous and previous['source']!=str(source):raise ValueError('Source mismatch')
    files=list(source.glob('*.py'))+list(source.glob('*.txt'))
    files+=[source/'sam2_hiera_small.pt',source/'copy_baseline.json']
    if (source/'original_source_integrity.json').exists():files.append(source/'original_source_integrity.json')
    files+=list((source/'experiments').glob('*.json'))
    files+=list((source/'V5_different_split').glob('*.py'))
    planned=[]
    for src in files:
        rel=src.relative_to(source).as_posix();dst=dest/rel
        before=previous['files'].get(rel) if previous else None
        if dst.exists() and (not before or digest(dst)!=before['sha256']):
            raise RuntimeError(f'Independent WSL edits: {rel}')
        hashed=digest(src)
        planned.append((src,rel,hashed,not dst.exists() or before['sha256']!=hashed))
    dest.mkdir(parents=True,exist_ok=True)
    report=dict(source=str(source),destination=str(dest),files={},training_started=False)
    changed=[]
    for src,rel,hashed,copy in planned:
        dst=dest/rel
        if copy:
            dst.parent.mkdir(parents=True,exist_ok=True)
            tmp=dst.with_name(dst.name+'.sync-tmp')
            shutil.copy2(src,tmp)
            if digest(tmp)!=hashed or digest(src)!=hashed:raise IOError(rel)
            tmp.replace(dst);changed.append(rel)
        report['files'][rel]=dict(sha256=hashed,bytes=src.stat().st_size)
    tmp=mf.with_suffix('.json.tmp');tmp.write_text(json.dumps(report,indent=2));tmp.replace(mf)
    print(json.dumps(dict(files=len(files),copied=changed,destination=str(dest)),indent=2))


if __name__=='__main__':main()
