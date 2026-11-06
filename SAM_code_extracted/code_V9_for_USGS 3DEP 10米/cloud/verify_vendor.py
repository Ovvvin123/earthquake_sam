"""Verify the bundled SAM2 archive before pip executes its build configuration."""
import hashlib
import json
from pathlib import Path


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def verify():
    root=Path(__file__).resolve().parent
    meta=json.loads((root/'vendor/manifest.json').read_text())
    path=root/'vendor'/meta['archive']
    if sha256(path)!=meta['sha256']:raise RuntimeError('SAM2 archive checksum mismatch')
    return meta


if __name__=='__main__':
    meta=verify()
    print('SAM2 archive verified:',meta['source_commit'])
