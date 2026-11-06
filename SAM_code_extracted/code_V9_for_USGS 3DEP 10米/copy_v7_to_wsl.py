"""Run inside WSL. Copy a new, verified runtime bundle; never move originals."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import uuid
from training_sources import training_source_snapshot


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=Path('/home/ovvvin/projects/earthquake-sam'))
    parser.add_argument('--copy', action='store_true', help='Without this flag only show the inventory')
    parser.add_argument('--sync-code', action='store_true', help='Update only verified runtime code in an existing copied bundle')
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('Run this command with the WSL Python environment')
    source = Path(__file__).resolve().parent
    destination = args.destination.expanduser().resolve()
    if not destination.is_relative_to(Path.home().resolve()) or destination.is_relative_to(Path('/mnt')):
        parser.error('Destination must be inside the Linux home directory')
    if args.copy and args.sync_code:
        parser.error('Use only one of --copy and --sync-code')
    if destination.exists() and not args.sync_code:
        raise FileExistsError(f'Refusing to overwrite: {destination}')
    files = set(training_source_snapshot(source))
    files.update(['run_experiment.py', 'training_sources.py', 'copy_v7_to_wsl.py', 'sam2_hiera_small.pt',
                  'check_v7_wsl.py', 'WSL_V7_README.txt'])
    files.update(p.relative_to(source).as_posix() for p in (source/'experiments').glob('*.json'))
    files.update(['test_wsl_v7.py','test_experiment_launcher.py','test_aio2_ema.py',
                  'test_sam2_lora.py','test_training_sources.py','test_aio2_preview.py',
                  'test_training_numerics.py','test_unimatch_fault.py','test_aio2_online.py',
                  'V5_different_split/sam2_lora.py'])
    for name in ('faults.h5','key.lst','key_active.lst','key_train.lst','key_val.lst','key_test.lst',
                 'manifest.json','dataset_usage.json','README.txt','tile_footprints_epsg3310.geojson',
                 'qa/validation_report.json','qa/reader_validation.json','qa/geometry_audit.json'):
        files.add('Earth_dataset_v7/'+name)
    total = sum((source/name).stat().st_size for name in files)
    print(json.dumps(dict(files=len(files), bytes=total, source=str(source), destination=str(destination)), indent=2), flush=True)
    if args.sync_code:
        manifest_path = destination/'migration_verification.json'
        report = json.loads(manifest_path.read_text())
        if report['source'] != str(source) or report['destination'] != str(destination):
            raise ValueError('Bundle provenance mismatch')
        changes = []
        for name in sorted(files):
            if name.startswith('Earth_dataset_v7/') or name == 'sam2_hiera_small.pt':
                continue  # Never alter migrated dataset or base weights during code sync.
            src, dst = source/name, destination/name
            expected = report['files'].get(name)
            if dst.exists() and (expected is None or digest(dst) != expected['sha256']):
                raise RuntimeError(f'WSL copy has independent edits; refusing overwrite: {name}')
            source_hash = digest(src)
            if not expected or expected['sha256'] != source_hash or not dst.exists():
                changes.append((name,source_hash))
        for name, source_hash in changes:
            src, dst = source/name, destination/name
            dst.parent.mkdir(parents=True,exist_ok=True)
            temporary = dst.with_name(dst.name+'.sync-'+uuid.uuid4().hex)
            shutil.copy2(src,temporary)
            if digest(temporary) != source_hash or digest(src) != source_hash:
                raise IOError(f'Code synchronization hash mismatch: {name}')
            temporary.replace(dst)
            report['files'][name] = dict(bytes=dst.stat().st_size,sha256=source_hash)
        report.update(file_count=len(report['files']),total_bytes=sum(f['bytes'] for f in report['files'].values()),
                      last_code_sync=[name for name,_ in changes])
        temporary = manifest_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(report,indent=2),encoding='utf-8')
        temporary.replace(manifest_path)
        print('Verified code sync:', report['last_code_sync'],flush=True)
        return
    if not args.copy:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Resolve the parent's mount as well, so symlinks cannot send this to /mnt/f.
    mount = os.statvfs(destination.parent)
    if mount.f_bavail * mount.f_frsize < total + 2*1024**3:
        raise OSError('Insufficient destination space for the runtime bundle plus margin')
    stage = destination.parent / ('.earthquake-sam-copy-'+uuid.uuid4().hex)
    stage.mkdir()
    report = dict(source=str(source), destination=str(destination), files={}, training_started=False)
    for name in sorted(files):
        src = (source/name).resolve()
        if not src.is_relative_to(source):
            raise ValueError(f'Source escaped project: {name}')
        before = src.stat()
        dst = stage/name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        source_hash, target_hash = digest(src), digest(dst)
        after = src.stat()
        if source_hash != target_hash or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise IOError(f'Copy verification failed: {name}')
        report['files'][name] = dict(bytes=after.st_size, sha256=target_hash)
        if after.st_size > 100*1024**2:
            print('Verified', name, after.st_size, flush=True)
    report.update(status='PASS', total_bytes=total, file_count=len(files))
    (stage/'migration_verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    stage.rename(destination)
    print('Published verified bundle:', destination, flush=True)


if __name__ == '__main__':
    main()
