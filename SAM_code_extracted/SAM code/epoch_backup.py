"""Opt-in synchronous backups of closed epoch outputs, never live writer files.

Published epoch directories are immutable. Failed copies remain .partial-* and
are not advertised as recoverable snapshots. Backup failure does not kill a run.
"""
import hashlib
import json
import os
from pathlib import Path
import uuid
import warnings


def due(epoch, config):
    if not config.get('backupDir'):
        return False
    milestones = {int(x) for x in config.get('backupMilestones', '').split(',') if x.strip()}
    every = int(config.get('backupEvery', 20))
    return epoch == int(config['epochs']) or epoch in milestones or (every > 0 and epoch % every == 0)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def publish_epoch(run_dir, epoch, config):
    run = Path(run_dir).resolve()
    root = Path(config['backupDir']).expanduser().resolve()
    if root == run or run in root.parents or root in run.parents:
        raise ValueError('Backup root must be separate from the live run')
    checkpoint = run / ('ema_checkpoint.pth' if config.get('emaTeacher') else 'training_checkpoint.pth')
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    root.mkdir(parents=True, exist_ok=True)
    final = root / f'epoch_{epoch:04d}'
    if final.exists():
        raise FileExistsError(f'Refusing to overwrite backup: {final}')
    staging = root / f'.partial-{epoch:04d}-{uuid.uuid4().hex}'
    staging.mkdir()
    # console.log has a live stdout writer; omit it. Metadata writers have closed
    # before this hook. Large preview trees and duplicate model.pth are omitted.
    files = [checkpoint]
    files += sorted(p for p in run.iterdir() if p.is_file() and p.suffix in ('.json', '.jsonl', '.csv', '.txt'))
    if epoch == int(config['epochs']):
        for name in ('best_student', 'best_teacher'):
            directory = run / name
            if directory.is_dir():
                files += sorted(p for p in directory.iterdir() if p.name in ('model.pth', 'metrics.json'))
    manifest = dict(format='epoch_backup_v1', epoch=epoch, source=str(run), files={})
    for source in files:
        relative = source.relative_to(run)
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        before = source.stat()
        digest = hashlib.sha256()
        with source.open('rb') as src, target.open('xb') as dst:
            for block in iter(lambda: src.read(4 * 1024 * 1024), b''):
                dst.write(block)
                digest.update(block)
            dst.flush()
            os.fsync(dst.fileno())
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f'Source changed during backup: {source}')
        if sha256(target) != digest.hexdigest():
            raise IOError(f'Backup verification failed: {target}')
        manifest['files'][relative.as_posix()] = dict(bytes=after.st_size, sha256=digest.hexdigest())
    manifest['status'] = 'complete'
    with (staging / 'backup_manifest.json').open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    staging.rename(final)
    return final


def backup_epoch(run_dir, epoch, config):
    if not due(epoch, config):
        return None
    try:
        result = publish_epoch(run_dir, epoch, config)
        event = dict(epoch=epoch, status='complete', path=str(result))
        print(f'Backup verified: {result}', flush=True)
    except Exception as error:
        result = None
        event = dict(epoch=epoch, status='failed', error=str(error))
        warnings.warn(f'Backup failed; live WSL checkpoint retained: {error}')
    with (Path(run_dir) / 'backup_events.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(event) + '\n')
    return result
