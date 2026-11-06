"""Inspect ETA or watch one existing run, preserve provenance, optionally power off.

Never starts/restarts training. Shutdown requires the launcher's verified complete
status, final history and checkpoint files. Failure leaves the instance running.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parent.parent


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def history_rows(path):
    if not path.exists():
        return []
    lines = path.read_text(encoding='utf-8').splitlines()
    rows = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines)-1:
                raise
            # A writer may currently be appending the final line.
    return rows


def read_state(run):
    return json.loads((run/'launch.json').read_text(encoding='utf-8')), history_rows(run/'adaptation_history.jsonl')


def progress(launch, rows):
    cfg = launch['resolved_config']
    end, transition = int(cfg['epochs']), int(cfg.get('emaStartEpoch', 250))
    epoch = int(rows[-1]['epoch']) if rows else 0
    def durations(late):
        return [float(r['epoch_seconds']) for r in rows
                if (int(r['epoch']) >= transition) == late and int(r['epoch']) > 2
                and math.isfinite(float(r.get('epoch_seconds', 0))) and float(r.get('epoch_seconds', 0)) > 0][-5:]
    early, late = durations(False), durations(True)
    estimate = None
    basis = 'Need at least 3 completed epochs; preview/backup overhead is not fully included.'
    if epoch >= end:
        estimate, basis = 0., 'Epoch loop ended; launcher may still be validating final results.'
    elif len(late) >= 3:
        estimate = (end-epoch)*statistics.median(late)
        basis = 'Recent post-EMA epoch timing; excludes some preview/backup overhead.'
    elif early:
        early_seconds = statistics.median(early)
        before = max(0, min(end, transition-1)-epoch)
        after = max(0, end-max(epoch, transition-1))
        # Measured batch4 probes were ~6.08s before EMA and ~11.29s after.
        # These are synthetic short probes, not a full-epoch benchmark.
        estimate = before*early_seconds + after*early_seconds*(11.29/6.08)
        basis = 'Rough: measured early epochs + 1.86x late-phase factor from short probes; not a guaranteed finish time.'
    return dict(launcher_status=launch['status'], completed_epochs=epoch, target_epochs=end,
                recent_pre_ema_seconds=round(statistics.median(early), 2) if early else None,
                recent_post_ema_seconds=round(statistics.median(late), 2) if late else None,
                estimated_remaining_hours=round(estimate/3600, 2) if estimate is not None else None,
                estimate_basis=basis)


def completion_ready(run, launch, rows):
    if launch.get('status') != 'complete':
        return False
    cfg = launch['resolved_config']
    if not rows or int(rows[-1]['epoch']) != int(cfg['epochs']):
        raise RuntimeError('Complete status conflicts with final history; shutdown refused')
    checkpoint = 'ema_checkpoint.pth' if cfg.get('emaTeacher') else 'training_checkpoint.pth'
    for name in (checkpoint, 'model.pth', 'resolved_config.json', 'console.log'):
        if not (run/name).is_file() or (run/name).stat().st_size == 0:
            raise RuntimeError(f'Missing/empty final artifact: {name}; shutdown refused')
    return True


def sha256(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def capture_setup(run, destination, launch):
    from training_sources import training_source_snapshot
    current = training_source_snapshot(ROOT)
    expected = launch.get('preflight', {}).get('source_sha256')
    if not expected or current != expected:
        raise RuntimeError('Current training source does not match launch fingerprint; snapshot/automatic shutdown refused')
    write_json(destination/'launch_at_monitor_start.json', launch)
    write_json(destination/'source_sha256.json', current)
    candidates = list(ROOT.glob('*.py'))
    candidates += [p for directory in ('cloud', 'experiments') for p in (ROOT/directory).rglob('*')
                   if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py', '.sh', '.json', '.txt', '.gz')]
    with tarfile.open(destination/'code_and_settings.tar.gz', 'w:gz') as archive:
        for path in sorted(set(candidates)):
            archive.add(path, arcname='project/'+path.relative_to(ROOT).as_posix(), recursive=False)
        for name in ('resolved_config.json', 'launch.json'):
            archive.add(run/name, arcname='run/'+name, recursive=False)
        for name in ('earthquake-v9-autodl5090.json', 'gpu-constraints.txt', 'pyvenv.cfg'):
            path = Path(sys.prefix)/name
            if path.is_file():
                archive.add(path, arcname='environment/'+name, recursive=False)
    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze', '--all'], capture_output=True, text=True, check=True, timeout=60)
    (destination/'pip_freeze.txt').write_text(freeze.stdout, encoding='utf-8')
    gpu = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True, timeout=15)
    (destination/'nvidia-smi.txt').write_text(gpu.stdout, encoding='utf-8')
    write_json(destination/'snapshot_info.json', dict(captured_utc=now(), run=str(run),
        note='Captured when monitor was attached, not necessarily before training. Launch config/source fingerprint verified.',
        separate_off_instance_backup=False))


def finalize(run, destination, launch, rows):
    if not completion_ready(run, launch, rows):
        raise RuntimeError('Training not successfully complete')
    manifest = {}
    for path in sorted(run.rglob('*')):
        if not path.is_file() or destination.parent in path.parents:
            continue
        if path.is_symlink():
            raise RuntimeError(f'Result symlink cannot be included in integrity manifest: {path}')
        before = path.stat()
        digest = sha256(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f'Result is still changing: {path}; shutdown refused')
        manifest[path.relative_to(run).as_posix()] = dict(bytes=after.st_size, sha256=digest)
    write_json(destination/'results_manifest.json', manifest)
    write_json(destination/'completion_summary.json', dict(completed_utc=now(),
        progress=progress(launch, rows), verified_result_files=len(manifest),
        results_directory=str(run), results_kept_in_place=True, separate_off_instance_backup=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('status', 'watch'))
    parser.add_argument('--run', type=Path, default=Path('runs/v9_b4_r01'))
    parser.add_argument('--shutdown', action='store_true', help='Power off only after verified successful completion')
    args = parser.parse_args()
    if args.mode == 'status' and args.shutdown:
        parser.error('--shutdown is only valid with watch')
    run = args.run if args.run.is_absolute() else ROOT/args.run
    run = run.resolve()
    launch, rows = read_state(run)
    print(json.dumps(progress(launch, rows), ensure_ascii=False, indent=2), flush=True)
    if args.mode == 'status':
        return
    if sys.platform != 'linux':
        raise RuntimeError('Watcher requires Linux')
    if launch['status'] not in ('running', 'reserved', 'complete'):
        raise RuntimeError('Run is not active/successful; no shutdown will be armed')
    if args.shutdown and not os.access('/usr/bin/shutdown', os.X_OK):
        raise RuntimeError('AutoDL /usr/bin/shutdown is unavailable')
    import fcntl
    parent = run/'auto_finish'
    parent.mkdir(exist_ok=True)
    lock = (parent/'watch.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stop = run/'STOP_AUTO_SHUTDOWN'
    if stop.exists():
        raise RuntimeError(f'Cancellation marker exists: {stop}')
    destination = parent/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    destination.mkdir()
    sys.path.insert(0, str(ROOT))
    capture_setup(run, destination, launch)
    print('WATCHING: setup snapshot saved; shutdown_on_success='+str(args.shutdown), flush=True)
    try:
        last_epoch = None
        while True:
            if stop.exists():
                print('CANCELLED: monitor stopped; training left untouched.', flush=True)
                return
            try:
                launch, rows = read_state(run)
            except json.JSONDecodeError:
                time.sleep(30)
                continue
            if launch['status'] in ('failed_or_interrupted', 'blocked_by_predecessor'):
                raise RuntimeError('Training failed/interrupted; results retained, instance left running')
            if completion_ready(run, launch, rows):
                finalize(run, destination, launch, rows)
                if stop.exists():
                    print('CANCELLED before shutdown.', flush=True)
                    return
                os.sync()
                print('SUCCESS: result hashes saved; shutdown_requested='+str(args.shutdown), flush=True)
                if args.shutdown:
                    subprocess.run(['/usr/bin/shutdown'], check=True)
                return
            epoch = rows[-1]['epoch'] if rows else 0
            if epoch != last_epoch:
                print(json.dumps(progress(launch, rows), ensure_ascii=False), flush=True)
                last_epoch = epoch
            time.sleep(30)
    except Exception as error:
        write_json(destination/'monitor_error.json', dict(utc=now(), error=str(error), shutdown_requested=False))
        raise


if __name__ == '__main__':
    main()
