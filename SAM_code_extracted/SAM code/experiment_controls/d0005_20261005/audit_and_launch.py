"""Audit and launch the user-requested smaller-d0 experiment without editing training code."""
import json
import os
from pathlib import Path
import subprocess
import sys
import hashlib
import h5py
import numpy as np

JOB = Path(__file__).resolve().parent
CODE = JOB.parents[1]
DATA = CODE / 'earth_dataset_greece_nz_stride64_v2'
OLD = CODE / 'train_greece_nz_stride64_unimatch_d001'
NEW = CODE / 'train_greece_nz_stride64_unimatch_d0005'

def main():
    if NEW.exists() and any(NEW.iterdir()):
        raise RuntimeError('New experiment directory is not empty; refusing to overwrite.')
    report = {'old_d0': json.loads((OLD/'ema_config.json').read_text())['d0'],
              'new_d0': .005, 'epochs': 200, 'resume': False,
              'purpose': 'Change only d0 and output path; same prepared data and training algorithm.',
              'scope': 'Full stored tiles; repeated overlapping pixels counted. Not model evaluation.',
              'data': {}, 'source_sha256': {}}
    for filename in ['trainV10.py', 'torchUnet.py', 'tool.py', 'config.py', 'unimatch_fault.py', 'aio2_online.py', 'aio2_ema.py', 'run_greece_nz_training.py']:
        report['source_sha256'][filename] = hashlib.sha256((CODE/filename).read_bytes()).hexdigest()
    with h5py.File(DATA/'faults.h5', 'r') as f:
        for split in ['train', 'val']:
            keys = (DATA/f'key_{split}.lst').read_text().split()
            rows = {str(t): {'positive_pixels': 0, 'pixels': 0, 'empty_tiles': 0} for t in [.02, .01, .005]}
            for key in keys:
                distance = f[key]['k0'][:]
                for t, stats in rows.items():
                    label = distance < float(t)
                    stats['positive_pixels'] += int(label.sum())
                    stats['pixels'] += label.size
                    stats['empty_tiles'] += int(not label.any())
            for stats in rows.values():
                stats['positive_fraction'] = stats['positive_pixels']/stats['pixels']
            report['data'][split] = {'tiles': len(keys), 'thresholds': rows}
    command = [sys.executable, str(CODE/'run_greece_nz_training.py'), '--d0', '.005', '--epochs', '200', '--res-dir', str(NEW)]
    env = dict(os.environ, MPLBACKEND='Agg', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    preflight = subprocess.run(command, cwd=CODE, env=env, text=True, capture_output=True, check=True)
    (JOB/'training_command.txt').write_text(preflight.stdout)
    report['preflight_passed'] = True
    (JOB/'threshold_audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report['data'], indent=2), flush=True)
    with (JOB/'smoke_console.log').open('x') as log:
        subprocess.run([sys.executable, str(CODE/'smoke_unimatch.py'), '--dataDir', str(DATA), '--d0', '.005', '--outDir', str(JOB/'smoke')], cwd=CODE, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    with (JOB/'console.log').open('x') as log:
        process = subprocess.Popen(command+['--execute'], cwd=CODE, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (JOB/'launch.json').write_text(json.dumps({'pid': process.pid, 'command': command+['--execute'], 'result_dir': str(NEW), 'log': str(JOB/'console.log')}, indent=2))
    print('Training launched, wrapper PID:', process.pid, flush=True)

if __name__ == '__main__':
    main()