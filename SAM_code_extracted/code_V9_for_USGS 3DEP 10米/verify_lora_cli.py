"""Opt-in integration verification with small copies of real training/validation tiles.

Runs actual training, continuation and inference entry points. Temporary HDF5 and
checkpoints are removed after verification; logs and a compact report are retained.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import h5py

HERE = Path(__file__).resolve().parent


def make_fixture(source, destination, train_count, val_count):
    destination.mkdir()
    keys = {}
    with h5py.File(source/'faults.h5', 'r') as src, h5py.File(destination/'faults.h5', 'w') as dst:
        for split, count in [('train', train_count), ('val', val_count)]:
            keys[split] = (source/f'key_{split}.lst').read_text().split()[:count]
            for key in keys[split]:
                src.copy(key, dst)
            (destination/f'key_{split}.lst').write_text('\n'.join(keys[split])+'\n')
        (destination/'manifest.json').write_text(json.dumps(dict(purpose='CLI fixture', keys=keys)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outDir', type=Path, required=True)
    parser.add_argument('--only-grid', action='store_true', help='Only verify the isolated grid entry points')
    args = parser.parse_args()
    out = args.outDir.resolve()
    if out.exists() and any(out.iterdir()): parser.error('Use an empty output directory')
    out.mkdir(parents=True, exist_ok=True)
    results = []

    def run(name, cmd, cwd=HERE):
        print(name, flush=True)
        import os
        env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', MPLBACKEND='Agg')
        with (out/f'{name}.log').open('w') as stream:
            completed = subprocess.run([sys.executable, '-u', *map(str, cmd)], cwd=cwd,
                                       env=env, stdout=stream, stderr=subprocess.STDOUT)
        results.append(dict(name=name, exit_code=completed.returncode,
                            command=[str(a) for a in cmd], log=f'{name}.log'))
        (out/'commands.json').write_text(json.dumps(results, indent=2))
        if completed.returncode:
            raise RuntimeError(f'{name} failed; see {out/name}.log')

    # GridDataset intentionally rejects paths outside the isolated experiment.
    with tempfile.TemporaryDirectory(dir=HERE/'V5_different_split') as directory:
        tmp = Path(directory)
        if not args.only_grid:
            data = tmp/'main_data'
            make_fixture(HERE/'earth_dataset_greece_nz_stride64_v2', data, 4, 2)
            common = ['trainV10.py','--job','train','--modelType','mars_sam',
                '--dataset',data/'key_train.lst','--mn','256','--batchsize','2',
                '--stepPerMiniBatch','1','--lr0','1e-4','--d0','.01','--dw','.08','--alpha','.1',
                '--lossFunc','FocalTversky','--k0Only','--pu','0','--fixedVal','--seed','7']
            for mode in ('frozen','lora'):
                folder=tmp/mode
                command=common+['--samAdaptation',mode,'--resDir',str(folder)+'/']
                run(f'main_{mode}', command+['--epochs','1'])
                run(f'main_{mode}_resume', command+['--epochs','2','--resume',folder/'training_checkpoint.pth'])
                history=[json.loads(line) for line in (folder/'adaptation_history.jsonl').read_text().splitlines()]
                assert [r['global_step'] for r in history] == [2,4]
            teacher_dir=tmp/'teacher'
            teacher=common+['--samAdaptation','lora','--resDir',str(teacher_dir)+'/',
                '--emaTeacher','--emaDecay','.98','--onlineCorrection','--unimatch',
                '--onlineStart','1','--onlineHoldEnd','1','--onlineRampEnd','2',
                '--unimatchStart','1','--unimatchRampEnd','2']
            run('main_lora_teacher',teacher+['--epochs','1'])
            run('main_lora_teacher_resume',teacher+['--epochs','2','--emaResume',teacher_dir/'ema_checkpoint.pth'])
            run('main_lora_evaluate',['evaluate.py','--resDir',tmp/'lora','--dataDir',data,
                                     '--keyFile','key_val.lst','--d0','.01'])
            run('main_lora_teacher_evaluate',['evaluate.py','--resDir',teacher_dir/'teacher',
                     '--dataDir',data,'--keyFile','key_val.lst','--d0','.01'])
        grid=HERE/'V5_different_split'
        grid_data=tmp/'grid_data'
        make_fixture(grid/'datasets/ca_stride384',grid_data,3,1)
        grid_out=tmp/'grid_lora'
        grid_cmd=['trainV10.py','--dataset',grid_data/'key_train.lst','--resDir',str(grid_out)+'/',
            '--modelType','mars_sam','--mn','256','--batchsize','2','--stepPerMiniBatch','2',
            '--lossFunc','FocalTversky','--k0Only','--pu','0','--d0','.02','--dw','.08',
            '--alpha','.1','--samAdaptation','lora','--evalEverySteps','1']
        run('grid_lora',grid_cmd+['--maxSteps','1'],grid)
        run('grid_lora_resume',grid_cmd+['--maxSteps','2','--resume',grid_out/'grid_checkpoint.pth'],grid)
        run('grid_lora_evaluate',['evaluate.py','--resDir',grid_out,'--dataDir',grid_data],grid)
        history=[json.loads(line) for line in (grid_out/'history.jsonl').read_text().splitlines()]
        assert [r['global_step'] for r in history] == [1,2]
        report=dict(passed=True, runs=results, main_steps=None if args.only_grid else [2,4], grid_steps=[1,2],
                    scope='Small real-data CLI integration checks; not accuracy experiments')
        (out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(passed=True, commands=len(results), report=str(out/'report.json'))))


if __name__ == '__main__': main()
