"""Validate by default. --execute runs profiles sequentially, never concurrently.

Example: python run_experiment.py --sequence experiments/california_ds5.json
         experiments/greece_nz_original.json [--execute]
No automatic retries/resumes. A failed or incomplete predecessor stops the queue.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
from training_sources import training_source_snapshot, verify_training_sources

HERE=Path(__file__).resolve().parent


def read_profile(path, seen=None):
    path=Path(path).resolve(); seen=set() if seen is None else seen
    if path in seen: raise ValueError('Cyclic profile inheritance')
    seen.add(path)
    raw=json.loads(path.read_text(encoding='utf-8'))
    if set(raw)-{'extends','arguments'}: raise ValueError(f'Unknown profile fields: {path}')
    args=read_profile(path.parent/raw['extends'],seen) if raw.get('extends') else {}
    args.update(raw.get('arguments',{}))
    return args


def profile_arguments(path):
    args=read_profile(path)
    for name in ('dataset','dataFile','resDir'):
        if not args.get(name): raise ValueError(f'{name} must be explicit')
        args[name]=str((HERE/args[name]).resolve())
    args['resDir']+=os.sep
    if args.get('resume') or args.get('emaResume'):
        raise ValueError('This launcher creates fresh experiments; resume is a separate operation')
    return args


def command(args):
    result=[sys.executable,str(HERE/'trainV10.py')]
    for name,value in args.items():
        if isinstance(value,bool):
            if value: result.append('--'+name)
            elif name=='collapseGuard': result.append('--no-collapseGuard')
            else: raise ValueError(f'Omit false store_true option {name}')
        else: result.extend(['--'+name,str(value)])
    return result


def validate_parser(cmd):
    # Import only argparse configuration, NEVER trainV10 (which starts training).
    result=subprocess.run([sys.executable,'-c',
        'import config,json; print(json.dumps(vars(config.args)))',*cmd[2:]],
        cwd=HERE,capture_output=True,text=True,check=True)
    return json.loads(result.stdout)


def check_outputs(profiles):
    outputs=[Path(a['resDir']).resolve() for a in profiles]
    for i,path in enumerate(outputs):
        if path.exists(): raise FileExistsError(f'Output already exists; refusing overwrite: {path}')
        for other in outputs[:i]:
            if path==other or path in other.parents or other in path.parents:
                raise ValueError('Experiment output directories must be separate')
        for a in profiles:
            data=Path(a['dataFile']).parent
            if path==data or path in data.parents or data in path.parents:
                raise ValueError('Outputs must be separate from datasets')


def inspect_data(args):
    import h5py
    import numpy as np
    from tool import Generator_new
    if args.get('fullTile'):
        from v7_dataset import FullTileGenerator as Generator_new
    dimensions = str(args.get('mn', '256')).split(',')
    n, m = int(dimensions[0]), int(dimensions[-1])
    from known_crops import crop_inventory
    directory=Path(args['dataset']).parent
    keys={s:(directory/f'key_{s}.lst').read_text().split() for s in ('train','val')}
    if any(not v or len(v)!=len(set(v)) for v in keys.values()):
        raise ValueError('Empty/duplicate train or val list')
    if set(keys['train'])&set(keys['val']): raise ValueError('Train and val keys overlap')
    if Path(args['dataset']).read_text().split()!=keys['train']:
        raise ValueError('--dataset must identify the active key_train.lst')
    result={}
    with h5py.File(args['dataFile'],'r') as f:
        for split,listed in keys.items():
            for k in listed:
                for name in ('CTX','image','TW','k0','k1'):
                    if name not in f[k]: raise ValueError(f'{k}: missing {name}')
                shape=f[k]['CTX'].shape
                if shape!=f[k]['image'].shape or shape!=f[k]['k0'].shape:
                    raise ValueError(f'{k}: image/label shape mismatch')
    # Local RNG so preflight is repeatable and cannot modify experiment RNG state.
    state=np.random.get_state(); np.random.seed(args['seed'])
    try:
        for split,listed in keys.items():
            reader=Generator_new(args['dataFile'],listed,m=m,n=n,training=split=='train',
                fixed_crop=split=='val',supervision_key=args.get('supervisionKey',''),
                known_crop=args.get('knownCrop',False),
                min_known_pixels=args.get('minKnownPixels',1),
                known_crop_stride=args.get('knownCropStride',32))
            counts=[]
            for index in range(len(reader)):
                image,label=reader[index]
                if image.shape!=(7,n,m) or not np.isfinite(image).all() or not np.isfinite(label).all():
                    raise ValueError('Nonfinite or malformed production-loader sample')
                valid=(image[2:4]>.5).all(0)
                if args.get('knownCrop'):
                    known=valid&(label[3]>.5)
                    if known.sum()<args.get('minKnownPixels',1):
                        raise ValueError('Known crop lost its supervision')
                    counts.append(int(known.sum()))
                reader.dataD.clear()
            result[split]=dict(source_samples=len(listed),selected_samples=len(reader),
                selected_keys=reader.list,excluded_keys=reader.excluded_keys,
                minimum_sampled_known_pixels=min(counts) if counts else None,
                crop_protocol=('known_weighted_D4' if split=='train' else 'fixed_max_known')
                if args.get('knownCrop') else
                ('full_tile_D4' if split=='train' else 'full_tile_fixed') if args.get('fullTile') else
                ('legacy_random_augmented' if split=='train' else 'fixed_center'))
    finally: np.random.set_state(state)
    if result['train']['selected_samples']<args['batchsize']:
        raise ValueError('Fewer training samples than one batch')
    if 'greece_nz' in directory.name:
        from prepare_greece_nz import overlap_audit,REGIONS
        manifest=json.loads((directory/'manifest.json').read_text())
        if {t['region'] for t in manifest['tiles']}!=set(REGIONS):
            raise ValueError('Both Greece and New Zealand are required')
        if overlap_audit(manifest['tiles'])['train_val_overlap_pairs']:
            raise ValueError('Greece/NZ spatial overlap')
        with h5py.File(args['dataFile'],'r') as f:
            for split,listed in keys.items():
                if set(listed)!={t['key'] for t in manifest['tiles'] if t['split']==split}:
                    raise ValueError('Greece/NZ list/manifest mismatch')
                if any(f[k].attrs['split']!=split for k in listed):
                    raise ValueError('Greece/NZ HDF5 split mismatch')
    result['expected_updates_per_epoch']=math.ceil(
        (result['train']['selected_samples']//args['batchsize'])/args['stepPerMiniBatch'])
    result['supervision']='known_mask' if args.get('supervisionKey') else 'original_catalog_background_assumption'
    return result


def verify_finished(args, expected_identity):
    import torch
    folder=Path(args['resDir'])
    with (folder/'adaptation_history.jsonl').open() as f:
        rows=[json.loads(line) for line in f if line.strip()]
    if not rows or rows[-1]['epoch']!=args['epochs']:
        raise RuntimeError('Training history did not reach the requested final epoch')
    state=torch.load(folder/'ema_checkpoint.pth',map_location='cpu',weights_only=True)
    if state.get('format')!='aio2_ema_v1' or state.get('next_epoch')!=args['epochs']:
        raise RuntimeError('Final EMA checkpoint is missing/incomplete')
    saved=state['config']
    for key,value in args.items():
        if saved.get(key)!=value:
            raise RuntimeError(f'Final checkpoint configuration mismatch: {key}')
    if state.get('identity')!=expected_identity:
        raise RuntimeError('Final checkpoint dataset/base identity mismatch')
    if not state['teacher']['num_updates'] or not state['student']:
        raise RuntimeError('Empty training state')
    del state


def execute_sequence(profiles, reports):
    from sam2_lora import checkpoint_identity
    from uuid import uuid4
    queue_id=str(uuid4())
    # Reserve every output before launching the first process. A second launcher
    # cannot obtain the same directories, even if both preflights raced.
    check_outputs(profiles)
    for args,report in zip(profiles,reports):
        folder=Path(args['resDir']); folder.mkdir(parents=True,exist_ok=False)
        record=dict(queue_id=queue_id,status='reserved',created_utc=datetime.now(timezone.utc).isoformat(),
                    command=command(args),resolved_config=args,preflight=report)
        (folder/'launch.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
        (folder/'resolved_config.json').write_text(json.dumps(args,indent=2),encoding='utf-8')
    for index,(args,report) in enumerate(zip(profiles,reports)):
        folder=Path(args['resDir']); record_path=folder/'launch.json'
        record=json.loads(record_path.read_text())
        try:
            if checkpoint_identity(args)!=report['identity']:
                raise RuntimeError('Dataset or base weights changed after preflight')
            verify_training_sources(HERE, report['source_sha256'])
            env=os.environ.copy();env.update(MPLBACKEND='Agg',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
            with (folder/'console.log').open('x',encoding='utf-8') as log:
                child=subprocess.Popen(command(args)[:1]+['-u']+command(args)[1:],
                    cwd=HERE,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                record.update(status='running',pid=child.pid)
                record_path.write_text(json.dumps(record,indent=2))
                print(f'RUN {index+1}/{len(profiles)} PID={child.pid} LOG={folder/"console.log"}',flush=True)
                try: code=child.wait()
                except BaseException:
                    child.terminate()
                    try: child.wait(timeout=30)
                    except subprocess.TimeoutExpired: child.kill(); child.wait()
                    raise
            if code!=0: raise RuntimeError(f'Training exited with status {code}; queue stopped')
            verify_finished(args,report['identity'])
            record.update(status='complete',completed_utc=datetime.now(timezone.utc).isoformat())
            record_path.write_text(json.dumps(record,indent=2))
        except BaseException as error:
            record.update(status='failed_or_interrupted',error=str(error))
            record_path.write_text(json.dumps(record,indent=2))
            for pending in profiles[index+1:]:
                p=Path(pending['resDir'])/'launch.json'; item=json.loads(p.read_text())
                item.update(status='blocked_by_predecessor');p.write_text(json.dumps(item,indent=2))
            raise


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sequence',nargs='+',type=Path,required=True)
    ap.add_argument('--execute',action='store_true',help='Explicitly authorize real sequential training')
    ap.add_argument('--reportDir',type=Path,help='Optional NEW directory for preflight reports only')
    cli=ap.parse_args()
    profiles=[profile_arguments(p) for p in cli.sequence]
    check_outputs(profiles)
    if cli.reportDir and cli.reportDir.exists(): raise FileExistsError(cli.reportDir)
    from sam2_lora import checkpoint_identity,sha256_file
    reports=[]
    for path,args in zip(cli.sequence,profiles):
        cmd=command(args); resolved=validate_parser(cmd)
        if not resolved['emaTeacher']: raise ValueError('This sequence requires EMA final checkpoints')
        print('CHECK',path,flush=True)
        data=inspect_data(args)
        report=dict(profile=str(path.resolve()),command=cmd,resolved_config=resolved,
                    data=data,identity=checkpoint_identity(args),
                    source_scope='trainV10_local_imports_v1',
                    source_sha256=training_source_snapshot(HERE))
        reports.append(report)
        print('PASS',json.dumps({k:v for k,v in data.items() if k not in ('train','val')}),flush=True)
        print('SAMPLES',data['train']['selected_samples'],data['val']['selected_samples'],flush=True)
        print(shlex.join(cmd),flush=True)
    if cli.reportDir:
        cli.reportDir.mkdir(parents=True,exist_ok=False)
        for path,report in zip(cli.sequence,reports):
            (cli.reportDir/(path.stem+'.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
    if cli.execute: execute_sequence(profiles,reports)
    else: print('CHECK ONLY: no training started; no experiment output directories created.',flush=True)


if __name__=='__main__': main()
