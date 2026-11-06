#!/usr/bin/env python3
"""Validate the prepared dataset and print a training command; --execute runs it."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import math
from prepare_greece_nz import HERE, REGIONS, DEFAULT_DATA_DIR, overlap_audit


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir',type=Path,default=DEFAULT_DATA_DIR)
    ap.add_argument('--res-dir',type=Path,default=HERE/'train_greece_nz_stride64_unimatch_d001')
    ap.add_argument('--epochs',type=int,default=200)
    ap.add_argument('--lossFunc', choices=('FocalTversky', 'BCEWithLogits', 'BCEFocalTversky'), default='FocalTversky')
    ap.add_argument('--tverskyWeight', type=float, default=.5)
    ap.add_argument('--lr0', type=float, default=1e-4)
    ap.add_argument('--supervisionKey', default='')
    ap.add_argument('--accumulationVersion', choices=('legacy_v1', 'window_v2'), default='window_v2')
    ap.add_argument('--collapseGuard', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--consistencyLoss', choices=('probability', 'logits'), default='probability')
    ap.add_argument('--d0',type=float,default=.01,help='k0 distance threshold; matches train_v5_unimatch_d001 by default')
    ap.add_argument('--execute',action='store_true')
    from sam2_lora import add_adaptation_arguments, adaptation_config
    add_adaptation_arguments(ap)
    ap.add_argument('--trainingRecipe', choices=('supervised','unimatch'), default='unimatch')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--resume', type=Path, help='Full checkpoint matching the selected training recipe')
    args=ap.parse_args()
    if not math.isfinite(args.lr0) or args.lr0 <= 0 or not math.isfinite(args.tverskyWeight) or args.tverskyWeight < 0:
        ap.error('lr0 must be positive and tverskyWeight nonnegative; both must be finite')
    try:
        adaptation_config(args)
    except ValueError as error:
        ap.error(str(error))
    data=args.data_dir.resolve();res=args.res_dir.resolve()
    if not (data/'manifest.json').is_file() or not (data/'faults.h5').is_file():
        ap.error('real dataset has not been built: supply the GEBCO paths to prepare_greece_nz.py first')
    man=json.loads((data/'manifest.json').read_text())
    rows=man['tiles']
    if {t['region'] for t in rows} != set(REGIONS):
        ap.error('this experiment must contain Greece and New Zealand only')
    if overlap_audit(rows)['train_val_overlap_pairs']:
        ap.error('train/validation footprints overlap')
    import h5py
    with h5py.File(data/'faults.h5','r') as f:
        listed=[]
        for split in ['train','val']:
            keys=(data/f'key_{split}.lst').read_text().split()
            expected={t['key'] for t in rows if t['split']==split}
            if not keys or len(set(keys))!=len(keys) or set(keys)!=expected:
                ap.error(f'{split}: list and manifest disagree or are empty')
            if {k.split('__')[0] for k in keys}!=set(REGIONS):
                ap.error(f'{split}: both regions must be represented')
            for k in keys:
                if k not in f or f[k].attrs['split']!=split:
                    ap.error(f'{k}: missing HDF5 data or split mismatch')
                if args.supervisionKey:
                    if args.supervisionKey not in f[k]:
                        ap.error(f'{k}: missing supervision mask {args.supervisionKey}; prepare reviewed masks first')
                    import numpy as np
                    known = f[k][args.supervisionKey][:]
                    if known.shape != f[k]['k0'].shape or not np.isin(known, [0, 1]).all():
                        ap.error(f'{k}: supervision mask must be binary and match k0 shape')
            listed.extend(keys)
        if len(set(listed))!=len(listed) or set(listed)!=set(f):
            ap.error('keys duplicated between splits or HDF5 keys disagree')
        if len([t for t in rows if t['split']=='train'])<2:
            ap.error('need at least two train tiles for batch size 2')
    if args.resume and not args.resume.is_file():
        ap.error('resume checkpoint does not exist')
    if not args.resume and res.exists() and any(res.iterdir()):
        ap.error('use an empty result directory to protect prior runs')
    if args.epochs<1 or not 0<args.d0<.35:
        ap.error('invalid epochs/d0')
    cmd=[sys.executable,str(HERE/'trainV10.py'),'--job','train','--modelType','mars_sam',
         '--dataset',str(data/'key_train.lst'),'--resDir',str(res)+'/',
         '--mn','256','--batchsize','2','--stepPerMiniBatch','4','--epochs',str(args.epochs),
         '--lr0',str(args.lr0),'--d0',str(args.d0),'--dw','.08','--alpha','.1','--dropout','0',
         '--lossFunc',args.lossFunc,'--k0Only','--pu','0','--fixedVal',
         '--emaTeacher','--emaDecay','.98','--emaPreview',
         '--previewCount','6','--previewEvery','5','--previewThreshold','.7',
         '--previewBuffer','2','--previewMinPixels','8','--checkpointEvery','20',
         '--onlineCorrection','--onlineStart','41','--onlineHoldEnd','50','--onlineRampEnd','80',
         '--onlineWeightStart','.05','--onlineWeightMax','.2','--onlineThreshold','.9',
         '--onlineSupport','.5','--onlineBuffer','3','--onlineMinPixels','16','--onlineMaxFraction','.05',
         '--unimatch','--unimatchStart','41','--unimatchRampEnd','80','--unimatchWeight','.1',
         '--unimatchContrast','.2','--unimatchBrightness','.1','--unimatchNoise','.03']
    if args.trainingRecipe == 'supervised':
        cmd = cmd[:cmd.index('--emaTeacher')]
    cmd += ['--tverskyWeight', str(args.tverskyWeight), '--supervisionKey', args.supervisionKey,
            '--accumulationVersion', args.accumulationVersion, '--consistencyLoss', args.consistencyLoss,
            '--collapseGuard' if args.collapseGuard else '--no-collapseGuard']
    cmd += ['--seed', str(args.seed)]
    for key, value in adaptation_config(args).items():
        cmd += ['--'+key, str(value)]
    if args.resume:
        cmd += ['--emaResume' if args.trainingRecipe == 'unimatch' else '--resume', str(args.resume.resolve())]
    print(shlex.join(cmd),flush=True)
    if args.execute:
        # Match activate_wsl.sh used by the California UniMatch baseline.
        env=os.environ.copy()
        env.update(MPLBACKEND='Agg',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        subprocess.run(cmd,cwd=HERE,env=env,check=True)


if __name__=='__main__':
    main()
