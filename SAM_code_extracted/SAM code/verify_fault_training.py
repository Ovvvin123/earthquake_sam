"""Small real SAM CLI integration, including masked supervision and resumed training.

Uses copied data and an artificial known mask solely to test plumbing. It is NOT
a reviewed geological label mask. Logs/report survive; temporary checkpoints do not.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import h5py
import numpy as np
from verify_lora_cli import make_fixture

HERE=Path(__file__).resolve().parent


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--outDir',type=Path,required=True)
    args=ap.parse_args()
    out=args.outDir.resolve()
    if out.exists() and any(out.iterdir()): ap.error('Use an empty output directory')
    out.mkdir(parents=True,exist_ok=True)
    records=[]
    def run(name,cmd,expect_success=True):
        env=dict(os.environ,MPLBACKEND='Agg',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        with (out/f'{name}.log').open('w') as log:
            result=subprocess.run([sys.executable,'-u',*map(str,cmd)],cwd=HERE,env=env,
                                  stdout=log,stderr=subprocess.STDOUT)
        records.append(dict(name=name,exit_code=result.returncode,expected_success=expect_success))
        (out/'commands.json').write_text(json.dumps(records,indent=2))
        if (result.returncode==0)!=expect_success: raise RuntimeError(f'{name}: inspect {out/name}.log')
        print(name,'passed',flush=True)
    with tempfile.TemporaryDirectory(dir=HERE) as directory:
        tmp=Path(directory); data=tmp/'data'
        # Six batches per epoch => one full four-batch window and one two-batch tail.
        make_fixture(HERE/'earth_dataset_greece_nz_stride64_v2',data,12,2)
        with h5py.File(data/'faults.h5','a') as f:
            for key in f:
                shape=f[key]['k0'].shape
                known=np.ones(shape,dtype=np.uint8);known[:, :shape[1]//4]=0
                f[key]['known_test']=known
        common=['trainV10.py','--job','train','--modelType','mars_sam','--dataset',data/'key_train.lst',
                '--mn','256','--batchsize','2','--stepPerMiniBatch','4','--lr0','1e-4',
                '--d0','.01','--dw','.08','--alpha','.1','--k0Only','--fixedVal','--seed','7',
                '--samAdaptation','lora','--supervisionKey','known_test']
        for name,loss,extra,resume_flag,checkpoint in [
            ('bce','BCEWithLogits',[],'--resume','training_checkpoint.pth'),
            ('mixed_ema','BCEFocalTversky',['--emaTeacher','--onlineCorrection','--unimatch',
             '--consistencyLoss','logits','--onlineStart','1','--onlineHoldEnd','1','--onlineRampEnd','2',
             '--unimatchStart','1','--unimatchRampEnd','2'],'--emaResume','ema_checkpoint.pth')]:
            folder=tmp/name
            cmd=common+['--resDir',str(folder)+'/', '--lossFunc',loss]+extra
            run(name,cmd+['--epochs','1'])
            run(name+'_resume',cmd+['--epochs','2',resume_flag,folder/checkpoint])
            rows=[json.loads(x) for x in (folder/'adaptation_history.jsonl').read_text().splitlines()]
            assert [r['global_step'] for r in rows]==[2,4]
            assert all(r['validation_health']['supervision']=='mask' for r in rows)
            import torch
            ck=torch.load(folder/checkpoint,map_location='cpu',weights_only=False)
            assert not ck['gradients'], 'Corrected accumulation must not leave pending gradients'
            assert ck['config']['supervisionKey']=='known_test'
            assert ck['config']['accumulationVersion']=='window_v2'
            run(name+'_reject_changed_loss',cmd+['--epochs','3',resume_flag,folder/checkpoint,
                                                '--tverskyWeight','.75'],False)
        # New losses also work through the ordinary (non-fixedVal) validation branch.
        ordinary=[v for v in common if v!='--fixedVal']
        index=ordinary.index('--supervisionKey'); del ordinary[index:index+2]
        index=ordinary.index('--samAdaptation');ordinary[index+1]='full'
        run('ordinary_validation',ordinary+['--resDir',str(tmp/'ordinary')+'/',
                                            '--lossFunc','BCEWithLogits','--epochs','1'])
    (out/'report.json').write_text(json.dumps(dict(passed=True,commands=records,
         note='Real-data CLI plumbing only; artificial known mask; not a convergence or geological accuracy test'),indent=2))
    print('All CLI checks passed',flush=True)


if __name__=='__main__': main()
