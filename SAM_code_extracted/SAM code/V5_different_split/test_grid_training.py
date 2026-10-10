"""Exercise actual trainer accumulation and exact checkpoint continuation on a tiny CPU model."""
import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
import h5py
import numpy as np
import torch
from grid_core import BASE
from train_grid import train

class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.not_sam = torch.nn.Sequential(torch.nn.Conv2d(7,3,1), torch.nn.Softmax(1))
        self.in_sam = torch.nn.ModuleList([torch.nn.Identity()])
    def forward(self, x): return self.not_sam(x)

class TrainingTests(unittest.TestCase):
    def test_resume_matches_uninterrupted_including_partial_accumulation(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory(dir=BASE) as tmp:
            d = Path(tmp)/'data'; d.mkdir()
            with h5py.File(d/'faults.h5','w') as f:
                for i in range(4):
                    g=f.create_group(str(i)); g['valid']=np.ones((384,384),np.uint8)
                    g['CTX']=np.full((384,384),100+i,np.float32)
                    g['image']=np.tile(np.arange(384,dtype=np.float32),(384,1))
                    g['TW']=np.full((3,384,384),100,np.float32)
                    label=np.full((384,384),.1,np.float32); label[:,100:110]=0
                    for k in ('k0','k1','k2'): g[k]=label
            (d/'manifest.json').write_text('{}')
            (d/'key_train.lst').write_text('0\n1\n2\n')
            (d/'key_val.lst').write_text('3\n')
            def args(out,steps,resume=''):
                return SimpleNamespace(modelType='mars_sam',k0Only=True,lossFunc='FocalTversky',isalign=False,
                    datasetAdd='',isOther=False,emaResume='',emaStageTransition=False,mn='256',
                    dataset=str(d/'key_train.lst'),resDir=str(out),resume=str(resume),seed=7,device='cpu',
                    dropout=0.,emaTeacher=True,emaDecay=.98,lr0=.0001,batchsize=2,stepPerMiniBatch=4,
                    maxSteps=steps,warmupSteps=20,onlineCorrection=False,onlineStartStep=800,onlineHoldStep=1000,
                    onlineRampStep=1600,unimatch=False,unimatchStartStep=800,unimatchRampStep=1600,
                    pu=0.,puWarmupSteps=400,puRampSteps=400,d0=.02,dw=.08,alpha=.1,evalEverySteps=1,emaPreview=False)
            full=Path(tmp)/'full'; part=Path(tmp)/'part'
            train(args(full,3), TinyModel)
            train(args(part,1), TinyModel)
            train(args(part,3,part/'grid_checkpoint.pth'), TinyModel)
            a=torch.load(full/'grid_checkpoint.pth',weights_only=False,map_location='cpu')
            b=torch.load(part/'grid_checkpoint.pth',weights_only=False,map_location='cpu')
            self.assertEqual(b['global_step'],3)
            self.assertEqual(b['teacher']['num_updates'],3)
            for name in a['student']:
                torch.testing.assert_close(a['student'][name],b['student'][name],rtol=0,atol=0)
            rows=[json.loads(s) for s in (part/'history.jsonl').read_text().splitlines()]
            self.assertEqual([r['global_step'] for r in rows],[1,2,3])
            self.assertTrue(all(r['samples']==3 for r in rows))

if __name__=='__main__': unittest.main()
