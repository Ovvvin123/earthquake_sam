"""No real training: sampler, schedule, collision and queue-failure tests."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch,Mock

import h5py
import numpy as np
import torch

import run_experiment as launcher
from known_crops import crop_inventory
from aio2_online import pseudo_weight
from unimatch_fault import consistency_weight


class ExperimentTests(unittest.TestCase):
    def test_profiles_and_schedule_boundaries(self):
        a=launcher.profile_arguments(launcher.HERE/'experiments/california_ds5.json')
        b=launcher.profile_arguments(launcher.HERE/'experiments/greece_nz_original.json')
        self.assertEqual((a['epochs'],b['epochs']),(500,200))
        self.assertEqual(a['supervisionKey'],'label_known')
        self.assertEqual(b['supervisionKey'],'')
        self.assertNotEqual(a['resDir'],b['resDir'])
        for config,start,hold,end in [(a,121,150,240),(b,41,50,80)]:
            args=SimpleNamespace(**config)
            self.assertEqual(pseudo_weight(start-1,args),0)
            self.assertEqual(consistency_weight(start-1,args),0)
            self.assertEqual(pseudo_weight(start,args),.05)
            self.assertEqual(pseudo_weight(hold,args),.05)
            self.assertGreater(pseudo_weight(hold+1,args),.05)
            self.assertAlmostEqual(pseudo_weight(end,args),.2)
            self.assertAlmostEqual(consistency_weight(end,args),.1)
            self.assertAlmostEqual(consistency_weight(end+1,args),.1)
            self.assertTrue(args.emaTeacher)

    def test_known_edge_crops_alignment_and_stable_validation(self):
        from tool import Generator_new
        with TemporaryDirectory() as folder:
            path=Path(folder)/'faults.h5'
            mask=np.zeros((384,384),dtype=np.float32);mask[0:16,0:16]=1
            with h5py.File(path,'w') as f:
                for key in ['edge','unknown']:
                    g=f.create_group(key)
                    g['CTX']=mask*255;g['image']=np.arange(384*384,dtype=np.float32).reshape(384,384)
                    g['TW']=np.stack([mask*255]*3)
                    for k in ['k0','k1','k2']:g[k]=np.where(mask,.001,.1).astype(np.float32)
                    g['label_known']=mask if key=='edge' else mask*0
            windows,excluded=crop_inventory(path,['edge','unknown'],'label_known')
            self.assertEqual(excluded,['unknown'])
            self.assertTrue(windows['edge'])
            reader=Generator_new(path,['edge','unknown'],training=True,
                                 supervision_key='label_known',known_crop=True)
            self.assertEqual(reader.list,['edge'])
            for seed in range(16):
                np.random.seed(seed);x,d=reader[0]
                self.assertGreater(d[3].sum(),0)
                np.testing.assert_array_equal(x[0]>0,d[3]>.5)
                np.testing.assert_array_equal(d[0]<.01,d[3]>.5)
            val=Generator_new(path,['edge'],training=False,fixed_crop=True,
                              supervision_key='label_known',known_crop=True)
            np.random.seed(1);x1,d1=val[0]
            np.random.seed(9);x2,d2=val[0]
            np.testing.assert_array_equal(x1,x2);np.testing.assert_array_equal(d1,d2)
            self.assertEqual(int(d1[3].sum()),256)
            from aio2_preview import FixedPreview
            preview=FixedPreview(Generator_new,path,['edge'],Path(folder)/'preview',256,256,.01,
                count=1,dataset_options=dict(supervision_key='label_known',known_crop=True))
            key,image,truth,valid=preview.samples[0]
            self.assertEqual(int(preview.known_masks[key].sum()),256)
            self.assertFalse((truth & ~preview.known_masks[key]).any())

    def test_output_collisions_and_existing_results_rejected(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);a=dict(resDir=str(root/'run'),dataFile=str(root/'data/faults.h5'))
            with self.assertRaises(ValueError):launcher.check_outputs([a,a])
            (root/'run').mkdir();(root/'run'/'sentinel').write_text('keep')
            with self.assertRaises(FileExistsError):launcher.check_outputs([a])
            self.assertEqual((root/'run'/'sentinel').read_text(),'keep')

    def test_failed_predecessor_never_launches_second(self):
        with TemporaryDirectory() as folder:
            root=Path(folder)
            profiles=[dict(resDir=str(root/f'run{i}'),dataFile=str(root/'data/faults.h5')) for i in range(2)]
            from sam2_lora import sha256_file
            reports=[dict(identity={},source_sha256={p.name:sha256_file(p) for p in launcher.HERE.glob('*.py')})]*2
            child=Mock(pid=123);child.wait.return_value=1
            with patch('sam2_lora.checkpoint_identity',return_value={}),patch.object(launcher.subprocess,'Popen',return_value=child) as popen:
                with self.assertRaises(RuntimeError):launcher.execute_sequence(profiles,reports)
                self.assertEqual(popen.call_count,1)
            second=json.loads((root/'run1/launch.json').read_text())
            self.assertEqual(second['status'],'blocked_by_predecessor')
            self.assertFalse((root/'run1/console.log').exists())

    def test_completion_requires_final_epoch_and_matching_checkpoint(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);args=dict(resDir=str(root),epochs=500)
            (root/'adaptation_history.jsonl').write_text('{"epoch":499}\n')
            with self.assertRaises(RuntimeError):launcher.verify_finished(args,{})
            (root/'adaptation_history.jsonl').write_text('{"epoch":500}\n')
            state=dict(format='aio2_ema_v1',next_epoch=499,config=args,identity={},
                       teacher={'num_updates':1},student={'x':torch.tensor(1)})
            torch.save(state,root/'ema_checkpoint.pth')
            with self.assertRaises(RuntimeError):launcher.verify_finished(args,{})
            state['next_epoch']=500;torch.save(state,root/'ema_checkpoint.pth')
            launcher.verify_finished(args,{})
            with self.assertRaises(RuntimeError):launcher.verify_finished(args,{'different':1})

    def test_sequence_verifies_predecessor_before_next_launch(self):
        from sam2_lora import sha256_file
        for incomplete in (False,True):
            with TemporaryDirectory() as folder:
                root=Path(folder)
                profiles=[dict(resDir=str(root/f'run{i}'),dataFile=str(root/'data/faults.h5')) for i in range(2)]
                reports=[dict(identity={},source_sha256={p.name:sha256_file(p) for p in launcher.HERE.glob('*.py')})]*2
                events=[]
                def spawn(*args,**kwargs):
                    events.append('launch');child=Mock(pid=123);child.wait.return_value=0;return child
                def verify(*args):
                    events.append('verify')
                    if incomplete: raise RuntimeError('Incomplete final epoch')
                with patch('sam2_lora.checkpoint_identity',return_value={}),\
                     patch.object(launcher.subprocess,'Popen',side_effect=spawn),\
                     patch.object(launcher,'verify_finished',side_effect=verify):
                    if incomplete:
                        with self.assertRaises(RuntimeError):launcher.execute_sequence(profiles,reports)
                        self.assertEqual(events,['launch','verify'])
                    else:
                        launcher.execute_sequence(profiles,reports)
                        self.assertEqual(events,['launch','verify','launch','verify'])

    def test_default_cli_does_not_execute(self):
        args=['run_experiment.py','--sequence','experiments/california_ds5.json']
        data={'train':{'selected_samples':80},'val':{'selected_samples':10}}
        with patch('sys.argv',args),patch.object(launcher,'check_outputs'),\
             patch.object(launcher,'validate_parser',return_value={'emaTeacher':True}),\
             patch.object(launcher,'inspect_data',return_value=data),\
             patch('sam2_lora.checkpoint_identity',return_value={}),\
             patch.object(launcher,'execute_sequence') as run:
            launcher.main();run.assert_not_called()


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
