import unittest
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from pathlib import Path
import numpy as np
import torch
from evaluate_cv import ranking_metrics, fixed_metrics, split_audit, predict_split


class EvaluationTests(unittest.TestCase):
    def test_exact_average_precision_known_example(self):
        result=ranking_metrics([.1,.4,.35,.8],[0,0,1,1])
        self.assertAlmostEqual(result['AP'],5/6)
        self.assertAlmostEqual(result['ROC_AUC'],.75)
        self.assertAlmostEqual(result['threshold'],.35)
        self.assertAlmostEqual(result['best_F1'],.8)

    def test_tied_scores_are_one_threshold_and_degenerate_classes_explicit(self):
        self.assertEqual(ranking_metrics([.5]*4,[1,0,1,0])['AP'],.5)
        self.assertEqual(ranking_metrics([.3,.2],[1,1])['AP'],1.)
        self.assertIsNone(ranking_metrics([.3,.2],[0,0])['AP'])
        with self.assertRaises(ValueError):
            ranking_metrics([float('nan')],[1])

    def test_false_positives_on_empty_regions_count_in_buffered_precision(self):
        p=np.zeros((12,12)); t=np.zeros_like(p,dtype=bool); v=np.ones_like(t)
        t[2,2]=True
        p[2,3]=1
        p[10,10]=1
        result=fixed_metrics(p,t,v,.5,1)
        self.assertEqual(result['precision'],0)
        self.assertEqual(result['buffered_precision'],.5)
        self.assertEqual(result['buffered_recall'],1)
        self.assertAlmostEqual(result['buffered_F1'],2/3)
        self.assertEqual(fixed_metrics(p,t,v,.5,0)['buffered_F1'],0)

    def test_invalid_pixels_are_excluded(self):
        p=np.ones((2,2));t=np.array([[1,0],[0,0]],dtype=bool)
        valid=t.copy()
        self.assertEqual(fixed_metrics(p,t,valid,.5)['F1'],1)

    def test_test_overlap_rejected(self):
        fake={k:{'k0':np.zeros((4,4))} for k in ['t_0_0','t_0_2','t_10_10']}
        with self.assertRaises(ValueError):
            split_audit(fake,{'train':['t_0_0'],'val':['t_10_10'],'test':['t_0_2']})

    def test_geographic_duplicate_pixels_are_merged(self):
        fake={}
        truth=np.ones((4,6),dtype='float32')
        truth[:,3]=0
        elevation=np.arange(24,dtype='float32').reshape(4,6)
        for x in [0,2]:
            fake[f't_0_{x}']={
                'CTX':np.full((4,4),127.5,dtype='float32'),
                'image':elevation[:,x:x+4].copy(),
                'TW':np.full((3,4,4),127.5,dtype='float32'),
                'k0':truth[:,x:x+4].copy()}
        class Model(torch.nn.Module):
            def forward(self,x):
                return x[:,:1]/2+.5
        with TemporaryDirectory() as directory:
            p,t,v,info=predict_split(Model(),fake,list(fake),SimpleNamespace(crop=2,d0=.02),
                                     'cpu',Path(directory),'test')
            self.assertEqual(info['unique_valid_pixels'],24)
            self.assertEqual(info['tile_weighted_valid_pixels'],32)
            self.assertEqual(info['positive_pixels'],4)
            np.testing.assert_allclose(p,.5)
            self.assertTrue(v.all())
            fake['t_0_2']['k0'][:,1]=1  # simulate a crop-local missing positive band
            p,t,v,info=predict_split(Model(),fake,list(fake),SimpleNamespace(crop=2,d0=.02),
                                     'cpu',Path(directory),'union')
            self.assertEqual(info['positive_pixels'],4)
            self.assertEqual(info['conflicting_label_pixels'],4)
            fake['t_0_2']['image'][:]=99
            with self.assertRaisesRegex(ValueError,'terrain'):
                predict_split(Model(),fake,list(fake),SimpleNamespace(crop=2,d0=.02),
                              'cpu',Path(directory),'invalid')


if __name__=='__main__':
    unittest.main()
