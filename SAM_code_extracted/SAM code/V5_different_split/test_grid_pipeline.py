"""Meaningful tests for non-overlap, masks, split reproducibility and step timing."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import h5py
import numpy as np
import torch
from grid_core import BASE, windows, allocate
from grid_data import GridDataset, target_weights, sliding_predict
from step_schedule import online_weight, consistency_weight
from grid_validation import pixel_metrics

class GridTests(unittest.TestCase):
    def test_full_coverage_including_exact_multiple_and_partial(self):
        for shape in ((768, 768), (769, 390), (12, 31)):
            counts = np.zeros(shape, np.uint8)
            for i, j, h, w in windows(*shape, 384): counts[i:i+h, j:j+w] += 1
            np.testing.assert_array_equal(counts, np.ones(shape))

    def test_split_seed_classes_and_completeness(self):
        tiles = [dict(key=f't_{i}', is_positive=i%3 == 0) for i in range(36)]
        a, b, _ = allocate(tiles)
        self.assertEqual((len(a), len(b)), (29, 7))
        self.assertFalse(set(a)&set(b)); self.assertEqual(len(set(a+b)), 36)
        self.assertEqual(allocate(tiles), allocate(list(reversed(tiles))))
        self.assertNotEqual(allocate(tiles, seed=8)[:2], (a, b))
        for group in (a, b): self.assertEqual({t['is_positive'] for t in tiles if t['key'] in group}, {False, True})

    def test_padding_never_enters_normalization_or_loss(self):
        with tempfile.TemporaryDirectory(dir=BASE) as temp:
            path = Path(temp)/'test.h5'
            with h5py.File(path, 'w') as f:
                g = f.create_group('tile')
                valid = np.zeros((384, 384), np.uint8); valid[:360, :40] = 1
                g['valid'] = valid
                depth = np.full((384, 384), 1e9, np.float32)
                depth[:360, :40] = np.arange(360)[:, None]
                g['image'] = depth; g['CTX'] = np.full((384, 384), 127, np.float32)
                g['TW'] = np.full((3, 384, 384), 127, np.float32)
                for k in ('k0', 'k1', 'k2'): g[k] = np.zeros((384, 384), np.float32)
            ds = GridDataset(path, ['tile'], full=True)
            x, y = ds[0]; self.assertAlmostEqual(float(x[1][valid > 0].mean()), 0, places=5)
            self.assertTrue((x[[0,1,4,5,6]][:,valid == 0] == 0).all())
            _, weight = target_weights(torch.tensor(x[None]), torch.tensor(y[None]), .02, .08, .1)
            self.assertTrue((weight.numpy()[0,0][valid == 0] == 0).all())
            train = GridDataset(path, ['tile'], training=True)
            for _ in range(10):
                xx, yy = train[0]
                self.assertEqual(xx.shape, (7,256,256)); self.assertTrue(xx[2].any())
                self.assertTrue((yy[:,xx[2] == 0] == 999).all())

    def test_sliding_coverage_and_metric_ties(self):
        class Constant(torch.nn.Module):
            def forward(self, x): return torch.ones_like(x[:,:1])*.25
        x = torch.zeros(1,7,384,384)
        torch.testing.assert_close(sliding_predict(Constant(), x), torch.full((1,1,384,384),.25))
        m = pixel_metrics([np.array([.5,.5])], [np.array([1,0])])
        self.assertEqual(m['AP'], .5)

    def test_builder_reads_missing_values_and_keeps_all_regions(self):
        import netCDF4
        import geopandas as gpd
        from shapely.geometry import LineString
        from prepare_dataset_gem import build
        from make_random_split import split_dataset
        from audit_grid_dataset import audit
        with tempfile.TemporaryDirectory(dir=BASE) as tmp:
            root = Path(tmp)
            nc = root/'terrain.nc'; shp = root/'faults.shp'
            with netCDF4.Dataset(nc, 'w') as ds:
                ds.createDimension('lat', 400); ds.createDimension('lon', 500)
                ds.createVariable('lat', 'f8', ('lat',))[:] = np.arange(400)*.01
                ds.createVariable('lon', 'f8', ('lon',))[:] = np.arange(500)*.01
                z = ds.createVariable('elevation','f4',('lat','lon'),fill_value=-9999.)
                vals = np.arange(400*500,dtype=np.float32).reshape(400,500)
                vals[10:20,10:20] = -9999
                z[:] = vals
            gpd.GeoDataFrame({'SLIP_SENSE':['N']}, geometry=[LineString([(0,0),(4.9,3.9)])], crs=4326).to_file(shp)
            out = root/'built'
            build(SimpleNamespace(nc=str(nc),shp=str(shp),bbox=[0,0,4.99,3.99],out=str(out),
                tile=384,stride=384,edge_pctl=96,maxdist=.35,min_fault_px=40))
            with h5py.File(out/'faults.h5','r') as f:
                self.assertEqual(len(f),4)
                self.assertFalse(f['t_0_0']['valid'][10:20,10:20].any())
            split_dataset(out, .25)
            result = audit(out)
            self.assertEqual(result['valid_pixels'],400*500-100)
            self.assertEqual(result['padded_tiles'],3)
            with self.assertRaises(FileExistsError): split_dataset(out, .25)

    def test_teacher_schedule_uses_completed_updates(self):
        a = SimpleNamespace(onlineCorrection=True, onlineStartStep=800, onlineHoldStep=1000,
            onlineRampStep=1600, onlineWeightStart=.05, onlineWeightMax=.2,
            unimatch=True, unimatchStartStep=800, unimatchRampStep=1600, unimatchWeight=.1)
        self.assertEqual(online_weight(799,a), 0)
        self.assertEqual(online_weight(800,a), .05)
        self.assertAlmostEqual(online_weight(1600,a), .2)
        self.assertEqual(consistency_weight(799,a), 0)
        self.assertAlmostEqual(consistency_weight(1600,a), .1)

if __name__ == '__main__': unittest.main()
