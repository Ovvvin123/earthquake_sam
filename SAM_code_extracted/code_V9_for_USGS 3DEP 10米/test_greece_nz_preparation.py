"""Regression tests for splitting/leakage and a synthetic end-to-end build."""
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys
from unittest.mock import patch

import prepare_greece_nz as prep


def tile(key, x, y, region='greece_aegean'):
    return dict(key=key,lon_c=x,lat_c=y,extent_deg=1.6,region=region,is_positive=True)


class PreparationTests(unittest.TestCase):
    def test_random_reproducible_disjoint_and_both_regions(self):
        tiles=[tile(f'{r}_{i}',offset+i*1.6,0,r) for r,offset in
               [('greece_aegean',20),('new_zealand',166)] for i in range(10)]
        a,_=prep.assign_splits(tiles)
        b,_=prep.assign_splits(list(reversed(tiles)))
        self.assertEqual(a,b)
        self.assertEqual(prep.overlap_audit(a)['train_val_overlap_pairs'],0)
        for region in ['greece_aegean','new_zealand']:
            self.assertEqual(sum(t['split']=='val' for t in a if t['region']==region),2)

    def test_overlap_is_detected_and_touching_is_not_overlap(self):
        a=tile('a',0,0);a['split']='train'
        b=tile('b',.3,0);b['split']='val'
        self.assertEqual(prep.overlap_audit([a,b])['train_val_overlap_pairs'],1)
        b['lon_c']=1.6
        self.assertEqual(prep.overlap_audit([a,b])['train_val_overlap_pairs'],0)

    def test_spatial_boundary_drops_and_cross_split_separation(self):
        rows=[tile(str(i),x,.9) for i,x in enumerate([.9,2,3.9,5,7,8.9,9.9])]
        assigned,dropped=prep.assign_splits(rows,mode='spatial',cell_deg=4)
        self.assertTrue(dropped)
        self.assertEqual({t['split'] for t in assigned},{'train','val'})
        self.assertEqual(prep.overlap_audit(assigned)['train_val_overlap_pairs'],0)

    def test_overlapping_random_stride_is_rejected(self):
        with self.assertRaises(SystemExit):
            prep.main(['--split','random','--stride','64','--labels-only'])

    def test_spatial_split_balances_unequal_cell_populations(self):
        rows=[tile(f'{i}_{j}',i*4+1,1) for i,n in enumerate([81,56,9,9]) for j in range(n)]
        assigned,_=prep.assign_splits(rows,mode='spatial',cell_deg=4)
        self.assertEqual(sum(t['split']=='val' for t in assigned),18)
        self.assertEqual(prep.overlap_audit(assigned)['train_val_overlap_pairs'],0)

    def test_stride64_spatial_groups_preserve_dense_training_without_leakage(self):
        rows=[tile(f'{region}_{i}_{j}',offset+.8+j*64/240,.8+i*64/240,region)
              for region,offset in [('greece_aegean',20),('new_zealand',168)]
              for i in range(24) for j in range(24)]
        assigned,dropped=prep.assign_splits(rows,mode='spatial',cell_deg=4)
        repeated,_=prep.assign_splits(list(reversed(rows)),mode='spatial',cell_deg=4)
        self.assertEqual(assigned,repeated)
        self.assertTrue(dropped)
        self.assertEqual(prep.overlap_audit(assigned)['train_val_overlap_pairs'],0)
        for region in ['greece_aegean','new_zealand']:
            subset=[t for t in assigned if t['region']==region]
            self.assertEqual({t['split'] for t in subset},{'train','val'})
            train=[t for t in subset if t['split']=='train']
            self.assertTrue(any(prep.overlaps(a,b) for i,a in enumerate(train) for b in train[i+1:]))

    def test_actual_california_overlap_regression(self):
        data=prep.HERE/'earth_dataset_v3'
        man=json.loads((data/'manifest.json').read_text())
        extent=man['provenance']['tile']*man['provenance']['res_deg']
        groups={s:set((data/f'key_{s}.lst').read_text().split()) for s in ['train','val','test']}
        rows=[dict(t,extent_deg=extent,split=s) for t in man['tiles']
              for s,keys in groups.items() if t['key'] in keys]
        self.assertEqual(prep.overlap_audit(rows)['train_val_overlap_pairs'],769)
        for s in ['train','val']:
            self.assertFalse(any(prep.overlaps(a,b) for a in rows if a['split']==s
                                 for b in rows if b['split']=='test'))

    def test_synthetic_end_to_end_and_input_bounds(self):
        import geopandas as gpd
        import netCDF4
        import numpy as np
        import h5py
        from shapely.geometry import LineString
        regions={'greece_aegean':[19,34,24,39],'new_zealand':[166,-47,171,-42]}
        with tempfile.TemporaryDirectory(prefix='greece_nz_fixture_') as temp:
            root=Path(temp);features=[];paths={}
            for name,(x,y,X,Y) in regions.items():
                path=root/f'{name}.nc';paths[name]=path
                with netCDF4.Dataset(path,'w') as ds:
                    ds.createDimension('lat',1200);ds.createDimension('lon',1200)
                    ds.createVariable('lat','f8',('lat',))[:]=y+(np.arange(1200)+.5)/240
                    ds.createVariable('lon','f8',('lon',))[:]=x+(np.arange(1200)+.5)/240
                    z=ds.createVariable('elevation','i2',('lat','lon'),zlib=True)
                    z[:]=(np.arange(1200)[:,None]-600+np.zeros((1,1200))).astype('int16')
                for dy in [.7,2.3]:
                    features.append(dict(catalog_name='fixture_survey',slip_type='Normal',
                                         geometry=LineString([(x+.1,y+dy),(X-.1,y+dy)])))
                features.append(dict(catalog_name='Bird 2003',slip_type='Subduction_Thrust',
                                     geometry=LineString([(x+.2,y+.1),(x+.2,Y-.1)])))
                features.append(dict(catalog_name='fixture_survey',slip_type='Anticline',
                                     geometry=LineString([(x+.25,y+.1),(x+.25,Y-.1)])))
            gem=root/'gem.geojson';gpd.GeoDataFrame(features,crs=4326).to_file(gem,driver='GeoJSON')
            with self.assertRaises(ValueError):
                prep.validate_nc(paths['greece_aegean'],regions['new_zealand'])
            out=root/'output'
            with patch.object(prep,'REGIONS',regions):
                prep.main(['--gem',str(gem),'--out',str(out),'--greece-nc',str(paths['greece_aegean']),
                           '--nz-nc',str(paths['new_zealand']),'--split','random','--stride','384'])
            report=json.loads((out/'preparation_report.json').read_text())
            self.assertEqual(report['status'],'dataset_built')
            self.assertEqual(report['audit']['train_val_overlap_pairs'],0)
            self.assertTrue(any(t['reason']=='intersects_excluded_trace_buffer' for t in report['dropped']))
            self.assertTrue(any(t['reason']=='insufficient_catalog_fault_pixels' for t in report['dropped']))
            man=json.loads((out/'manifest.json').read_text())
            with h5py.File(out/'faults.h5','r') as f:
                for s in ['train','val']:
                    keys=(out/f'key_{s}.lst').read_text().split()
                    self.assertEqual({k.split('__')[0] for k in keys},set(regions))
                    for k in keys:
                        self.assertEqual(f[k].attrs['split'],s)
                        self.assertEqual(f[k]['TW'].shape,(3,384,384))
                        self.assertTrue(np.isfinite(f[k]['k0'][:]).all())
                        self.assertTrue((f[k]['k0'][:]==0).any())
                self.assertEqual(len(f),len(man['tiles']))
            # Real reader: fixed validation is deterministic and has 7 inputs.
            from tool import Generator_new
            keys=(out/'key_val.lst').read_text().split()
            reader=Generator_new(str(out/'faults.h5'),keys,m=256,n=256,training=False,fixed_crop=True)
            a=reader[0];b=reader[0]
            self.assertEqual(a[0].shape,(7,256,256))
            np.testing.assert_array_equal(a[0],b[0])
            self.assertTrue(np.isfinite(a[0]).all())
            preview=subprocess.run([sys.executable,str(prep.HERE/'run_greece_nz_training.py'),
                                    '--data-dir',str(out),'--res-dir',str(root/'training')],
                                   capture_output=True,text=True,check=True)
            self.assertIn('--fixedVal',preview.stdout)
            self.assertFalse((root/'training').exists())
            # Refuse stale lists rather than train on a different split silently.
            (out/'key_val.lst').write_text('not_a_real_key\n')
            failed=subprocess.run([sys.executable,str(prep.HERE/'run_greece_nz_training.py'),
                                   '--data-dir',str(out)],capture_output=True,text=True)
            self.assertNotEqual(failed.returncode,0)


if __name__=='__main__':
    unittest.main(verbosity=2)
