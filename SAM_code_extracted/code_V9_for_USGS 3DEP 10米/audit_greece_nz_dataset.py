"""Check the real built dataset against source GEBCO and the training reader."""
import json
import argparse
from pathlib import Path
from collections import Counter
import numpy as np
import h5py
import netCDF4
from prepare_greece_nz import DEFAULT_DATA_DIR, overlap_audit
from tool import Generator_new

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data-dir',type=Path,default=DEFAULT_DATA_DIR)
data=parser.parse_args().data_dir.resolve()
man=json.loads((data/'manifest.json').read_text())
rows=man['tiles'];prov=man['provenance']
checks={'tile_count':len(rows),'raw_tiles_exact_match':0,'all_stored_arrays_finite':True,
        'all_expected_shapes':True,'list_attributes_manifest_agree':True,
        'fixed_validation_repeatable':True,'fixed_validation_inputs_finite':True,
        'augmented_inputs_finite':True,'regions':dict(Counter(t['region'] for t in rows)),
        'dropped_reasons':dict(Counter(t['reason'] for t in prov['dropped'])),
        'overlap':overlap_audit(rows)}
with h5py.File(data/'faults.h5','r') as f:
    for region,info in prov['regions'].items():
        with netCDF4.Dataset(info['gebco']['path']) as nc:
            lat=np.asarray(nc['lat'][:]);lon=np.asarray(nc['lon'][:]);b=info['bbox']
            i0=int(np.searchsorted(lat,b[1]));j0=int(np.searchsorted(lon,b[0]))
            for t in [t for t in rows if t['region']==region]:
                i,j=map(int,t['source_key'].split('_')[1:]);T=prov['tile']
                z=np.asarray(nc['elevation'][i0+i:i0+i+T,j0+j:j0+j+T],dtype='float32')
                g=f[t['key']]
                checks['raw_tiles_exact_match']+=int(np.array_equal(z,g['image'][:]))
                for name in ['image','CTX','TW','k0','k1','k2']:
                    arr=g[name][:]
                    checks['all_stored_arrays_finite'] &= bool(np.isfinite(arr).all())
                    checks['all_expected_shapes'] &= arr.shape==((3,T,T) if name=='TW' else (T,T))
                checks['list_attributes_manifest_agree'] &= g.attrs['split']==t['split']
    for split in ['train','val']:
        keys=(data/f'key_{split}.lst').read_text().split()
        checks['list_attributes_manifest_agree'] &= set(keys)=={t['key'] for t in rows if t['split']==split}
    checks['list_attributes_manifest_agree'] &= set(f)=={t['key'] for t in rows}
keys=(data/'key_val.lst').read_text().split()
reader=Generator_new(str(data/'faults.h5'),keys,m=256,n=256,training=False,fixed_crop=True)
for i in range(len(keys)):
    a,b=reader[i],reader[i]
    checks['fixed_validation_repeatable'] &= all(np.array_equal(x,y) for x,y in zip(a,b))
    checks['fixed_validation_inputs_finite'] &= a[0].shape==(7,256,256) and all(np.isfinite(x).all() for x in a)
    reader.dataD.clear()  # Audit each tile once without retaining a large dataset in RAM.
keys=(data/'key_train.lst').read_text().split()
reader=Generator_new(str(data/'faults.h5'),keys,m=256,n=256,training=True)
for i in range(len(keys)):
    a=reader[i]
    checks['augmented_inputs_finite'] &= a[0].shape==(7,256,256) and all(np.isfinite(x).all() for x in a)
    reader.dataD.clear()
checks={k:bool(v) if isinstance(v,np.bool_) else v for k,v in checks.items()}
checks['passed']=(checks['raw_tiles_exact_match']==len(rows) and
                  all(checks[k] for k in ['all_stored_arrays_finite','all_expected_shapes',
                    'list_attributes_manifest_agree','fixed_validation_repeatable',
                    'fixed_validation_inputs_finite','augmented_inputs_finite']) and
                  checks['overlap']['train_val_overlap_pairs']==0)
(data/'data_verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print(json.dumps(checks,indent=2))
if not checks['passed']:
    raise SystemExit('dataset validation failed')
