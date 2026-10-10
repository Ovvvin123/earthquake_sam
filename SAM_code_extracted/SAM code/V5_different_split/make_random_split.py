"""Assign saved tiles to train/val once; never rebuild or resample terrain."""
import argparse
import json
import h5py
from grid_core import local_path, allocate

def split_dataset(data_dir, val_fraction=.2, seed=7, stratified=True):
    d = local_path(data_dir)
    for name in ('key_train.lst', 'key_val.lst', 'split_info.json'):
        if (d/name).exists():
            raise FileExistsError(f'Refusing to replace established split: {d/name}')
    man = json.loads((d/'manifest.json').read_text(encoding='utf-8'))
    train, val, warnings = allocate(man['tiles'], val_fraction, seed, stratified)
    sets = dict(train=set(train), val=set(val))
    with h5py.File(d/'faults.h5', 'r+') as f:
        if set(f) != sets['train'] | sets['val']:
            raise ValueError('H5 keys differ from manifest')
        for t in man['tiles']:
            t['split'] = 'train' if t['key'] in sets['train'] else 'val'
            f[t['key']].attrs['split'] = t['split']
    info = dict(seed=seed, val_fraction=val_fraction, stratified=stratified,
                counts=dict(train=len(train), val=len(val)), warnings=warnings,
                evaluation_scope='random within-region validation; no held-out test')
    man['provenance']['counts'] = info['counts']
    man['provenance']['split'] = info
    for name, keys in (('train', train), ('val', val)):
        (d/f'key_{name}.lst').write_text(''.join(k+'\n' for k in keys), encoding='utf-8')
    (d/'manifest.json').write_text(json.dumps(man, indent=2), encoding='utf-8')
    (d/'split_info.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(json.dumps(info, indent=2))
    return info

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataDir', default='datasets/ca_stride384')
    p.add_argument('--valFraction', type=float, default=.2)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--random', action='store_true', help='unstratified random split')
    a = p.parse_args()
    split_dataset(a.dataDir, a.valFraction, a.seed, not a.random)
