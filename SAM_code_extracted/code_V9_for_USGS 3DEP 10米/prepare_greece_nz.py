#!/usr/bin/env python3
"""Prepare Greece + New Zealand GEM data using the existing GEBCO channel builder.

No arguments: prepare filtered labels and a readiness report only.
Build: --greece-nc FILE --nz-nc FILE, or --nc GLOBAL_FILE.
Default: 384px tiles at stride 64; seeded 80/20 geographic-cell split within
each region; land and sea treated alike; California remains external test data.
Excluded Bird/fold traces cause intersecting tiles (plus 0.02deg buffer) to
be omitted, not relabelled as background. Default keeps only tiles with >=40
catalog fault pixels. This reduces empty-catalog negatives but DOES NOT prove
remaining background pixels are fault-free. No survey-coverage mask is inferred.
--negative-policy legacy retains the old builder's sampled background tiles.
--split spatial assigns whole geographic cells then drops boundary tiles.
Existing training files, model code and California splits are never edited.
"""
from pathlib import Path
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import random
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REGIONS = {
    'greece_aegean': [19, 34, 29, 42],
    'new_zealand': [166, -47, 179, -34],
}
DEFAULT_DATA_DIR = HERE/'earth_dataset_greece_nz_stride64_v2'


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def footprint(tile):
    e = tile['extent_deg'] / 2
    return (tile['lon_c']-e, tile['lat_c']-e,
            tile['lon_c']+e, tile['lat_c']+e)


def overlaps(a, b):
    x, y, X, Y = footprint(a)
    u, v, U, V = footprint(b)
    return min(X, U)-max(x, u) > 1e-8 and min(Y, V)-max(y, v) > 1e-8


def overlap_audit(tiles):
    tr = [t for t in tiles if t['split'] == 'train']
    va = [t for t in tiles if t['split'] == 'val']
    pairs = [(a['key'], b['key']) for a in tr for b in va if overlaps(a, b)]
    return {'train_val_overlap_pairs': len(pairs), 'examples': pairs[:10]}


def assign_splits(tiles, fraction=.2, seed=42, mode='random', cell_deg=4.):
    if not 0 < fraction < 1:
        raise ValueError('val fraction must lie strictly between 0 and 1')
    if len({t['key'] for t in tiles}) != len(tiles):
        raise ValueError('duplicate tile keys')
    rng = random.Random(seed)
    kept, dropped = [], []
    for region in sorted({t['region'] for t in tiles}):
        candidates = sorted([dict(t) for t in tiles if t['region'] == region],
                            key=lambda t: t['key'])
        if mode == 'random':
            if len(candidates) < 2:
                raise ValueError(f'{region}: need at least two tiles')
            rng.shuffle(candidates)
            nval = max(1, min(len(candidates)-1, round(len(candidates)*fraction)))
            for i, t in enumerate(candidates):
                t['split'] = 'val' if i < nval else 'train'
            kept.extend(candidates)
        elif mode == 'spatial':
            import math
            if cell_deg <= max(t['extent_deg'] for t in candidates):
                raise ValueError('spatial cell must be larger than a tile')
            groups = defaultdict(list)
            for t in candidates:
                x, y, X, Y = footprint(t)
                c = (math.floor(x/cell_deg), math.floor(y/cell_deg))
                d = (math.floor((X-1e-8)/cell_deg), math.floor((Y-1e-8)/cell_deg))
                if c != d:
                    dropped.append({'key': t['key'], 'reason': 'crosses_spatial_cell'})
                else:
                    groups[c].append(t)
            cells = sorted(groups)
            if len(cells) < 2:
                raise ValueError(f'{region}: fewer than two usable spatial cells')
            rng.shuffle(cells)
            # Balance tile counts, not cell counts: one dense cell can otherwise
            # put most of a region into validation. Whole cells stay indivisible.
            reachable = {0: ()}
            for c in cells:
                for count, selected in list(reachable.items()):
                    reachable.setdefault(count+len(groups[c]), selected+(c,))
            total = sum(map(len, groups.values()))
            nval = min((n for n in reachable if 0 < n < total),
                       key=lambda n: (abs(n-total*fraction), n))
            val_cells = set(reachable[nval])
            for c in cells:
                for t in groups[c]:
                    t['split'] = 'val' if c in val_cells else 'train'
                    t['spatial_cell'] = list(c)
                    kept.append(t)
        else:
            raise ValueError(f'unknown split: {mode}')
    return kept, dropped


def validate_nc(path, bbox):
    import netCDF4
    import numpy as np
    with netCDF4.Dataset(path) as f:
        for name in ['lat', 'lon', 'elevation']:
            if name not in f.variables:
                raise ValueError(f'{path}: missing variable {name}')
        lat, lon = np.asarray(f['lat'][:]), np.asarray(f['lon'][:])
        expected = 1/240
        for name, arr in [('lat', lat), ('lon', lon)]:
            if len(arr) < 2 or not np.allclose(np.diff(arr), expected, atol=1e-8, rtol=0):
                raise ValueError(f'{path}: {name} must be ascending at 15 arcseconds')
        x, y, X, Y = bbox
        if (lon[0] > x+expected/2+1e-8 or lon[-1] < X-expected/2-1e-8 or
                lat[0] > y+expected/2+1e-8 or lat[-1] < Y-expected/2-1e-8):
            raise ValueError(f'{path}: does not cover requested bbox {bbox}')
        if f['elevation'].dimensions != ('lat', 'lon'):
            raise ValueError(f'{path}: expected elevation dimensions (lat, lon)')
        return {'path': str(path.resolve()), 'size_bytes': path.stat().st_size,
                'resolution_deg': expected, 'bbox': bbox}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--nc', type=Path, help='global GEBCO netCDF covering both regions')
    ap.add_argument('--greece-nc', type=Path)
    ap.add_argument('--nz-nc', type=Path)
    ap.add_argument('--gem', type=Path, default=ROOT/'GEBCO_Global/GEM_label/gem-global-active-faults-master/geojson/gem_active_faults_harmonized.geojson')
    ap.add_argument('--out', type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument('--labels-only', action='store_true')
    ap.add_argument('--split', choices=['random', 'spatial'], default='spatial')
    ap.add_argument('--val-fraction', type=float, default=.2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--tile', type=int, default=384)
    ap.add_argument('--stride', type=int, default=64)
    ap.add_argument('--cell-deg', type=float, default=4.)
    ap.add_argument('--excluded-buffer-deg', type=float, default=.02)
    ap.add_argument('--negative-policy', choices=['positive-tiles-only', 'legacy'], default='positive-tiles-only')
    args = ap.parse_args(argv)
    if args.tile < 344 or args.stride < 1:
        ap.error('tile must be >=344 for 256px crops at 0.75x augmentation; stride must be positive')
    if not 0 < args.val_fraction < 1 or args.excluded_buffer_deg < 0:
        ap.error('invalid fraction or exclusion buffer')
    if args.split == 'random' and args.stride < args.tile:
        ap.error('random split requires stride >= tile; use --split spatial for overlapping tiles')
    out = args.out.resolve()
    if (out/'faults.h5').exists() or (out/'build_started.json').exists():
        ap.error('output already contains a build; choose a fresh --out directory')
    out.mkdir(parents=True, exist_ok=True)
    import geopandas as gpd
    from shapely.geometry import box
    from shapely.ops import unary_union
    gem = gpd.read_file(args.gem).to_crs(4326)
    required = {'catalog_name', 'slip_type'}
    if not required <= set(gem.columns):
        ap.error('GEM must retain catalog_name and slip_type: use the full GeoJSON')
    report = {'gem': str(args.gem.resolve()), 'gem_sha256': hashlib.sha256(args.gem.read_bytes()).hexdigest(),
              'land_sea_filter': False, 'regions': {}, 'california_role': 'external_test_only',
              'split': args.split, 'val_fraction': args.val_fraction, 'seed': args.seed,
              'tile': args.tile, 'stride': args.stride, 'cell_deg': args.cell_deg,
              'negative_policy': args.negative_policy,
              'spatial_assignment': 'whole cells; closest attainable validation tile count per region; seeded tie order' if args.split=='spatial' else None,
              'excluded_buffer_deg': args.excluded_buffer_deg,
              'supervision': 'catalog-based weak labels; no surveyed-coverage or reliable-negative mask available',
              'validation_limit': 'random neighboring tiles are spatially correlated even without footprint overlap' if args.split=='random' else 'spatial blocks within two regions; not a held-out-country evaluation',
              'evaluation_reference': str(HERE/'earth_dataset_v3/key_test.lst'),
              'unseen_evaluation_reference': str(HERE/'earth_dataset_v3/key_test_unseen.lst')}
    excluded = {}
    for name, bbox in REGIONS.items():
        data = gem[gem.intersects(box(*bbox))].copy()
        bad = data['catalog_name'].fillna('').eq('Bird 2003') | data['slip_type'].fillna('').isin(['Anticline','Syncline'])
        kept, removed = data[~bad], data[bad]
        if kept.empty:
            raise ValueError(f'{name}: no retained fault features')
        labelpath = out/f'{name}_labels.geojson'
        kept.to_file(labelpath, driver='GeoJSON')
        # Preserve every excluded record for audit, even when no features exist.
        (out/f'{name}_excluded.geojson').write_text(removed.to_json(), encoding='utf-8')
        excluded[name] = unary_union(list(removed.geometry)).buffer(args.excluded_buffer_deg)
        report['regions'][name] = {'bbox': bbox, 'retained_features': len(kept),
                                   'excluded_features': len(removed), 'labels': str(labelpath),
                                   'catalogs': dict(Counter(kept['catalog_name'])),
                                   'quality_nonmissing': {f:int(kept[f].notna().sum()) if f in kept else 0 for f in ['activity_confidence','exposure_quality','epistemic_quality']}}
    paths = {'greece_aegean': args.greece_nc or args.nc,
             'new_zealand': args.nz_nc or args.nc}
    missing = [name for name,p in paths.items() if p is None or not p.is_file()]
    if args.labels_only or missing:
        report['status'] = 'labels_prepared_dataset_not_built'
        report['missing_gebco_regions'] = missing
        dump(out/'preparation_report.json', report)
        print(json.dumps(report,ensure_ascii=False,indent=2))
        return 0 if args.labels_only or all(p is None for p in paths.values()) else 2
    # Validate both inputs before starting any costly build.
    for name,path in paths.items():
        report['regions'][name]['gebco'] = validate_nc(path, REGIONS[name])
    dump(out/'build_started.json', report)
    alltiles, dropped = [], []
    for name,path in paths.items():
        work = out/'region_builds'/name
        cmd = [sys.executable, str(HERE/'prepare_dataset_gem.py'), '--nc', str(path.resolve()),
               '--shp', report['regions'][name]['labels'], '--bbox', *map(str,REGIONS[name]),
               '--out', str(work), '--tile', str(args.tile), '--stride', str(args.stride),
               '--seed', str(args.seed), '--no_split']
        subprocess.run(cmd, check=True, cwd=HERE)
        man = json.loads((work/'manifest.json').read_text())
        extent = man['provenance']['res_deg'] * man['provenance']['tile']
        for tile in man['tiles']:
            t = dict(tile, source_key=tile['key'], key=f'{name}__{tile["key"]}',
                     region=name, extent_deg=extent)
            if box(*footprint(t)).intersects(excluded[name]):
                dropped.append({'key':t['key'], 'reason':'intersects_excluded_trace_buffer'})
            elif args.negative_policy=='positive-tiles-only' and not t['is_positive']:
                dropped.append({'key':t['key'], 'reason':'insufficient_catalog_fault_pixels'})
            else:
                alltiles.append(t)
    tiles, boundary_drops = assign_splits(alltiles,args.val_fraction,args.seed,args.split,args.cell_deg)
    dropped += boundary_drops
    audit = overlap_audit(tiles)
    if audit['train_val_overlap_pairs']:
        raise ValueError(f'cross-split overlap detected: {audit}')
    import h5py
    # No California tiles are copied into this dataset.
    with h5py.File(out/'faults.h5','w') as dest:
        for name in REGIONS:
            with h5py.File(out/'region_builds'/name/'faults.h5','r') as source:
                for t in [x for x in tiles if x['region']==name]:
                    source.copy(t['source_key'], dest, name=t['key'])
                    dest[t['key']].attrs['split'] = t['split']
                    dest[t['key']].attrs['region'] = name
    rng = random.Random(args.seed)
    for split in ['train','val']:
        keys = sorted(t['key'] for t in tiles if t['split']==split)
        rng.shuffle(keys)
        (out/f'key_{split}.lst').write_text('\n'.join(keys)+'\n',encoding='utf-8')
    (out/'key.lst').write_text('\n'.join(t['key'] for t in tiles)+'\n',encoding='utf-8')
    report.update(status='dataset_built', audit=audit,
                  counts=dict(Counter(t['split'] for t in tiles)),
                  region_split_counts=dict(Counter(t['region']+'/'+t['split'] for t in tiles)),
                  dropped=dropped)
    dump(out/'manifest.json', {'provenance':report,'tiles':tiles})
    dump(out/'preparation_report.json', report)
    (out/'build_started.json').unlink()
    print(json.dumps({k:report[k] for k in ['status','counts','region_split_counts','audit']},indent=2))


if __name__ == '__main__':
    sys.exit(main())
