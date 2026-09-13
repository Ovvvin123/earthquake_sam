#!/usr/bin/env python3
"""
build_multiregion.py — v6: build the GLOBAL multi-region training set.

For each region in regions_global.json this script calls
prepare_dataset_gem.py (one bbox at a time, slicing the global GEBCO file),
then merges everything into ONE dataset directory:

    earth_dataset_global/
        faults.h5        keys prefixed region__  (e.g. japan__t_12_640)
        key_train.lst    all selected tiles from role=train regions
        key_val.lst      all selected tiles from the role=val region(s)
        manifest.json

Split design (leakage-free BY CONSTRUCTION, no buffer tiles wasted):
  * train / val are whole, geographically disjoint regions
  * TEST is the held-out California offshore region — evaluate with the
    existing earth_dataset_v3 key_test.lst / key_test_unseen.lst, so v6
    numbers are directly comparable to v5.

Per-region cap (--cap, default 150): positives first, then negatives up to
~45%% of the selection — keeps regions balanced and Mac training time sane.

Resumable: a region whose regions_v6/<name>/manifest.json exists is not
rebuilt (delete that folder to force).

Usage (after downloading the global GEBCO nc and cloning GEM):
    python build_multiregion.py --nc GEBCO_2026.nc
    python build_multiregion.py --nc GEBCO_2026.nc --cap 300   # bigger set
"""
import os, sys, json, argparse, subprocess
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--nc', required=True, help='global GEBCO netCDF file')
ap.add_argument('--regions', default='regions_global.json')
ap.add_argument('--gem_shp', default='gem-global-active-faults/shapefile/gem_active_faults_harmonized.shp')
ap.add_argument('--qfaults_shp', default='../../Qfaults_GIS/SHP/Qfaults_US_Database.shp')
ap.add_argument('--workdir', default='regions_v6')
ap.add_argument('--out', default='earth_dataset_global')
ap.add_argument('--cap', type=int, default=150, help='max tiles kept per region')
ap.add_argument('--neg_frac', type=float, default=0.45, help='max negative share within the cap')
ap.add_argument('--stride', type=int, default=96, help='tile stride px (96 = less overlap than CA v3)')
ap.add_argument('--seed', type=int, default=11)
args = ap.parse_args()

cfg = json.load(open(args.regions))['regions']
shp_of = {'gem': args.gem_shp, 'qfaults': args.qfaults_shp}
for kind, p in shp_of.items():
    if any(r['catalog'] == kind for r in cfg) and not os.path.exists(p):
        sys.exit(f"missing {kind} shapefile: {p}\n"
                 f"  gem: git clone https://github.com/GEMScienceTools/gem-global-active-faults.git")
if not os.path.exists(args.nc):
    sys.exit(f"missing GEBCO file: {args.nc}")

# ---------- 1. build each region ----------
os.makedirs(args.workdir, exist_ok=True)
for r in cfg:
    rd = os.path.join(args.workdir, r['name'])
    if os.path.exists(os.path.join(rd, 'manifest.json')):
        print(f"[skip] {r['name']} already built"); continue
    cmd = [sys.executable, 'prepare_dataset_gem.py',
           '--nc', args.nc, '--shp', shp_of[r['catalog']],
           '--bbox', *[str(b) for b in r['bbox']],
           '--out', rd, '--stride', str(args.stride), '--no_split']
    print(f"[build] {r['name']}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

# ---------- 2. select tiles per region ----------
rng = np.random.default_rng(args.seed)
selection = {}   # name -> list of keys
summary = []
for r in cfg:
    rd = os.path.join(args.workdir, r['name'])
    tiles = json.load(open(os.path.join(rd, 'manifest.json')))['tiles']
    pos = [t['key'] for t in tiles if t['is_positive']]
    neg = [t['key'] for t in tiles if not t['is_positive']]
    rng.shuffle(pos); rng.shuffle(neg)
    n_neg = min(len(neg), int(args.cap * args.neg_frac))
    n_pos = min(len(pos), args.cap - n_neg)
    n_neg = min(len(neg), args.cap - n_pos)          # backfill if few positives
    selection[r['name']] = pos[:n_pos] + neg[:n_neg]
    summary.append((r['name'], r['role'], len(tiles), n_pos, n_neg))
    print(f"[select] {r['name']:14s} ({r['role']:5s}) built {len(tiles):4d} -> keep {n_pos}+{n_neg}neg")

# ---------- 3. merge into one h5 + key lists ----------
import h5py
os.makedirs(args.out, exist_ok=True)
H5 = os.path.join(args.out, 'faults.h5')
man = []
with h5py.File(H5, 'w') as o:
    for r in cfg:
        rd = os.path.join(args.workdir, r['name'])
        info = {t['key']: t for t in json.load(open(os.path.join(rd, 'manifest.json')))['tiles']}
        with h5py.File(os.path.join(rd, 'faults.h5'), 'r') as f:
            for k in selection[r['name']]:
                nk = f"{r['name']}__{k}"
                f.copy(k, o, name=nk)
                t = dict(info[k]); t['key'] = nk; t['region'] = r['name']; t['split'] = r['role']
                man.append(t)
print(f"[merge] {len(man)} tiles -> {H5}")

for role, fn in (('train', 'key_train.lst'), ('val', 'key_val.lst')):
    ks = [t['key'] for t in man if t['split'] == role]
    rng.shuffle(ks)
    open(os.path.join(args.out, fn), 'w').write('\n'.join(ks) + '\n')
    print(f"[write] {fn}: {len(ks)} tiles")
open(os.path.join(args.out, 'key.lst'), 'w').write('\n'.join(t['key'] for t in man) + '\n')

json.dump({'provenance': {'nc': args.nc, 'regions': cfg, 'cap': args.cap,
                          'stride': args.stride, 'note': 'TEST = California offshore, use earth_dataset_v3 key_test.lst'},
           'tiles': man}, open(os.path.join(args.out, 'manifest.json'), 'w'))
print('\nSummary:')
for name, role, built, npos, nneg in summary:
    print(f"  {name:14s} {role:5s}  built {built:4d}  kept {npos}+{nneg}neg")
print(f"\nDone. Train with:\n"
      f"  PYTORCH_ENABLE_MPS_FALLBACK=1 python trainV10.py --job train --model mars_sam --modelType mars_sam \\\n"
      f"    --dataset {args.out}/key_train.lst --resDir train_v6_global/ \\\n"
      f"    --mn 256 --batchsize 2 --stepPerMiniBatch 4 --epochs 40 \\\n"
      f"    --lr0 1e-4 --d0 0.02 --dw 0.08 --alpha 0.1 --dropout 0 \\\n"
      f"    --lossFunc FocalTversky --k0Only --pu 0.4")
