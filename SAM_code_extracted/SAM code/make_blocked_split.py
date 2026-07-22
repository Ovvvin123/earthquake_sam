#!/usr/bin/env python3
"""
Checkerboard / blocked spatial re-split for an existing tile dataset.

WHY (v5 fix): the latitude-band split gave v3 a test set of only 25 tiles,
all at the southern edge — tiny and unrepresentative, so AP comparisons
between versions were extremely noisy. This script re-splits the SAME tiles
(no data rebuild) into train/val/test using 2-D spatial cells, so the test
set samples the whole region, while still guaranteeing ZERO footprint
overlap between tiles of different splits (no leakage; publication-safe).

Method
  1. Read tile centres from manifest.json (written by prepare_dataset_gem.py).
  2. Bin centres into cells of --cell degrees; order cells in a serpentine
     path (S->N, alternating E/W) so each split is a set of contiguous runs.
  3. Assign cell runs to test / val / train by tile-count targets.
  4. Tiles overlap (stride << tile), so remove the minimum number of tiles
     such that no tile of one split overlaps a tile of another split
     (greedy max-degree vertex cover on the cross-split conflict graph;
     ties prefer dropping train tiles to protect val/test).
  5. Verify zero cross-split overlap, then write key_train/val/test.lst
     (originals backed up to *.bands.bak once).

Usage:
  python make_blocked_split.py --dataDir earth_dataset_v3 [--cell 2.0]
  python make_blocked_split.py --dataDir earth_dataset_v3 --dry   # preview only
"""
import argparse, json, os, shutil, sys
from collections import Counter, defaultdict
from project_paths import DATA_DIR, resolve_input

ap = argparse.ArgumentParser()
ap.add_argument('--dataDir', default=str(DATA_DIR))
ap.add_argument('--cell', type=float, default=2.0, help='cell size in degrees')
ap.add_argument('--ftest', type=float, default=0.20, help='target test fraction')
ap.add_argument('--fval', type=float, default=0.12, help='target val fraction')
ap.add_argument('--dry', action='store_true', help='print summary, write nothing')
ap.add_argument('--every', type=int, default=4, help='scatter pattern: every Nth cell -> test')
ap.add_argument('--pattern', default='threeway', choices=['threeway', 'twoway', 'scatter'],
                help="twoway: spatial test/train only; val is a random 10%% of train "
                     "(monitoring-only, cheaper in buffer tiles — recommended here: "
                     "tile footprint 1.6 deg vs region span ~7.7 deg makes every "
                     "extra spatial boundary cost ~20%% of all tiles)")
ap.add_argument('--seed', type=int, default=7)
args = ap.parse_args()
args.dataDir = str(resolve_input(args.dataDir))

man = json.load(open(os.path.join(args.dataDir, 'manifest.json')))
prov, tiles = man['provenance'], man['tiles']
ext = prov['res_deg'] * prov['tile']          # tile footprint in degrees (~1.6)
print(f"{len(tiles)} tiles, footprint {ext:.3f} deg, cell {args.cell} deg")

# --- 1-2: serpentine cell ordering -----------------------------------------
lat0 = min(t['lat_c'] for t in tiles)
lon0 = min(t['lon_c'] for t in tiles)
cell_of = {}
for t in tiles:
    cell_of[t['key']] = (int((t['lat_c']-lat0)/args.cell), int((t['lon_c']-lon0)/args.cell))
cells = sorted(set(cell_of.values()), key=lambda c: (c[0], c[1] if c[0] % 2 == 0 else -c[1]))
ntile = {c: sum(1 for t in tiles if cell_of[t['key']] == c) for c in cells}

# --- 3: assign contiguous serpentine runs ----------------------------------
# Order: [test][train][val][train] — val is embedded inside train (val/train
# leakage costs only train tiles), and never borders test.
N = len(tiles)
fval = 0.0 if args.pattern == 'twoway' else args.fval
b_test = args.ftest * N
b_val0 = b_test + (1 - args.ftest - fval) * N * 0.5
b_val1 = b_val0 + fval * N
split_of_cell, acc = {}, 0
if args.pattern == 'scatter':
    # every Nth cell along the serpentine becomes test -> test patches are
    # spread over the whole region instead of stacked at one edge
    for i, c in enumerate(cells):
        split_of_cell[c] = 'test' if i % args.every == args.every // 2 else 'train'
else:
    for c in cells:
        if acc < b_test:               split_of_cell[c] = 'test'
        elif b_val0 <= acc < b_val1:   split_of_cell[c] = 'val'
        else:                          split_of_cell[c] = 'train'
        acc += ntile[c]
split = {t['key']: split_of_cell[cell_of[t['key']]] for t in tiles}

# --- 4: remove minimal conflict set -----------------------------------------
info = {t['key']: t for t in tiles}
keys = [t['key'] for t in tiles]
conf = defaultdict(set)
for i, a in enumerate(keys):
    ta = info[a]
    for b in keys[i+1:]:
        tb = info[b]
        if split[a] != split[b] and abs(ta['lat_c']-tb['lat_c']) < ext and abs(ta['lon_c']-tb['lon_c']) < ext:
            conf[a].add(b); conf[b].add(a)
dropped = set()
rank = {'train': 0, 'val': 1, 'test': 2}     # train is least precious
while True:
    live = {k: {b for b in v if b not in dropped} for k, v in conf.items()
            if k not in dropped and any(b not in dropped for b in v)}
    if not live:
        break
    # the buffer cost is paid by the least precious split first: among
    # conflicted tiles, drop from the lowest rank present, highest degree
    lo = min(rank[split[k]] for k in live)
    k = max((k for k in live if rank[split[k]] == lo), key=lambda k: len(live[k]))
    dropped.add(k)

# --- 5: verify + report ------------------------------------------------------
kept = [k for k in keys if k not in dropped]
for i, a in enumerate(kept):
    for b in kept[i+1:]:
        if split[a] != split[b]:
            assert not (abs(info[a]['lat_c']-info[b]['lat_c']) < ext and
                        abs(info[a]['lon_c']-info[b]['lon_c']) < ext), 'leak!'
if args.pattern in ('twoway', 'scatter'):
    # carve a random val out of train: used ONLY for loss monitoring /
    # early stopping, NOT for reported metrics (it shares geography with
    # train). The reported metrics come from the spatially disjoint test.
    import random
    rng = random.Random(args.seed)
    tr = [k for k in kept if split[k] == 'train']
    for k in rng.sample(tr, max(1, int(len(tr) * 0.10))):
        split[k] = 'val'
cnt = Counter(split[k] for k in kept)
pos = Counter(split[k] for k in kept if info[k]['is_positive'])
print(f"dropped {len(dropped)}/{len(keys)} buffer tiles; kept {len(kept)}")
for s in ('train', 'val', 'test'):
    print(f"  {s:5s}: {cnt[s]:4d} tiles ({pos[s]} positive)")

if args.dry:
    sys.exit(0)
for s in ('train', 'val', 'test'):
    p = os.path.join(args.dataDir, f'key_{s}.lst')
    if os.path.exists(p) and not os.path.exists(p + '.bands.bak'):
        shutil.copy(p, p + '.bands.bak')
    with open(p, 'w') as f:
        f.write('\n'.join(k for k in kept if split[k] == s) + '\n')
json.dump({'cell_deg': args.cell, 'dropped': sorted(dropped),
           'counts': {s: cnt[s] for s in ('train','val','test')},
           'positives': {s: pos[s] for s in ('train','val','test')}},
          open(os.path.join(args.dataDir, 'blocked_split_info.json'), 'w'), indent=1)
print("wrote key_train/val/test.lst (+ blocked_split_info.json); originals -> *.bands.bak")
