"""Pure grid geometry, deterministic allocation, and experiment path boundaries."""
from pathlib import Path
import random
BASE = Path(__file__).resolve().parent

def local_path(path):
    p = Path(path)
    p = (BASE/p).resolve() if not p.is_absolute() else p.resolve()
    if not p.is_relative_to(BASE):
        raise ValueError(f'Output/dataset must be inside experiment directory: {p}')
    return p

def windows(height, width, tile):
    if min(height, width, tile) <= 0:
        raise ValueError('Grid dimensions must be positive')
    for i in range(0, height, tile):
        for j in range(0, width, tile):
            yield i, j, min(tile, height-i), min(tile, width-j)

def allocate(tiles, val_fraction=.2, seed=7, stratified=True):
    n = len(tiles)
    if n < 2 or not 0 < val_fraction < 1:
        raise ValueError('Need >=2 tiles and 0<val_fraction<1')
    if len({t['key'] for t in tiles}) != n:
        raise ValueError('Duplicate tile keys')
    rng = random.Random(seed)
    total = min(n-1, max(1, int(n*val_fraction+.5)))
    groups = [[t['key'] for t in tiles if bool(t['is_positive']) == label] for label in (False, True)]
    warnings = []
    if not stratified:
        groups = [[t['key'] for t in tiles]]
    groups = [sorted(g) for g in groups if g]
    for g in groups:
        rng.shuffle(g)
    feasible = [k for k in range(total+1) if len(groups) == 2 and
                0 <= k <= len(groups[0]) and 0 <= total-k <= len(groups[1])]
    if len(groups) == 2:
        balanced = [k for k in feasible if 1 <= k < len(groups[0]) and 1 <= total-k < len(groups[1])]
        if not balanced:
            warnings.append('Too few samples to guarantee both classes in both splits')
        k = min(balanced or feasible, key=lambda k: abs(k-total*len(groups[0])/n))
        counts = [k, total-k]
    else:
        counts = [total]
        if stratified:
            warnings.append('Only one class present; stratification is not possible')
    val = sorted(k for g, c in zip(groups, counts) for k in g[:c])
    train = sorted(k for g, c in zip(groups, counts) for k in g[c:])
    return train, val, warnings
