"""Verify grid coverage, split membership, masks and metadata; render split map."""
import argparse
import json
import h5py
import numpy as np
from grid_core import local_path


def audit(data_dir):
    d = local_path(data_dir)
    man = json.loads((d/'manifest.json').read_text())
    shape = man['provenance']['shape']
    coverage = np.zeros(shape, np.uint8)
    keysets = {s: (d/f'key_{s}.lst').read_text().split() for s in ('train', 'val')}
    assert all(len(v) == len(set(v)) and v for v in keysets.values())
    assert not set(keysets['train']) & set(keysets['val'])
    assert set(keysets['train']+keysets['val']) == {t['key'] for t in man['tiles']}
    assert (d/'key.lst').read_text().split() == [t['key'] for t in man['tiles']]
    count_valid = 0
    with h5py.File(d/'faults.h5', 'r') as f:
        assert set(f) == set(keysets['train']+keysets['val'])
        for t in man['tiles']+man['skipped']:
            i, j, h, w = (t[k] for k in ('row', 'col', 'height', 'width'))
            coverage[i:i+h, j:j+w] += 1
            if t in man['skipped']: continue
            g = f[t['key']]; valid = g['valid'][:].astype(bool)
            assert not valid[h:, :].any() and not valid[:, w:].any()
            assert valid.sum() == t['valid_pixels'] > 0
            count_valid += int(valid.sum())
            assert g.attrs['split'] == t['split']
            assert t['key'] in keysets[t['split']]
            for name in ('CTX', 'image', 'TW', 'k0', 'k1', 'k2'):
                a = g[name][:]
                assert np.isfinite(a).all(), (t['key'], name)
                assert a.shape[-2:] == valid.shape
                if name.startswith('k'): assert (a[~valid] == 999).all()
    assert (coverage == 1).all(), 'Gap or overlapping source pixels'
    assert count_valid == man['provenance']['valid_pixels']
    assert man['provenance']['counts'] == {s: len(v) for s, v in keysets.items()}
    report = dict(passed=True, full_source_coverage=True, overlapping_pixels=0,
                  valid_pixels=count_valid, counts={s: len(v) for s, v in keysets.items()},
                  positives={s: sum(t['is_positive'] for t in man['tiles'] if t['split'] == s) for s in keysets},
                  padded_tiles=sum(t['height'] < man['provenance']['tile'] or t['width'] < man['provenance']['tile'] for t in man['tiles']))
    (d/'audit.json').write_text(json.dumps(report, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Patch
    fig, ax = plt.subplots(figsize=(8, 9))
    colors = dict(train='#3975b9', val='#e4933d')
    for t in man['tiles']:
        x0, y0, x1, y1 = t['bbox']
        ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, facecolor=colors[t['split']], edgecolor='white', alpha=.7))
        ax.text((x0+x1)/2, (y0+y1)/2, t['key'], fontsize=6, ha='center')
    ax.autoscale(); ax.set_aspect('equal'); ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.set_title('Non-overlapping grid / random train-validation split')
    ax.legend(handles=[Patch(facecolor=c, label=s) for s, c in colors.items()])
    fig.tight_layout(); fig.savefig(d/'split_map.png', dpi=150); plt.close(fig)
    print(json.dumps(report, indent=2)); return report

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataDir', default='datasets/ca_stride384')
    audit(p.parse_args().dataDir)
