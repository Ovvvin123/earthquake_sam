"""Compute first-version best-student train/validation AP from saved probabilities."""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from count_fault_segments import make_mosaic
from evaluate_cv import inputs_for_tile, ranking_metrics, split_audit


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(f.read()).hexdigest()


def main():
    base = Path(__file__).resolve().parent
    run = base/'train_v5_online'
    out = run/'cv_evaluation/train_validation_ap'
    out.mkdir(exist_ok=True)
    previous = json.loads((run/'cv_evaluation/student/metrics.json').read_text())
    fingerprints = json.loads((run/'cv_evaluation/evaluation_provenance.json').read_text())['sha256']
    checked = {}
    for rel in ['evaluate_cv.py', 'earth_dataset_v3/faults.h5', 'earth_dataset_v3/key_train.lst',
                'earth_dataset_v3/key_val.lst', 'earth_dataset_v3/key_test.lst', 'train_v5_online/best_student/model.pth']:
        checked[rel] = sha(base/rel)
        assert checked[rel] == fingerprints[rel], f'Historical source changed: {rel}'
    splits = previous['split_keys']
    manifest = json.loads((run/'predictions_seen_unseen/manifest.json').read_text())
    assert manifest['status'] == 'complete' and manifest['epoch'] == 186
    assert manifest['model_sha256'] == previous['model_sha256'] == checked['train_v5_online/best_student/model.pth']
    assert manifest['d0'] == previous['d0_degrees'] == .02
    rows = [r for r in manifest['images'] if r['split'] == 'train_seen']
    assert set(r['key'] for r in rows) == set(splits['train'])
    sources = {'training': run/'segment_counts/train_seen_mosaic.npz',
               'validation': run/'cv_evaluation/student/validation_mosaic.npz'}
    with h5py.File(base/'earth_dataset_v3/faults.h5', 'r') as h5:
        audit = split_audit(h5, splits)
        p, valid, truth, origin, _ = make_mosaic(run/'predictions_seen_unseen', rows, .5, .02, h5)
        with np.load(sources['training']) as z:
            assert np.array_equal(p, z['probability'])
            assert np.array_equal(valid, z['valid']) and np.array_equal(truth, z['truth'])
            assert origin == (int(z['row_origin']), int(z['col_origin']))
        # Independently rebuild validation coverage and original-label union.
        with np.load(sources['validation']) as z:
            vv, vt = np.zeros_like(z['valid']), np.zeros_like(z['truth'])
            y0, x0 = int(z['row_origin']), int(z['col_origin'])
            for key in splits['val']:
                y, x = map(int, key.split('_')[1:])
                _, distance, tile_valid = inputs_for_tile(h5[key])
                h, w = tile_valid.shape
                region = np.s_[y-y0:y-y0+h, x-x0:x-x0+w]
                vv[region] |= tile_valid
                vt[region] |= (distance < .02) & tile_valid
            assert np.array_equal(vv, z['valid']) and np.array_equal(vt, z['truth'])
    report = dict(model='best_student', epoch=186, model_sha256=previous['model_sha256'],
                  d0_degrees=.02, evaluated_channel='k0', inference='existing 256x256 sliding-window averages over full 384x384 tiles',
                  aggregation='average probabilities over overlapping geographic pixels; union original positive labels; count each valid geographic pixel once',
                  AP_definition='non-interpolated pixel AP: sum of recall increments times precision at each distinct score; no binary threshold or distance tolerance',
                  pseudo_labels_used_as_truth=False, split_audit=audit, verified_source_sha256=checked,
                  script_sha256=sha(Path(__file__)), results={},
                  limitations=['Training and validation overlap geographically; validation is not an independent generalization estimate.',
                               'The checkpoint was selected by validation loss; it was not selected by train/validation AP.',
                               'Original labels are incomplete; AP measures agreement with the catalogue.'])
    curves = []
    for name, path in sources.items():
        with np.load(path) as z:
            valid, truth, p = z['valid'], z['truth'], z['probability']
            assert np.isfinite(p[valid]).all() and (p[valid] >= 0).all() and (p[valid] <= 1).all()
            r = ranking_metrics(p[valid], truth[valid])
            if name == 'validation':
                assert np.isclose(r['AP'], previous['validation']['AP'], atol=1e-12, rtol=0)
            report['results'][name] = dict(AP=r['AP'], AP_percent=100*r['AP'],
                tiles=len(splits['train' if name == 'training' else 'val']),
                unique_valid_pixels=int(valid.sum()), original_positive_pixels=int(truth[valid].sum()),
                positive_fraction=float(truth[valid].mean()), source=str(path), source_sha256=sha(path))
            curves.append((name, r))
    report['previous_held_out_test_AP'] = previous['test']['AP']
    (out/'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3))
    for ax, (name, r), color in zip(axes, curves, ['#2465a8', '#dd8933']):
        indices = np.unique(np.linspace(0, len(r['recall'])-1, min(2500, len(r['recall']))).astype(int))
        ax.step(np.r_[0., r['recall'][indices]], np.r_[1., r['precision'][indices]], where='pre', color=color, lw=1.7,
                label=f'Pixel AP = {r["AP"]:.6f} ({100*r["AP"]:.2f}%)')
        ax.axhline(report['results'][name]['positive_fraction'], color='#89939e', ls='--', lw=1,
                   label='Positive-pixel fraction')
        ax.set(xlim=(0,1), ylim=(0,1), xlabel='Recall', ylabel='Precision',
               title=f'{name.title()} set ({report["results"][name]["tiles"]} tiles)')
        ax.grid(alpha=.18); ax.legend(loc='upper right', fontsize=9)
    fig.suptitle('Original online correction: best student, epoch 186', fontsize=14)
    fig.text(.5,.025,'Unique geographic pixels; original k0 labels (d0=0.02). Train/validation footprints overlap.\n'
             'Exact AP uses all score levels; only the displayed curves are downsampled.', ha='center',fontsize=9,color='#536273')
    fig.tight_layout(rect=(0,.1,1,.94)); fig.savefig(out/'train_validation_pr.png',dpi=180);plt.close(fig)
    print(json.dumps(report['results'],indent=2),flush=True)
    print('train_val overlapping pairs:',audit['train_val']['overlapping_pairs'],flush=True)


if __name__=='__main__': main()
