"""Reproducible k0 metrics: unique spatial pixels, exact AP, validation-only threshold."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt


def ranking_metrics(probability, truth):
    p = np.asarray(probability, dtype=np.float64).ravel()
    t = np.asarray(truth, dtype=bool).ravel()
    if not len(p) or not np.isfinite(p).all():
        raise ValueError('Empty or non-finite probability scores')
    positives = int(t.sum())
    negatives = len(t) - positives
    order = np.argsort(-p, kind='stable')
    scores, labels = p[order], t[order]
    ends = np.r_[np.flatnonzero(np.diff(scores)), len(scores)-1]
    tp = np.cumsum(labels, dtype=np.int64)[ends]
    fp = ends + 1 - tp
    precision = tp / (tp + fp)
    recall = tp / positives if positives else np.zeros_like(precision)
    ap = float(np.sum(np.diff(np.r_[0., recall]) * precision)) if positives else None
    auc = float(np.trapezoid(np.r_[0., recall], np.r_[0., fp / negatives])) if positives and negatives else None
    f1 = 2 * tp / np.maximum(2 * tp + fp + positives - tp, 1)
    best = int(np.argmax(f1))  # descending scores: higher threshold wins ties
    return dict(AP=ap, ROC_AUC=auc, threshold=float(scores[ends[best]]),
                best_F1=float(f1[best]), precision=precision, recall=recall)


def fixed_metrics(prob, truth, valid, threshold, tolerance=3):
    truth = truth.astype(bool) & valid
    pred = (prob >= threshold) & valid
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    precision = tp / (tp+fp) if tp+fp else 0.
    recall = tp / (tp+fn) if tp+fn else 0.
    f1 = 2*tp / (2*tp+fp+fn) if 2*tp+fp+fn else 0.
    iou = tp / (tp+fp+fn) if tp+fp+fn else 0.
    near_truth = distance_transform_edt(~truth) <= tolerance if truth.any() else np.zeros_like(valid)
    near_pred = distance_transform_edt(~pred) <= tolerance if pred.any() else np.zeros_like(valid)
    bp = int((pred & near_truth).sum()) / int(pred.sum()) if pred.any() else 0.
    br = int((truth & near_pred).sum()) / int(truth.sum()) if truth.any() else 0.
    return dict(threshold=float(threshold), precision=precision, recall=recall, F1=f1,
                Dice=f1, IoU=iou, tp=tp, fp=fp, fn=fn,
                buffered_precision=bp, buffered_recall=br,
                buffered_F1=2*bp*br/(bp+br) if bp+br else 0., buffer_euclidean_pixels=tolerance)


def positions(length, crop):
    if length < crop:
        raise ValueError('Stored tile smaller than inference crop')
    return sorted({0, (length-crop)//2, length-crop})


def rect(key, shape):
    y, x = map(int, key.split('_')[1:])
    return y, x, y+shape[0], x+shape[1]


def split_audit(h5, splits):
    rectangles = {key: rect(key, h5[key]['k0'].shape) for keys in splits.values() for key in keys}
    report = {}
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        pairs = []
        for ka in splits[a]:
            ay, ax, ay1, ax1 = rectangles[ka]
            for kb in splits[b]:
                by, bx, by1, bx1 = rectangles[kb]
                if min(ay1, by1) > max(ay, by) and min(ax1, bx1) > max(ax, bx):
                    pairs.append([ka, kb])
        report[a+'_'+b] = dict(shared_keys=sorted(set(splits[a]) & set(splits[b])),
                               overlapping_pairs=len(pairs), examples=pairs[:5])
    if report['train_test']['overlapping_pairs'] or report['val_test']['overlapping_pairs']:
        raise ValueError('Test footprint overlaps train/validation; cannot report held-out scores')
    return report


def inputs_for_tile(group):
    ctx = group['CTX'][:].astype('float32')
    elevation = group['image'][:].astype('float32')
    rgb = group['TW'][:].astype('float32')
    v0 = np.isfinite(ctx) & (ctx > -2147483647)
    v1 = np.isfinite(elevation) & (elevation > -2147483647)
    ctx = np.where(v0, ctx, 0)
    ctx = (ctx/255-.5)/.5
    elevation = np.where(v1, elevation, np.nan)
    elevation = (elevation-np.nanmean(elevation))/(np.nanstd(elevation)+1e-6)
    elevation = np.nan_to_num(elevation)
    rgb = np.where(np.isfinite(rgb) & (rgb > -2147483647), rgb, 0)
    rgb = (rgb/255-.5)/.5
    image = np.concatenate([ctx[None], elevation[None], v0[None], v1[None], rgb]).astype('float32')
    distance = group['k0'][:]
    valid = v0 & v1 & np.isfinite(distance) & (distance >= 0)
    return image, distance, valid


def predict_split(net, h5, keys, args, device, output, split):
    import torch
    boxes = [rect(k, h5[k]['k0'].shape) for k in keys]
    y0, x0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    height, width = max(b[2] for b in boxes)-y0, max(b[3] for b in boxes)-x0
    summed = np.zeros((height, width), dtype=np.float64)
    counts = np.zeros((height, width), dtype=np.uint16)
    truth = np.zeros((height, width), dtype=bool)
    label_conflicts = np.zeros((height, width), dtype=bool)
    raw_elevation = np.zeros((height, width), dtype=np.float32)
    tile_pixels = 0
    with torch.inference_mode():
        for index, key in enumerate(keys):
            image, distances, valid = inputs_for_tile(h5[key])
            h, w = valid.shape
            acc = np.zeros((h, w), dtype=np.float32)
            weight = np.zeros((h, w), dtype=np.uint8)
            for y in positions(h, args.crop):
                for x in positions(w, args.crop):
                    tensor = torch.from_numpy(image[None, :, y:y+args.crop, x:x+args.crop]).to(device)
                    p = net(tensor)[0, 0].cpu().numpy()
                    if not np.isfinite(p).all() or p.min() < 0 or p.max() > 1:
                        raise ValueError(f'Invalid probabilities: {key}')
                    acc[y:y+args.crop, x:x+args.crop] += p
                    weight[y:y+args.crop, x:x+args.crop] += 1
            if not weight.all():
                raise ValueError('Sliding windows left uncovered pixels')
            p = acc/weight
            target = (distances < args.d0) & valid
            y, x, _, _ = boxes[index]
            region = np.s_[y-y0:y-y0+h, x-x0:x-x0+w]
            overlap = (counts[region] > 0) & valid
            elevation = h5[key]['image'][:].astype('float32')
            if overlap.any() and not np.allclose(raw_elevation[region][overlap], elevation[overlap], rtol=0, atol=1e-3):
                raise ValueError(f'Inconsistent terrain at shared coordinates: {key}')
            # Distance transforms were generated independently inside each tile.
            # Keep all original positive annotations when those boundaries disagree.
            label_conflicts[region] |= overlap & (truth[region] != target)
            raw_elevation[region] = np.where(valid, elevation, raw_elevation[region])
            summed[region] += p * valid
            counts[region] += valid.astype(np.uint16)
            truth[region] |= target
            tile_pixels += int(valid.sum())
            if (index+1) % 10 == 0 or index+1 == len(keys):
                print(f'{split}: {index+1}/{len(keys)} tiles', flush=True)
    valid = counts > 0
    p = (summed/np.maximum(counts, 1)).astype('float32')
    np.savez_compressed(output/f'{split}_mosaic.npz', probability=p, truth=truth, valid=valid,
                        coverage_count=counts, row_origin=y0, col_origin=x0)
    return p, truth, valid, dict(tiles=len(keys), unique_valid_pixels=int(valid.sum()),
                               tile_weighted_valid_pixels=tile_pixels,
                               positive_pixels=int(truth.sum()), positive_fraction=float(truth[valid].mean()),
                               conflicting_label_pixels=int(label_conflicts.sum()))


def main():
    import torch
    from h5py import File
    from torchUnet import conv_sam
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--outDir', required=True)
    parser.add_argument('--dataDir', default='earth_dataset_v3')
    parser.add_argument('--crop', type=int, default=256)
    parser.add_argument('--d0', type=float, default=.02)
    parser.add_argument('--tol', type=int, default=3)
    args = parser.parse_args()
    output = Path(args.outDir)
    output.mkdir(parents=True, exist_ok=True)
    data = Path(args.dataDir)
    splits = {s:(data/f'key_{s}.lst').read_text().split() for s in ('train','val','test')}
    if any(len(k) != len(set(k)) or not k for k in splits.values()):
        raise ValueError('Empty split or duplicate keys')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    net = conv_sam(32, dropout=0., isvit=False, isalign=False).to(device)
    net.load_state_dict(torch.load(args.model, map_location='cpu', weights_only=True), strict=True)
    net.eval()
    print('device =', device, 'model =', args.model, flush=True)
    with File(data/'faults.h5', 'r') as h5:
        audit = split_audit(h5, splits)
        vp, vt, vv, val_info = predict_split(net, h5, splits['val'], args, device, output, 'validation')
        val_ranking = ranking_metrics(vp[vv], vt[vv])
        threshold = val_ranking['threshold']  # frozen before reading test predictions
        print('Validation-selected threshold =', threshold, flush=True)
        p, t, v, test_info = predict_split(net, h5, splits['test'], args, device, output, 'test')
    ranking = ranking_metrics(p[v], t[v])
    report = dict(
        model=str(Path(args.model).resolve()),
        model_sha256=hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),
        protocol='k0_full_tile_sliding_windows_then_unique_pixel_mosaic_v1',
        overlap_label_policy='Union of original positive annotations; tile-local distance transforms can differ at edges',
        AP_definition='sum of recall increments times precision at each distinct descending score (non-interpolated)',
        threshold_selection='Maximum pixel F1 on validation mosaic only; ties prefer higher threshold',
        d0_degrees=args.d0, crop=args.crop, split_keys=splits, split_audit=audit,
        validation=dict(**val_info, AP=val_ranking['AP'], selected_threshold=threshold,
                        **{'at_selected_threshold':fixed_metrics(vp, vt, vv, threshold, args.tol)}),
        test=dict(**test_info, AP=ranking['AP'], ROC_AUC=ranking['ROC_AUC'],
                  at_validation_threshold=fixed_metrics(p, t, v, threshold, args.tol),
                  at_threshold_0_5=fixed_metrics(p, t, v, .5, args.tol)),
        limitations=[
            'Original fault labels are incomplete; this measures agreement with the catalogue, not verified missing faults.',
            'Training and validation tile footprints overlap; test footprints do not overlap either.',
            'Test contains overlapping tiles; scores use unique geographic pixels after averaging predictions.',
            'AP is pixel-level AP for one class, not object detection mAP or AP50.',
            'No claim of improvement over PU baseline without re-evaluation using this exact protocol.'
        ])
    (output/'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    n = len(ranking['recall'])
    indices = np.unique(np.linspace(0, n-1, min(n, 1500)).astype(int))
    axes[0].plot(ranking['recall'][indices], ranking['precision'][indices],
                 label=f"AP={ranking['AP']:.4f}")
    axes[0].axhline(test_info['positive_fraction'], ls='--', color='gray', label='Positive pixel fraction')
    axes[0].set(xlabel='Recall', ylabel='Precision', xlim=(0,1), ylim=(0,1), title='Spatial test: pixel PR curve')
    axes[0].legend()
    values = report['test']['at_validation_threshold']
    names = ['AP', 'Precision', 'Recall', 'F1', 'IoU', 'Buffered F1']
    nums = [ranking['AP'], values['precision'], values['recall'], values['F1'], values['IoU'], values['buffered_F1']]
    axes[1].barh(names[::-1], nums[::-1], color='#3178aa')
    axes[1].set(xlim=(0, 1), title=f'Threshold from validation: {threshold:.5g}')
    for i, val in enumerate(nums[::-1]):
        axes[1].text(val+.015, i, f'{val:.3f}', va='center')
    fig.tight_layout()
    fig.savefig(output/'metrics_overview.png', dpi=150)
    plt.close(fig)
    print(json.dumps(report['test'], indent=2), flush=True)
    print('Saved:', output/'metrics.json', flush=True)


if __name__ == '__main__':
    main()
