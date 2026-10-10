"""Count spatially deduplicated predicted components from saved NPZ exports.

A component is an 8-connected prediction region, not a named geological fault.
Matching means ANY part is within a fixed Euclidean pixel distance of the
original d0 mask. No ground-truth-dependent merging, threshold selection, or
new model inference is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi


def components(probability, valid, truth, threshold):
    prediction = (probability >= threshold) & valid
    labels, count = ndi.label(prediction, structure=np.ones((3, 3), np.uint8))
    ids = np.arange(1, count + 1)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)[1:]
    distance = ndi.distance_transform_edt(~truth) if truth.any() else np.full(truth.shape, np.inf)
    minimum = np.asarray(ndi.minimum(distance, labels, ids)) if count else np.array([])
    return labels, sizes, distance, minimum


def self_check():
    valid = np.ones((24, 24), bool)
    truth = np.zeros_like(valid)
    truth[3:10, 3] = True
    p = np.zeros(valid.shape, np.float32)
    p[3:10, 6:9] = .8  # 21-pixel segment, 3 pixels from label
    p[15:20, 16:20] = .8  # 20-pixel nonmatch
    p[0, 22] = .8  # a single-pixel fragment
    _, sizes, _, minimum = components(p, valid, truth, .5)
    assert (sizes >= 16).sum() == 2
    assert ((sizes >= 16) & (minimum <= 3)).sum() == 1
    assert ((sizes >= 16) & (minimum <= 2)).sum() == 0
    _, _, _, no_truth = components(p, valid, np.zeros_like(truth), .5)
    assert np.isinf(no_truth).all()
    # Eight-connectivity joins diagonal pixels; distinct regions stay distinct.
    diag = np.eye(4, dtype=np.float32)
    assert len(components(diag, np.ones((4, 4), bool), diag > 0, .5)[1]) == 1
    # Average of two identical overlapping predictions must not duplicate objects.
    assert np.array_equal(components((p + p) / 2, valid, truth, .5)[1], sizes)


def make_mosaic(source, rows, threshold, d0, h5):
    boxes = []
    for row in rows:
        y, x = map(int, row['key'].split('_')[1:])
        h, w = row['shape']
        boxes.append((y, x, h, w))
    y0, x0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    shape = (max(y+h for y, x, h, w in boxes)-y0, max(x+w for y, x, h, w in boxes)-x0)
    summed = np.zeros(shape, np.float64)
    coverage = np.zeros(shape, np.uint16)
    truth = np.zeros(shape, bool)
    terrain = np.zeros(shape, np.float32)
    fingerprints = []
    tile_pixels = 0
    for row, (y, x, h, w) in zip(rows, boxes):
        path = source / row['split'] / (row['key']+'.npz')
        with np.load(path, allow_pickle=False) as z:
            p, v, t = z['probability'], z['valid'], z['truth']
            assert p.shape == v.shape == t.shape == (h, w)
            assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
            assert np.isclose(float(z['threshold']), threshold) and np.isclose(float(z['d0']), d0)
            assert np.array_equal(z['prediction'], (p >= threshold) & v)
            assert int(v.sum()) == row['valid_pixels'] and int(t.sum()) == row['annotated_pixels']
            assert int(z['prediction'].sum()) == row['predicted_pixels']
            group = h5[row['key']]
            assert np.array_equal(t, (group['k0'][:] < d0) & v)
            elevation = group['image'][:].astype(np.float32)
            region = np.s_[y-y0:y-y0+h, x-x0:x-x0+w]
            overlap = (coverage[region] > 0) & v
            assert np.allclose(terrain[region][overlap], elevation[overlap], rtol=0, atol=1e-3)
            terrain[region] = np.where(v, elevation, terrain[region])
            summed[region] += p * v
            coverage[region] += v.astype(np.uint16)
            truth[region] |= t
            tile_pixels += int(v.sum())
        fingerprints.append(dict(path=str(path.relative_to(source)), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    p = (summed / np.maximum(coverage, 1)).astype(np.float32)
    return p, coverage > 0, truth, (y0, x0), dict(tiles=len(rows),
        tile_weighted_valid_pixels=tile_pixels, unique_valid_pixels=int((coverage > 0).sum()),
        maximum_overlap=int(coverage.max()), input_files=fingerprints)


def analyze(name, p, valid, truth, origin, out, threshold, min_pixels, tolerance):
    labels, sizes, distance, minimum = components(p, valid, truth, threshold)
    ids = np.arange(1, len(sizes)+1)
    near = (distance <= tolerance) & valid
    near_sizes = np.bincount(labels[near], minlength=len(sizes)+1)[1:]
    keep = sizes >= min_pixels
    matched = keep & (minimum <= tolerance)
    sensitivity = []
    for size in sorted({1, 8, 16, 32, min_pixels}):
        for tol in sorted({0, 1, 2, 3, 5, tolerance}):
            n = int((sizes >= size).sum())
            m = int(((sizes >= size) & (minimum <= tol)).sum())
            sensitivity.append(dict(min_pixels=size, tolerance_pixels=tol, predicted=n, matched=m, unmatched=n-m))
    boxes = ndi.find_objects(labels)
    border = valid & ~ndi.binary_erosion(valid, structure=np.ones((3, 3)), border_value=0)
    border_ids = set(np.unique(labels[border]).tolist()) - {0}
    details = []
    for index, box in enumerate(boxes):
        y, x = box
        details.append(dict(id=index+1, pixels=int(sizes[index]), retained=bool(keep[index]),
            matched=bool(minimum[index] <= tolerance),
            nearest_label_pixels=float(minimum[index]) if np.isfinite(minimum[index]) else None,
            pixels_near_label=int(near_sizes[index]), fraction_near_label=float(near_sizes[index]/sizes[index]),
            touches_coverage_boundary=index+1 in border_ids,
            global_bbox_row_col=[y.start+origin[0], x.start+origin[1], y.stop+origin[0], x.stop+origin[1]]))
    result = dict(predicted=int(keep.sum()), matched=int(matched.sum()), unmatched=int((keep & ~matched).sum()),
        raw_predicted=len(sizes), raw_matched=int((minimum <= tolerance).sum()),
        raw_unmatched=int((minimum > tolerance).sum()), excluded_small_components=int((~keep).sum()),
        majority_area_matched=int((keep & (near_sizes >= sizes*.5)).sum()),
        retained_touching_coverage_boundary=int(sum(int(i) in border_ids for i in ids[keep])),
        sensitivity=sensitivity)
    assert result['predicted'] == result['matched'] + result['unmatched']
    assert sum(d['retained'] and d['matched'] for d in details) == result['matched']
    (out/f'{name}_components.json').write_text(json.dumps(details, indent=2, allow_nan=False), encoding='utf-8')
    np.savez_compressed(out/f'{name}_mosaic.npz', probability=p, valid=valid, truth=truth,
        component_id=labels, row_origin=origin[0], col_origin=origin[1])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    status = np.zeros(len(sizes)+1, np.uint8)
    status[ids[keep]] = 3
    status[ids[matched]] = 2
    display = np.where(valid, 0, 4).astype(np.uint8)
    display[truth] = 1
    retained = status[labels] > 0
    display[retained] = status[labels][retained]
    colors = ['#f4f6f8', '#d69921', '#15865a', '#cc4381', '#ffffff']
    fig, ax = plt.subplots(figsize=(12, 11))
    ax.imshow(display, cmap=ListedColormap(colors), vmin=0, vmax=4, origin='lower', interpolation='nearest')
    ax.set_title(f'{name}: {result["predicted"]} segments; {result["matched"]} matched; {result["unmatched"]} unmatched\n'
                 f'p >= {threshold:g}; 8-connected; >= {min_pixels} pixels; any contact within {tolerance:g} pixels of label mask')
    ax.set_xlabel(f'Column relative to {origin[1]}'); ax.set_ylabel(f'Row relative to {origin[0]}')
    ax.legend(handles=[Patch(color=colors[i], label=s) for i, s in [(1, 'Original annotation'), (2, 'Tolerant match'), (3, 'Unmatched / unverified')]], loc='upper right')
    fig.tight_layout(); fig.savefig(out/f'{name}_overview.png', dpi=150); plt.close(fig)
    return result


def main():
    import h5py
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--h5', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--min-pixels', type=int, default=16)
    parser.add_argument('--tolerance', type=float, default=3)
    args = parser.parse_args()
    if args.min_pixels < 1 or args.tolerance < 0: parser.error('Invalid counting parameters')
    self_check()
    manifest_file = args.source/'manifest.json'
    manifest = json.loads(manifest_file.read_text())
    assert manifest['status'] == 'complete' and manifest['full_export']
    assert np.isfinite(manifest['d0']) and manifest['d0'] > 0
    assert len({(r['split'], r['key']) for r in manifest['images']}) == len(manifest['images'])
    for split, count in manifest['counts'].items():
        assert sum(r['split'] == split for r in manifest['images']) == count
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Choose an empty output directory.')
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(source=str(args.source), manifest_sha256=hashlib.sha256(manifest_file.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        model=manifest['model'], model_sha256=manifest['model_sha256'], epoch=manifest['epoch'],
        d0=manifest['d0'], threshold=manifest['threshold'], minimum_component_pixels=args.min_pixels,
        tolerance_pixels=args.tolerance, connectivity=8,
        definition='One 8-connected predicted region after averaging valid tile probabilities at the same geographic grid cell. No gap closing; touching branches count together.',
        matching='ANY predicted pixel within Euclidean tolerance of original d0 label mask. Non-exclusive matching: multiple predictions may match the same labeled fault.',
        label_policy='Union of original positive masks across overlapping tiles; buffer is ADDITIONAL to original d0 band.',
        limitations=['Segments are not named or geologically verified faults.',
                    'Unmatched segments are unverified, not automatically false.',
                    'Coverage boundaries and prediction gaps can split one geological fault.',
                    'Total covers exported train and test footprints, excludes unexported validation-only or other areas.'],
        split_audit=manifest['split_audit'], self_checks='passed', results={})
    with h5py.File(args.h5, 'r') as h5:
        for split in ['train_seen', 'test_unseen', 'all_exported']:
            rows = [r for r in manifest['images'] if split == 'all_exported' or r['split'] == split]
            p, valid, truth, origin, audit = make_mosaic(args.source, rows, manifest['threshold'], manifest['d0'], h5)
            result = analyze(split, p, valid, truth, origin, args.output, manifest['threshold'], args.min_pixels, args.tolerance)
            result['audit'] = audit
            report['results'][split] = result
            print(split, {k:v for k,v in result.items() if k not in ('sensitivity', 'audit')}, flush=True)
    (args.output/'counts.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    labels = {'train_seen':'训练区域', 'test_unseen':'未见测试区域', 'all_exported':'全部已导出区域（重新拼接）'}
    lines = [f'# d0={manifest["d0"]:g} 预测段计数', '',
        f'模型：best_student，第 {manifest["epoch"]} 轮；预测阈值 {manifest["threshold"]}。', '',
        f'主口径：地理重叠区域先平均概率，再按八连通区域计数，保留面积 ≥{args.min_pixels} 像素的预测段。',
        f'宽松匹配：预测段任意一部分距离原标签掩膜 ≤{args.tolerance:g} 像素即计为匹配，不要求完全重合。该容差额外叠加在 d0={manifest["d0"]:g} 的原标签带之外。', '',
        '| 范围 | 预测段数 | 大致重合 | 未匹配 | 排除的小碎点 |', '|---|---:|---:|---:|---:|']
    for name, result in report['results'].items():
        lines.append(f'| {labels[name]} | {result["predicted"]} | {result["matched"]} | {result["unmatched"]} | {result["excluded_small_components"]} |')
    lines += ['', '## 不过滤碎点的原始计数', '', '| 范围 | 预测连通区域 | 大致重合 | 未匹配 |', '|---|---:|---:|---:|']
    for name, result in report['results'].items():
        lines.append(f'| {labels[name]} | {result["raw_predicted"]} | {result["raw_matched"]} | {result["raw_unmatched"]} |')
    lines += ['', '## 解释与复核', '',
        '- “条”指连续预测区域；相接分支算一段，断开的片段分别计数，没有人为跨空隙连接。不能解释为独立命名地质断层的数量。',
        '- 匹配不是一一对应：多个预测段可以靠近同一条标签。本统计回答“多少预测段与标签符合”，没有声称发现多少不同的目录断层。',
        '- 任意局部靠近即可匹配，属于宽松筛查；counts.json 另外保留至少一半面积靠近标签的结果，及 0/1/2/3/5 像素容差和 1/8/16/32 像素面积门槛的敏感性计数。',
        '- 没匹配到的段仍需核查，不能直接叫误报；边界截断也可能增加段数。',
        f'- 统计使用已导出的 {manifest["counts"]["train_seen"]} 个训练图块和 {manifest["counts"]["test_unseen"]} 个测试图块。全部计数只覆盖其地理并集，不代表整个加州或全部数据集。',
        '- 重叠区域的原始地形一致性、NPZ 预测阈值、原 HDF5 标签、导出像素数均已核验；每个输入文件记录 SHA256。',
        '- *_components.json 保存每段编号、像素数、最短标签距离、匹配比例及原始栅格包围盒；*_mosaic.npz 保存对应编号图。',
        '- *_overview.png：绿色为宽松匹配段，粉色为未匹配段，金色为原标签。', '']
    (args.output/'计数说明.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
