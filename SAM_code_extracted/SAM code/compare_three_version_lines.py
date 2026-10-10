"""Compare saved best-student test centerlines against shared annotation masks."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from count_fault_branches import thin, trace_branches, self_check


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['test', 'train'], default='test')
    split = parser.parse_args().split
    training = split == 'train'
    export_split = 'train_seen' if training else 'test_unseen'
    region_cn = '训练区' if training else '测试区'
    self_check()
    base = Path(__file__).resolve().parent
    out = base / 'three_version_line_comparison'
    out.mkdir(exist_ok=True)
    specs = [
        ('online', 'Online correction', 'train_v5_online', 'segment_counts/test_unseen_mosaic.npz'),
        ('unimatch', 'Online + UniMatch', 'train_v5_unimatch', 'cv_evaluation/student/test_mosaic.npz'),
        ('unimatch_d001', 'UniMatch, training d0=0.01', 'train_v5_unimatch_d001', 'segment_counts/test_unseen_mosaic.npz'),
    ]
    if training:
        specs = [(k, title, folder, 'train_line_evaluation/train_seen_mosaic.npz'
                  if k == 'unimatch' else 'segment_counts/train_seen_mosaic.npz')
                 for k, title, folder, _ in specs]
    arrays = []
    for key, title, folder, rel in specs:
        with np.load(base / folder / rel) as z:
            arrays.append({k: z[k].copy() for k in z.files})
    valid = arrays[0]['valid']
    for z in arrays:
        assert np.array_equal(valid, z['valid'])
        assert int(z['row_origin']) == int(arrays[0]['row_origin'])
        assert int(z['col_origin']) == int(arrays[0]['col_origin'])
        assert np.isfinite(z['probability']).all()
    assert np.array_equal(arrays[0]['truth'], arrays[1]['truth'])
    assert not np.any(arrays[2]['truth'] & ~arrays[0]['truth'])
    if not training:
        prior_cv = np.load(base / 'train_v5_online/cv_evaluation/student/test_mosaic.npz')
        assert np.array_equal(prior_cv['probability'], arrays[0]['probability'])
    shared_truths = {'0.02': arrays[0]['truth'], '0.01': arrays[2]['truth']}
    distances = {k: ndi.distance_transform_edt(~v) for k, v in shared_truths.items()}
    eval_meta = json.loads((base / 'train_v5_unimatch/cv_evaluation/student/metrics.json').read_text())
    expected_keys = set(eval_meta['split_keys'][split])
    for folder in ('train_v5_online', 'train_v5_unimatch_d001'):
        manifest = json.loads((base / folder / 'predictions_seen_unseen/manifest.json').read_text())
        assert expected_keys == {r['key'] for r in manifest['images'] if r['split'] == export_split}
    report = dict(
        scope=('seen training region' if training else 'spatially held-out test region') + '; best student selected by validation loss',
        split=split,
        tiles=len(expected_keys), unique_valid_pixels=int(valid.sum()),
        keys=sorted(expected_keys), threshold=0.5, minimum_component_pixels=16,
        minimum_centerline_points=5, main_shared_label_d0=0.02, tolerance_pixels=3,
        main_match_rule='At least 50% of unique centerline points within 3 pixels of the shared d0=0.02 original annotation mask',
        loose_match_rule='Any centerline point within the same tolerance',
        script_sha256=sha(Path(__file__)), branch_algorithm_sha256=sha(base / 'count_fault_branches.py'),
        shared_label_pixels={k: int(v.sum()) for k, v in shared_truths.items()}, results=[])
    plot_data = []
    for (key, title, folder, rel), z in zip(specs, arrays):
        labels, n = ndi.label((z['probability'] >= .5) & valid, np.ones((3, 3)))
        sizes = np.bincount(labels.ravel())
        accepted = sizes >= 16
        accepted[0] = False
        mask = accepted[labels]
        skeleton = thin(mask)
        assert ndi.label(mask, np.ones((3, 3)))[1] == ndi.label(skeleton, np.ones((3, 3)))[1]
        raw_paths, _ = trace_branches(skeleton)
        paths = [p for p in raw_paths if len(np.unique(p, axis=0)) >= 5]
        details = []
        for i, p in enumerate(paths, 1):
            points = np.unique(p, axis=0)
            fractions = {d0: {str(t): float((d[points[:, 0], points[:, 1]] <= t).mean())
                             for t in (0, 1, 2, 3, 5)} for d0, d in distances.items()}
            details.append(dict(id=i, points=len(points), fractions_near_label=fractions,
                                global_row_col=(p+np.array([int(z['row_origin']), int(z['col_origin'])])).tolist()))
        sensitivity = []
        for d0 in distances:
            for t in (0, 1, 2, 3, 5):
                f = np.array([r['fractions_near_label'][d0][str(t)] for r in details])
                sensitivity.append(dict(label_d0=float(d0), tolerance_pixels=t,
                                        predicted=len(paths), loose=int((f > 0).sum()),
                                        majority=int((f >= .5).sum()), all_points=int((f == 1).sum())))
        main = next(r for r in sensitivity if r['label_d0'] == .02 and r['tolerance_pixels'] == 3)
        best = json.loads((base / folder / 'best_student/metrics.json').read_text())
        row = dict(version=key, title=title, folder=folder, epoch=best['epoch'],
                   input_mosaic=str(base / folder / rel), input_sha256=sha(base / folder / rel),
                   **main, majority_fraction=main['majority']/len(paths) if paths else None,
                   partial_only=main['loose']-main['majority'], unmatched=len(paths)-main['loose'],
                   raw_branches=len(raw_paths), discarded_short=len(raw_paths)-len(paths),
                   sensitivity=sensitivity)
        assert row['majority'] + row['partial_only'] + row['unmatched'] == row['predicted']
        report['results'].append(row)
        (out / f'{key}_{split}_segments.json').write_text(json.dumps(details, indent=2), encoding='utf-8')
        plot_data.append((title, row, paths, details))
        print(json.dumps({k: v for k, v in row.items() if k != 'sensitivity'}, ensure_ascii=False), flush=True)
    (out / ('train_comparison.json' if training else 'comparison.json')).write_text(json.dumps(report, indent=2), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    ys, xs = np.where(valid)
    fig, axes = plt.subplots(1, 3, figsize=(17, 9), sharex=True, sharey=True)
    bg = np.where(valid, 0, 2).astype(np.uint8)
    bg[shared_truths['0.02']] = 1
    for ax, (title, row, paths, details) in zip(axes, plot_data):
        ax.imshow(bg, origin='lower', cmap=ListedColormap(['#eef1f4', '#c49b31', 'white']), vmin=0, vmax=2, interpolation='nearest')
        colors = []
        for p, detail in zip(paths, details):
            f = detail['fractions_near_label']['0.02']['3']
            c = '#12774b' if f >= .5 else '#357dc4' if f > 0 else '#c73d77'
            colors.append(c)
            mid = p[len(p)//2]
            ax.text(mid[1], mid[0], str(detail['id']), fontsize=6, color=c)
        ax.add_collection(LineCollection([p[:, ::-1] for p in paths], colors=colors, linewidths=1.4))
        ax.set_title(f'{title}\n{row["majority"]}/{row["predicted"]} majority matches; {row["loose"]} any-contact matches', fontsize=11)
        ax.set_xlim(xs.min()-25, xs.max()+25)
        ax.set_ylim(ys.min()-25, ys.max()+25)
        ax.set_xlabel('Global grid column')
    axes[0].set_ylabel('Global grid row')
    fig.suptitle('Three versions: identical ' + ('seen training' if training else 'held-out test') + ' region and matching rules', fontsize=16)
    handles = [Line2D([0], [0], color=c, lw=2, label=t) for c, t in [
        ('#c49b31', 'Original annotation (d0=0.02)'), ('#12774b', '>=50% centerline near annotation'),
        ('#357dc4', 'Some contact, <50%'), ('#c73d77', 'No contact / unverified')]]
    fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=10)
    fig.tight_layout(rect=(0, .08, 1, .95))
    fig.savefig(out / f'{split}_line_comparison.png', dpi=180)
    plt.close(fig)
    names = ['① 原在线补标', '② 在线补标 + UniMatch', '③ UniMatch + d0=0.01']
    lines = [f'# 三版本{region_cn}线段对比', '',
             f'使用验证 loss 选定的 best_student；同一批{len(expected_keys)}个{region_cn}图块，地理去重后{int(valid.sum()):,}个有效像素。' + ('①③复用原概率图，②使用原最佳模型补做推理；没有重新训练。' if training else '复用已有概率图。'), '',
             '统一概率阈值0.5；排除小于16像素的独立预测区域；中心线在端点/交汇点处分段，保留至少5个中心线点。统一原标签 d0=0.02，额外距离容差3像素。', '',
             '主指标：至少50%的中心线点落入标签容差范围。宽松指标：至少一个中心线点落入范围。前者是新增的描述性统计，不是经人工认证的正确率，也没有用结果优化阈值。', '',
             '| 版本 | 最佳轮次 | 预测线段 | ≥50%匹配 | 匹配比例 | 仅局部匹配 | 完全未匹配 | 宽松匹配总数 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, r in zip(names, report['results']):
        lines.append(f'| {name} | {r["epoch"]} | {r["predicted"]} | {r["majority"]} | {r["majority_fraction"]:.1%} | {r["partial_only"]} | {r["unmatched"]} | {r["loose"]} |')
    lines += ['', '## 统一标签更窄时的敏感性检查', '',
              '三组同时使用 d0=0.01 标签，容差仍为3像素；预测与分段不变。', '',
              '| 版本 | ≥50%匹配 | 宽松匹配 |', '|---|---:|---:|']
    for name, r in zip(names, report['results']):
        s = next(s for s in r['sensitivity'] if s['label_d0'] == .01 and s['tolerance_pixels'] == 3)
        lines.append(f'| {name} | {s["majority"]} | {s["loose"]} |')
    lines += ['', '## 解释边界与核验', '',
              '- 统计的是预测中心线分支，不是独立地质断层。多个分支可以匹配同一条目录断层；数量也受碎裂、交汇点和阈值影响。',
              '- 未匹配预测尚未核实，可能包括漏标断层或误报。不能把该表理解为人工确认的新断层数量。',
              '- 3像素容差额外加在统一的标签带之外，因此即使≥50%匹配也属于带容差的目录一致性指标。',
              f'- 本表比较{region_cn}；训练区不用于衡量泛化能力。仅比较最佳学生模型。',
              '- 核验了三组图块ID、有效覆盖、栅格原点一致；①②标签逐像素一致；窄标签为宽标签的子集。',
              '- 沿用已有分段实现及直线/T形/闭环自检；每组细化前后连通区域数一致；全部图边均被追踪。',
              f'- {split}对应的comparison JSON保留输入哈希和不同容差的敏感性计数；*_{split}_segments.json 保留每条中心线坐标及匹配比例；图中编号对应JSON。', '']
    (out / ('三版本训练区线段对比.md' if training else '三版本线段对比.md')).write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
