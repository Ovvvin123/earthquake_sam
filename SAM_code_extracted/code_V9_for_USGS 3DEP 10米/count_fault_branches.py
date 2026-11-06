"""Split saved prediction mosaics into skeleton branches for reviewable counts.

Pure NumPy Zhang-Suen thinning; graph paths between endpoints and junctions.
The count is an operational line-segment count, not a geological fault count.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi


def thin(mask):
    a = np.pad(mask.astype(bool), 1)
    while True:
        changed = False
        for phase in (0, 1):
            p = [a[:-2, 1:-1], a[:-2, 2:], a[1:-1, 2:], a[2:, 2:],
                 a[2:, 1:-1], a[2:, :-2], a[1:-1, :-2], a[:-2, :-2]]
            count = sum((v.astype(np.uint8) for v in p), np.zeros(mask.shape, np.uint8))
            transitions = sum(((~p[i] & p[(i+1) % 8]).astype(np.uint8) for i in range(8)),
                              np.zeros(mask.shape, np.uint8))
            if phase == 0:
                protected = (p[0] & p[2] & p[4]) | (p[2] & p[4] & p[6])
            else:
                protected = (p[0] & p[2] & p[6]) | (p[0] & p[4] & p[6])
            remove = a[1:-1, 1:-1] & (count >= 2) & (count <= 6) & (transitions == 1) & ~protected
            if remove.any():
                a[1:-1, 1:-1][remove] = False
                changed = True
        if not changed:
            return a[1:-1, 1:-1].copy()


def trace_branches(skeleton):
    height, width = skeleton.shape
    ys, xs = np.where(skeleton)
    pixels = set((ys*width+xs).tolist())
    graph = {}
    for k in sorted(pixels):
        y, x = divmod(k, width)
        neighbors = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if (dy == 0 and dx == 0) or not (0 <= y+dy < height and 0 <= x+dx < width):
                    continue
                other = (y+dy)*width+x+dx
                if other not in pixels:
                    continue
                # Avoid triangular shortcut edges at digital right-angle bends.
                if dy and dx and ((y+dy)*width+x in pixels or y*width+x+dx in pixels):
                    continue
                neighbors.append(other)
        graph[k] = neighbors
    junctions = {k for k, v in graph.items() if len(v) >= 3}
    remaining = set(junctions)
    nodes = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        stack, cluster = [seed], {seed}
        while stack:
            k = stack.pop()
            for q in graph[k]:
                if q in remaining:
                    remaining.remove(q); cluster.add(q); stack.append(q)
        nodes.append(cluster)
    nodes.extend({k} for k in sorted(pixels) if len(graph[k]) <= 1)
    owner = {k: i for i, group in enumerate(nodes) for k in group}
    visited = set()
    def edge(a, b): return (min(a, b), max(a, b))
    # Junction-internal edges are contracted into one node.
    for group in nodes:
        for k in group:
            for q in graph[k]:
                if q in group: visited.add(edge(k, q))
    paths = []
    for group in nodes:
        for start in sorted(group):
            if not graph[start]:
                paths.append([start])
            for nxt in graph[start]:
                if edge(start, nxt) in visited:
                    continue
                visited.add(edge(start, nxt))
                path = [start, nxt]
                previous, current = start, nxt
                while current not in owner:
                    options = [q for q in graph[current] if q != previous]
                    assert len(options) == 1
                    nxt = options[0]
                    assert edge(current, nxt) not in visited
                    visited.add(edge(current, nxt)); path.append(nxt)
                    previous, current = current, nxt
                paths.append(path)
    # Closed loops with no endpoint/junction become a single segment.
    for start in sorted(pixels):
        for nxt in graph[start]:
            if edge(start, nxt) in visited: continue
            path = [start, nxt]
            visited.add(edge(start, nxt))
            previous, current = start, nxt
            while current != start:
                options = [q for q in graph[current] if q != previous]
                assert len(options) == 1
                nxt = options[0]
                assert edge(current, nxt) not in visited
                visited.add(edge(current, nxt)); path.append(nxt)
                previous, current = current, nxt
            paths.append(path)
    assert len(visited) == sum(len(v) for v in graph.values()) // 2
    return [np.array([(k//width, k % width) for k in path], np.int32) for path in paths], len(junctions)


def self_check():
    a = np.zeros((30, 30), bool)
    a[4:25, 10] = True
    paths, _ = trace_branches(thin(a))
    assert len(paths) == 1 and len(paths[0]) == 21
    a[12, 10:22] = True
    paths, _ = trace_branches(thin(a))
    assert len(paths) == 3  # T junction splits one connected component into 3 branches.
    ring = np.zeros((20, 20), bool)
    ring[4, 4:15] = ring[14, 4:15] = True
    ring[4:15, 4] = ring[4:15, 14] = True
    assert len(trace_branches(thin(ring))[0]) == 1
    a = np.zeros((30, 30), bool); a[3:26, 10:14] = True
    s = thin(a)
    assert s.sum() < a.sum() and ndi.label(s, np.ones((3, 3)))[1] == 1
    assert len(trace_branches(s)[0]) == 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--counts-dir', type=Path, required=True)
    parser.add_argument('--min-points', type=int, default=5)
    args = parser.parse_args()
    self_check()
    root = args.counts_dir
    counts = json.loads((root/'counts.json').read_text())
    out = root/'branches'
    if out.exists() and any(out.iterdir()): parser.error('Output already exists.')
    out.mkdir(parents=True, exist_ok=True)
    report = dict(model=counts['model'], model_sha256=counts['model_sha256'], epoch=counts['epoch'],
        d0=counts['d0'], threshold=counts['threshold'], tolerance_pixels=counts['tolerance_pixels'],
        minimum_component_pixels=counts['minimum_component_pixels'], minimum_centerline_points=args.min_points,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        definition='Zhang-Suen centerline; trace maximal paths between endpoints and contracted junctions; isolated cycles count once; no bridging of disconnected gaps.',
        matching='ANY centerline point within tolerance of original d0 label mask; non-exclusive matches.',
        self_checks='line, T junction, ring, thick-line connectivity, exhaustive graph-edge accounting passed', results={})
    for name in ['train_seen', 'test_unseen', 'all_exported']:
        with np.load(root/f'{name}_mosaic.npz') as z:
            p, valid, truth, component_id = z['probability'], z['valid'], z['truth'], z['component_id']
            origin = (int(z['row_origin']), int(z['col_origin']))
        sizes = np.bincount(component_id.ravel())
        accepted = sizes >= counts['minimum_component_pixels']; accepted[0] = False
        mask = accepted[component_id]
        skeleton = thin(mask)
        assert ndi.label(mask, np.ones((3, 3)))[1] == ndi.label(skeleton, np.ones((3, 3)))[1]
        paths, junction_pixels = trace_branches(skeleton)
        distance = ndi.distance_transform_edt(~truth) if truth.any() else np.full(truth.shape, np.inf)
        details, pixels_json = [], []
        tol = counts['tolerance_pixels']
        for index, path in enumerate(paths):
            unique = np.unique(path, axis=0)
            dyx = np.diff(path.astype(float), axis=0)
            near = distance[unique[:, 0], unique[:, 1]] <= tol
            minimum = float(distance[unique[:, 0], unique[:, 1]].min())
            details.append(dict(id=index+1, centerline_points=len(unique),
                length_grid_units=float(np.sqrt((dyx**2).sum(axis=1)).sum()),
                parent_component=int(component_id[path[0, 0], path[0, 1]]),
                retained=len(unique) >= args.min_points, matched=bool(near.any()),
                minimum_distance_pixels=minimum if np.isfinite(minimum) else None,
                fraction_near_label=float(near.mean()),
                global_bbox_row_col=[int(unique[:, 0].min()+origin[0]), int(unique[:, 1].min()+origin[1]),
                    int(unique[:, 0].max()+origin[0]+1), int(unique[:, 1].max()+origin[1]+1)]))
            pixels_json.append(dict(id=index+1, global_row_col=(path+np.array(origin)).tolist()))
        retained = [r for r in details if r['retained']]
        sensitivity = []
        for n in sorted({1, 3, 5, 8, 16, args.min_points}):
            for t in [0, 1, 2, 3, 5]:
                selected = [r for r in details if r['centerline_points'] >= n]
                m = sum(r['minimum_distance_pixels'] is not None and r['minimum_distance_pixels'] <= t for r in selected)
                sensitivity.append(dict(min_centerline_points=n, tolerance_pixels=t, predicted=len(selected), matched=m, unmatched=len(selected)-m))
        matched = sum(r['matched'] for r in retained)
        result = dict(predicted=len(retained), matched=matched, unmatched=len(retained)-matched,
            raw_branches=len(details), raw_matched=sum(r['matched'] for r in details),
            discarded_short_branches=len(details)-len(retained),
            majority_centerline_matched=sum(r['fraction_near_label'] >= .5 for r in retained),
            skeleton_pixels=int(skeleton.sum()), junction_pixels=junction_pixels, sensitivity=sensitivity)
        assert result['predicted'] == result['matched'] + result['unmatched']
        report['results'][name] = result
        (out/f'{name}_segments.json').write_text(json.dumps(details, indent=2, allow_nan=False), encoding='utf-8')
        (out/f'{name}_paths.json').write_text(json.dumps(pixels_json, separators=(',', ':')), encoding='utf-8')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.lines import Line2D
        from matplotlib.colors import ListedColormap
        fig, ax = plt.subplots(figsize=(12, 11))
        bg = np.where(valid, 0, 2).astype(np.uint8); bg[truth] = 1
        ax.imshow(bg, cmap=ListedColormap(['#edf0f3', '#d6a637', '#ffffff']), vmin=0, vmax=2, origin='lower', interpolation='nearest')
        linepaths, colors = [], []
        for r, path in zip(details, paths):
            if not r['retained']: continue
            linepaths.append(path[:, ::-1]); colors.append('#138559' if r['matched'] else '#cf337b')
        ax.add_collection(LineCollection(linepaths, colors=colors, linewidths=1.1))
        ax.set_title(f'{name}: {result["predicted"]} predicted segments / {result["matched"]} tolerant matches\n'
                     f'Centerline branches >= {args.min_points} points; any point within {tol:g} px of annotation band')
        ax.set_xlabel(f'Column relative to {origin[1]}'); ax.set_ylabel(f'Row relative to {origin[0]}')
        ax.legend(handles=[Line2D([0], [0], color=c, lw=2, label=l) for c, l in [('#d6a637', 'Original annotation'), ('#138559', 'Matched branch'), ('#cf337b', 'Unmatched branch')]], loc='upper right')
        fig.tight_layout(); fig.savefig(out/f'{name}_overview.png', dpi=170); plt.close(fig)
        print(name, {k:v for k,v in result.items() if k != 'sensitivity'}, flush=True)
    (out/'branch_counts.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    names = {'train_seen':'训练区域', 'test_unseen':'未见测试区域', 'all_exported':'全部已导出区域（拼接后重新分段）'}
    lines = [f'# d0={counts["d0"]:g} 预测线段计数', '',
        f'使用已有 {Path(counts["model"]).parent.name} 第{counts["epoch"]}轮概率图；阈值{counts["threshold"]:g}；{counts["results"]["train_seen"]["audit"]["tiles"]}个训练图块、{counts["results"]["test_unseen"]["audit"]["tiles"]}个测试图块，按地理位置拼接并平均重叠概率。', '',
        '## 主要结果', '', '| 范围 | 预测线段 | 大致匹配标签 | 未匹配 |', '|---|---:|---:|---:|']
    for name, r in report['results'].items(): lines.append(f'| {names[name]} | {r["predicted"]} | {r["matched"]} | {r["unmatched"]} |')
    lines += ['', '## 计数与宽松匹配规则', '',
        f'- 概率≥{counts["threshold"]:g}，去掉小于{counts["minimum_component_pixels"]}像素的独立碎点，提取中心线，在端点/交汇点处分段。闭合环计一段，不跨断开的间隙连接。',
        f'- 主要结果保留至少{args.min_points}个中心线点的段，避免将交汇点附近极短的小枝条当作一条。这个门槛是预设操作定义，不是地质真值；所有未过滤短枝条的数量也保留在下表。',
        f'- 只要一段的任一中心线点进入原标签外扩{counts["tolerance_pixels"]:g}像素的范围，即算大致匹配。允许部分重合，不要求整段或逐像素重合。',
        f'- 容差是原d0={counts["d0"]:g}标签带之外额外{counts["tolerance_pixels"]:g}个像素，不是相对原始断层中心线的距离；格距15角秒，南北每像素约0.463公里，东西方向随纬度变小。不同d0的标签带宽不同，不能把匹配数量差异完全归因于模型。',
        '- 同一条标签可以匹配多个预测段；未匹配不能直接解释为假断层。预测线段数量不能解释为独立地质断层数量。',
        '- 全部区域重新拼接后，跨训练/测试边界的段可连接或改变交汇点，因此总数不一定等于两组简单相加。', '',
        f'## 不删除短枝条的计数（仍已排除小于{counts["minimum_component_pixels"]}像素的独立区域）', '', '| 范围 | 全部中心线分支 | 大致匹配 |', '|---|---:|---:|']
    for name, r in report['results'].items(): lines.append(f'| {names[name]} | {r["raw_branches"]} | {r["raw_matched"]} |')
    lines += ['', '## 输出与验证', '',
        '- branch_counts.json：完整口径及1/3/5/8/16点长度门槛、0/1/2/3/5像素容差的敏感性计数。',
        '- *_segments.json：每段长度、是否匹配、靠近标签比例、位置及所属连通区域。',
        '- *_paths.json：每段在原始全局栅格上的中心线坐标，可重画核对。',
        '- *_overview.png：金色为原标签、绿色为匹配段、粉色为未匹配段。',
        '- 父目录保存连通区域计数，作为对照。存在庞大的相连预测网络，因此连通区域数会低估其中线段数。',
        '- 已通过直线、T形交汇、闭环、宽线细化的检查；实际数据核验细化前后连通区域数一致、所有图边恰好被追踪或归入交汇点。', '']
    (out/'线段计数说明.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
