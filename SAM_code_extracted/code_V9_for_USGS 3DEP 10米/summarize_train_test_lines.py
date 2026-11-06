"""Assemble train/test counts, preserving each version's original label definition."""
import json
from pathlib import Path


def main():
    base = Path(__file__).resolve().parent
    out = base/'three_version_line_comparison'
    train = json.loads((out/'train_comparison.json').read_text())
    test = json.loads((out/'comparison.json').read_text())
    names = ['① 原在线补标', '② 在线补标 + UniMatch', '③ UniMatch + d0=0.01']
    assert train['tiles'] == 161
    assert test.get('tiles', test.get('test_tiles')) == 38
    lines = ['# 三版本训练区与未见测试区线段汇总', '',
             '使用原 best_student：①第186轮、②第155轮、③第174轮。训练区161个图块，地理去重后2,703,360个有效像素；未见测试区38个图块，1,290,240个有效像素。', '',
             '本次补齐②训练区的原模型推理；①③训练区及三组测试区均复用已有概率图。使用原始目录标签，不用在线补出的标签评分，没有重新训练。', '',
             '## 统一口径比较', '',
             '概率阈值0.5；独立预测区域≥16像素；提取中心线并在端点/交汇点处分段，保留≥5个中心线点的段。三组统一以d0=0.02原标签带外加3像素容差匹配。', '',
             '“≥50%匹配”：至少一半中心线点落入标签容差范围。“宽松匹配”：任意中心线点进入即可，包含前一种匹配。', '',
             '| 范围 | 版本 | 预测线段 | ≥50%匹配 | 匹配比例 | 宽松匹配 | 完全未匹配 |',
             '|---|---|---:|---:|---:|---:|---:|']
    for region, payload in [('训练区', train), ('未见测试区', test)]:
        for name, r in zip(names, payload['results']):
            assert r['predicted'] == r['majority']+r['partial_only']+r['unmatched']
            lines.append(f'| {region} | {name} | {r["predicted"]} | {r["majority"]} | {r["majority_fraction"]:.1%} | {r["loose"]} | {r["unmatched"]} |')
    lines += ['', '## 训练区原标签口径数据', '',
              '保留每个版本自身的标签定义：①②为d0=0.02，③为d0=0.01。其他计数规则不变。本表用于对应历史原结果；标签宽度不同，应以上面的统一口径作横向比较。', '',
              '| 版本 | 自身标签d0 | 预测线段 | ≥50%匹配 | 匹配比例 | 宽松匹配 | 完全未匹配 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    native = []
    for name, r in zip(names, train['results']):
        d0 = .01 if r['version'] == 'unimatch_d001' else .02
        s = next(v for v in r['sensitivity'] if v['label_d0'] == d0 and v['tolerance_pixels'] == 3)
        native.append(dict(version=r['version'], **s))
        lines.append(f'| {name} | {d0} | {s["predicted"]} | {s["majority"]} | {s["majority"]/s["predicted"]:.1%} | {s["loose"]} | {s["predicted"]-s["loose"]} |')
        if r['version'] != 'unimatch':
            prior = json.loads((base/r['folder']/'segment_counts/branches/branch_counts.json').read_text())['results']['train_seen']
            assert s['predicted'] == prior['predicted']
            assert s['loose'] == prior['matched']
            assert s['majority'] == prior['majority_centerline_matched']
    lines += ['', '## 如何解释', '',
              '- 统一口径下，训练区≥50%匹配数量①19段、②16段、③12段；测试区分别4段、10段、6段。训练区拟合与测试区表现应分别报告。',
              '- 训练和测试地理范围与样本量不同，不能直接用两个区域的线段数量差值衡量过拟合。',
              '- 预测线段不是独立地质断层；同一条断层可被分成多个段，数量受碎裂和交汇点影响。',
              '- 未匹配不代表已确认错误，可能包括漏标；即使≥50%匹配也只是带容差的目录一致性，不是人工认证。', '',
              '## 来源与原始明细', '',
              '- train_comparison.json / comparison.json：输入概率图路径与哈希、统一规则、各容差和标签宽度下的计数。',
              '- *_train_segments.json / *_test_segments.json：每条线段的中心线坐标、点数、匹配比例，图中编号对应明细。',
              '- train_v5_unimatch/train_line_evaluation/train_seen_mosaic.npz：本次补做的②训练区域原概率、原标签、有效掩膜、重叠覆盖数和栅格原点；同目录provenance.json保存模型和数据哈希。',
              '- 已核验训练图块ID、三组覆盖及栅格原点相同；①②标签逐像素一致；①③自身标签口径统计与原历史分段统计一致；分段自检通过。', '',
              '![训练区线段对比](train_line_comparison.png)', '',
              '![未见测试区线段对比](test_line_comparison.png)', '']
    (out/'三版本训练与测试线段汇总.md').write_text('\n'.join(lines), encoding='utf-8')
    (out/'train_test_summary.json').write_text(json.dumps(dict(train=train, test=test, native_train=native), indent=2), encoding='utf-8')
    print(json.dumps(native, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
