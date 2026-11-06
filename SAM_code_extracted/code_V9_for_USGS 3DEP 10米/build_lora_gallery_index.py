"""Validate the two completed LoRA exports and build their shared gallery index."""
import json
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
TZ = timezone(timedelta(hours=8))
RUNS = {
    'V5': 'train_v5_unimatch_lora_r8_d001_seed7',
    'V6': 'train_greece_nz_unimatch_lora_r8_d001_seed7',
}


def main():
    summaries = []
    for label, name in RUNS.items():
        run = HERE / name
        folder = run / 'predictions_seen_unseen'
        report = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
        assert report['status'] == 'complete'
        assert len(report['images']) == sum(report['counts'].values())
        positive = valid = saturated = all_positive = 0
        for row in report['images']:
            stem = folder / row['split'] / row['key']
            with Image.open(stem.with_suffix('.png')) as panel:
                panel.verify()
            with np.load(stem.with_suffix('.npz')) as arrays:
                mask = arrays['valid'].astype(bool)
                probability = arrays['probability']
                prediction = arrays['prediction'].astype(bool)
                assert probability.shape == mask.shape == prediction.shape
                assert np.isfinite(probability).all()
                assert np.array_equal(prediction, (probability >= report['threshold']) & mask)
                n = int(mask.sum())
                p = int(prediction.sum())
                assert n == row['valid_pixels'] and p == row['predicted_pixels']
                assert n > 0
                positive += p
                valid += n
                saturated += p / n >= .99
                all_positive += p == n
        if label == 'V5':
            completion_file = run / 'preview/epoch_0500/metrics.json'
        else:
            completion_file = HERE / 'v6_after_v5.log'
        completed = datetime.fromtimestamp(completion_file.stat().st_mtime, TZ).isoformat(timespec='seconds')
        summaries.append(dict(model=label, run=name, best_student_epoch=report['epoch'],
                              counts=report['counts'], panel_count=len(report['images']),
                              saturated_tiles=int(saturated), all_positive_tiles=int(all_positive),
                              predicted_fraction=positive / valid, completion_artifact_time=completed,
                              completion_time_source=str(completion_file.relative_to(HERE)),
                              completion_time_note='Last output write time; approximate completion, not process exit timestamp.'))
        page = folder / 'index.html'
        document = page.read_text(encoding='utf-8')
        if 'id="lora-export-summary"' not in document:
            notice = (f'<div id="lora-export-summary"><p><a href="../../lora_galleries/index.html">返回 V5 / V6 图库总览</a></p>'
                      f'<h2>{label} · 冻结 SAM-2 + LoRA r8 · 最佳 Student 第 {report["epoch"]} 轮</h2>'
                      f'<p style="color:#a63228"><strong>逐图核验：</strong>{all_positive}/{len(report["images"])} 张图的全部有效像素被预测为断层；'
                      f'{saturated} 张图的预测正类比例至少为 99%。当前预测阈值为 0.5。此结果需用于排查模型退化，不能视为有效断层识别。</p></div>')
            document = document.replace('<header>', '<header>' + notice, 1)
            page.write_text(document, encoding='utf-8')
    out = HERE / 'lora_galleries'
    out.mkdir(exist_ok=True)
    (out / 'verification.json').write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding='utf-8')
    cards = []
    descriptions = {'V5': '加州：训练 161 张 / 独立测试 38 张。验证区与训练区存在空间重叠，未放入本图集。',
                    'V6': '希腊与新西兰：训练 312 张 / 验证 69 张；另含加州独立测试 38 张。'}
    for row in summaries:
        cards.append(f'<article><span class="tag">{row["panel_count"]} 张对比图</span><h2>{row["model"]}</h2>'
                     f'<p>{descriptions[row["model"]]}</p><p>最佳 Student：第 {row["best_student_epoch"]} 轮</p>'
                     f'<p>训练末次输出：<time>{escape(row["completion_artifact_time"].replace("T", " "))}</time></p>'
                     f'<p class="warning">全图预测为正类：{row["all_positive_tiles"]} / {row["panel_count"]} 张</p>'
                     f'<a class="button" href="../{row["run"]}/predictions_seen_unseen/index.html">打开 {row["model"]} 逐图图库 →</a></article>')
    document = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>V5 / V6 LoRA 逐图图库</title>
<style>body{margin:0;background:#f2f5f8;color:#203344;font:16px/1.8 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1050px;margin:auto;padding:40px 24px}h1{font-size:32px;line-height:1.3}h2{font-size:27px;margin:8px 0}.intro{color:#526575}section{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:24px;margin:28px 0}article{background:white;border:1px solid #d9e2ea;border-radius:16px;padding:26px}.tag{color:#086f77;font-weight:600}.warning{background:#fff0eb;color:#922e21;padding:12px;border-radius:8px}.button{display:inline-block;background:#116a75;color:white;padding:10px 18px;border-radius:8px;text-decoration:none}a{color:#116a75}footer{font-size:14px;color:#526575}</style></head><body><main>
<p class="tag">2026-10-11 · 训练结果核验</p><h1>V5 / V6 逐图预测图库</h1>
<p class="intro">冻结 SAM-2 + LoRA r8 · UniMatch + EMA · d0 = 0.01 · seed = 7<br>使用验证 loss 选定的最佳 Student，预测阈值统一为 0.5。</p>
<p class="warning">两组最佳权重均出现全图预测为断层的退化。图库保留原始概率与掩膜，用于逐图检查；未调整阈值来美化结果。</p>
<section>__CARDS__</section>
<p>每张对比图包含：地形、原始标注、预测概率、预测叠加、标注与预测对照、归一化深度。支持集合筛选、编号搜索、点击放大，以及下载 NPZ 概率和掩膜。</p>
<footer>时间为北京时间（UTC+8），根据最后预览文件或日志的写入时间估计，并非精确进程退出时间。图块内部存在重复地理像素，逐图统计不等于独立样本评估。V6 验证集用于选模型，不能视为独立测试。<br><a href="verification.json">查看导出核验记录</a></footer>
</main></body></html>'''.replace('__CARDS__', ''.join(cards))
    (out / 'index.html').write_text(document, encoding='utf-8')
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
