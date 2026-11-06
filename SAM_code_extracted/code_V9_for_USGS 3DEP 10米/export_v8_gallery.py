"""Read-only V8 full-tile gallery from best Student and EMA checkpoints."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import h5py
import numpy as np
from PIL import Image
import torch
from evaluate_cv import inputs_for_tile, split_audit
from export_bceft_galleries import digest, predict, panel, stats, STYLE
from sam2_lora import load_run_model

HERE = Path(__file__).resolve().parent
RUN = HERE / 'runs/v8_california_ds6_land12km_bceft_lora8_ema_unimatch_seed7_r02'
LABELS = {'train': '训练集', 'val': '验证集（选模）', 'test': '测试集（仅展示）'}


def page(out, report):
    rows = json.dumps(report['images'], ensure_ascii=False).replace('<', '\\u003c')
    opts = ''.join(f'<option value="{s}">{LABELS[s]} · {n} 张</option>' for s, n in report['counts'].items())
    student, teacher = report['epochs']['student'], report['epochs']['teacher']
    text = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>V8 · USGS陆地断层逐图图库</title><style>{STYLE} .facts{{display:flex;gap:10px;flex-wrap:wrap}}.facts span{{background:#e8f3ee;padding:5px 12px;border-radius:6px}}nav{{position:sticky;top:0;background:#f1f5f9;padding:10px 0;z-index:2}}.legend{{font-size:14px}}article h3{{margin-top:0}}</style>
<main><header><h1>V8 · USGS陆地断层逐图图库</h1>
<div class="facts"><span>12 km 监督 mask</span><span>训练完成 500 轮</span><span>Student：第 {student} 轮</span><span>EMA：第 {teacher} 轮</span><span>{len(report['images'])} 个图块</span></div>
<p>采用各自已知陆地验证损失最低的检查点；SAM-2 冻结，LoRA r8 / α16，seed=7。EMA 教师从第250轮创建，伪标签和一致性权重在第350轮达到最大。训练已结束，本图库只执行推理。</p>
<p>每张九宫格展示地形、标准化高程、USGS标签、监督mask、Student/EMA概率、两者阈值预测及Student与标签对照。完整384×384图块采用9个256×256滑窗平均，阈值固定0.5。图片北向上；NPZ保留原始栅格行序（由南向北）。</p>
<p class="warn">仅 mask 内的陆地参与标签对照。海洋及其他未知区域的预测面积不是准确率；超出目录标签的预测不能直接认定为误报或新断层。训练与验证存在原有空间重叠，逐图滑窗统计也不等同于训练日志的固定裁剪指标。</p>
<p class="legend">标签图：粉色=USGS标签，绿线=监督边界。对照图：绿色=预测与标签重合，黄色=漏掉已有标签，粉色=超出标签，灰色=未知。概率图统一采用0–1色标。</p>
<p><a href="manifest.json">完整模型来源与核验报告</a> · <a href="model_config.json">训练配置快照</a></p></header>
<nav><select id="split"><option value="all">全部集合</option>{opts}</select><select id="domain"><option value="all">全部地形</option><option value="sea">包含海洋</option><option value="unknown">全图无直接监督</option><option value="land">包含陆地监督</option></select><input id="search" placeholder="搜索图块编号，如 t_512"><button id="prev">上一页</button><button id="next">下一页</button><span id="count" aria-live="polite"></span></nav><section id="cards"></section></main>
<script>const rows={rows};let page=0;const limit=12;const el=id=>document.getElementById(id);const fmt=x=>x===null?'无可评估标签':x.toFixed(3);const pct=x=>x===null?'—':(x*100).toFixed(1)+'%';
function render(){{const s=el('split').value,d=el('domain').value,q=el('search').value.trim().toLowerCase();const filtered=rows.filter(r=>(s==='all'||r.split===s)&&r.key.toLowerCase().includes(q)&&(d==='all'||(d==='sea'&&r.marine_pixels>0)||(d==='unknown'&&r.known_pixels===0)||(d==='land'&&r.known_pixels>0)));const pages=Math.ceil(filtered.length/limit);page=Math.max(0,Math.min(page,pages-1));el('cards').innerHTML=filtered.slice(page*limit,(page+1)*limit).map(r=>`<article><h3>${{r.key}} · ${{r.split_label}}</h3><p>中心：${{r.lat.toFixed(3)}}°N，${{r.lon.toFixed(3)}}°E · 监督覆盖 ${{pct(r.known_pixels/r.valid_pixels)}}</p><a href="${{r.file}}" target="_blank"><img loading="lazy" src="${{r.file}}" alt="${{r.key}} 九宫格预测对比"></a><p>mask内F1：Student ${{fmt(r.student.known_F1)}} / EMA ${{fmt(r.teacher.known_F1)}}</p><p>海洋预测正类面积：Student ${{pct(r.student.marine_predicted_fraction)}} / EMA ${{pct(r.teacher.marine_predicted_fraction)}}（非准确率）</p><p><a href="${{r.file}}" target="_blank">查看原尺寸图片</a> · <a href="${{r.npz}}" download>下载概率、标签及mask</a></p></article>`).join('')||'<p>没有符合条件的图块。</p>';el('count').textContent=`${{filtered.length}} 张 · 第 ${{pages?page+1:0}} / ${{pages}} 页`;el('prev').disabled=page===0;el('next').disabled=(page+1)*limit>=filtered.length;}}
for(const id of ['split','domain','search'])el(id).addEventListener(id==='search'?'input':'change',()=>{{page=0;render()}});el('prev').onclick=()=>{{page--;render()}};el('next').onclick=()=>{{page++;render()}};render();</script></html>'''
    (out / 'index.html').write_text(text, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=RUN / 'gallery_best_student_ema')
    parser.add_argument('--limit', type=int, default=0, help='Optional per-split smoke limit')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.limit < 0 or args.output.exists():
        parser.error('Invalid limit or output already exists; refusing overwrite')
    torch.set_num_threads(4); torch.manual_seed(7)
    config = json.loads((RUN / 'model_config.json').read_text())
    assert config['emaStartEpoch'] == 250 and config['supervisionKey'] == 'label_known'
    assert not config['knownCrop'] and config['epochs'] == 500
    history = [json.loads(line) for line in (RUN / 'adaptation_history.jsonl').read_text().splitlines()]
    assert [r['epoch'] for r in history] == list(range(1, 501))
    assert all(r['teacher_active'] == (r['epoch'] >= 250) for r in history)
    assert json.loads((RUN / 'launch.json').read_text())['status'] == 'complete'
    data = HERE / 'earth_dataset_V6'
    models = {name: RUN / f'best_{name}/model.pth' for name in ('student', 'teacher')}
    epochs = {name: json.loads((path.parent / 'metrics.json').read_text())['epoch'] for name, path in models.items()}
    sources = {name: digest(path) for name, path in models.items()}
    data_hash = digest(data / 'faults.h5')
    assert data_hash == json.loads((data / 'qa/validation_report.json').read_text())['dataset_hdf5_sha256']
    print(f'Loading Student e{epochs["student"]}, EMA e{epochs["teacher"]}...', flush=True)
    nets = {name: load_run_model(path, args.device, config=config) for name, path in models.items()}
    splits = {s: (data / f'key_{s}.lst').read_text().split() for s in LABELS}
    assert len(set().union(*splits.values())) == sum(map(len, splits.values())) == 216
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(RUN / 'model_config.json', out / 'model_config.json')
    report = dict(status='exporting', experiment='V8', epochs=epochs, completed_epochs=500,
                  created_utc=datetime.now(timezone.utc).isoformat(), model_sha256=sources,
                  models={n: str(p) for n, p in models.items()}, dataset=str(data), dataset_hdf5_sha256=data_hash,
                  threshold=.5, d0=.01, full_export=not args.limit, images=[], counts={},
                  protocol='Full 384px tiles; nine overlapping 256px windows averaged; known-only label comparison; north-up PNG, south-to-north NPZ rows',
                  training_started_by_export=False)
    with h5py.File(data / 'faults.h5', 'r') as f:
        report['split_audit'] = split_audit(f, splits)
        with ThreadPoolExecutor(max_workers=2) as writer:
            pending = []
            for split, keys in splits.items():
                dest = out / split; dest.mkdir()
                chosen = keys[:args.limit] if args.limit else keys
                for index, key in enumerate(chosen):
                    g = f[key]
                    image, distance, valid = inputs_for_tile(g)
                    assert valid.any() and np.isfinite(image).all()
                    mask = g['label_known'][:]
                    assert np.isin(mask, [0, 1]).all()
                    known = (mask > .5) & valid
                    truth = (distance < .01) & valid
                    marine = (g['image'][:] < 0) & valid
                    assert not (known & marine).any() and not (truth & ~known).any()
                    probs = {name: predict(net, image, args.device) for name, net in nets.items()}
                    np.savez_compressed(dest / f'{key}.npz', student=probs['student'], teacher=probs['teacher'],
                                        truth=truth, valid=valid, known=known, marine=marine,
                                        lat_center=g.attrs['lat_c'], lon_center=g.attrs['lon_c'], threshold=.5, d0=.01)
                    pending.append(writer.submit(panel, dest / f'{key}.png', image[:, ::-1], truth[::-1], valid[::-1],
                                                 known[::-1], probs['student'][::-1], probs['teacher'][::-1],
                                                 f'V8 | {key} | {split} | Student e{epochs["student"]} / EMA e{epochs["teacher"]} | North up', True))
                    if len(pending) >= 4: pending.pop(0).result()
                    row = dict(key=key, split=split, split_label=LABELS[split],
                               file=f'{split}/{key}.png', npz=f'{split}/{key}.npz',
                               lat=float(g.attrs['lat_c']), lon=float(g.attrs['lon_c']),
                               valid_pixels=int(valid.sum()), known_pixels=int(known.sum()), marine_pixels=int(marine.sum()))
                    for name, p in probs.items():
                        row[name] = stats(p, truth, valid, known)
                        row[name]['marine_predicted_fraction'] = float((p[marine] >= .5).mean()) if marine.any() else None
                    report['images'].append(row)
                    if (index + 1) % 20 == 0 or index + 1 == len(chosen):
                        print(f'{split}: {index + 1}/{len(chosen)}', flush=True)
                        (out / 'progress.json').write_text(json.dumps({'status':'exporting','completed':len(report['images']),'split':split}), encoding='utf-8')
            for future in pending: future.result()
    print('Verifying every image and prediction array...', flush=True)
    for row in report['images']:
        with Image.open(out / row['file']) as image: image.verify()
        with np.load(out / row['npz']) as arrays:
            valid, known, truth, marine = (arrays[k] for k in ('valid', 'known', 'truth', 'marine'))
            assert not (known & marine).any()
            for name in nets:
                p = arrays[name]
                assert p.shape == (384, 384) and np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1
                assert all(row[name][k] == v for k, v in stats(p, truth, valid, known).items())
    assert {name: digest(path) for name, path in models.items()} == sources
    assert digest(data / 'faults.h5') == data_hash
    report['counts'] = dict(Counter(r['split'] for r in report['images']))
    if not args.limit: assert report['counts'] == {'train':161, 'val':17, 'test':38}
    report.update(status='complete', verified_panels=len(report['images']), inputs_and_weights_unchanged=True)
    (out / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    page(out, report)
    (out / 'progress.json').write_text(json.dumps({'status':'complete','completed':len(report['images'])}), encoding='utf-8')
    shutil.copy2(Path(__file__), out / Path(__file__).name)
    print(json.dumps({'status':'complete','epochs':epochs,'counts':report['counts'],'index':str(out/'index.html')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
