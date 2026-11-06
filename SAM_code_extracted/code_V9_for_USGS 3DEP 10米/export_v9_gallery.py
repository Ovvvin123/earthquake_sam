"""V9 / Earth_dataset_v7: full 1024px inference and offline per-tile gallery."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import colormaps
import torch
import sam2
from evaluate_cv import inputs_for_tile, split_audit
from sam2_lora import load_run_model

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RUN = ROOT / 'autodl_results/v9_b4_r01'
DATA = HERE.parent / 'SAM code/Earth_dataset_v7'
LABELS = {'train': '训练集', 'val': '验证集（用于选模）', 'test': '测试集（仅展示）'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def metrics(p, truth, valid, known):
    pred = (p >= .5) & valid
    tp = int((pred & truth & known).sum())
    fp = int((pred & ~truth & known).sum())
    fn = int((~pred & truth & known).sum())
    denom = 2*tp+fp+fn
    return dict(tp=tp, fp=fp, fn=fn, known_F1=2*tp/denom if known.any() and denom else None,
                predicted_fraction=float(pred[valid].mean()),
                unknown_predicted_fraction=float(pred[valid & ~known].mean()) if (valid & ~known).any() else None)


def paint(dest, key, split, image, truth, valid, known, probs, epochs):
    """V7 already has north-to-south rows: never flip the arrays."""
    def rgb(a): return (np.clip(a, 0, 1)*255).astype('uint8')
    base = np.repeat(np.clip(image[0]/2+.5, 0, 1)[..., None], 3, axis=-1)
    base[~valid] = .18
    def overlay(mask, color):
        a = base.copy(); a[mask] = a[mask]*.25+np.asarray(color)*.75
        return a
    def heat(p):
        a = colormaps['viridis'](p)[..., :3]; a[~valid] = .18
        return a
    elevation = colormaps['terrain'](np.clip((image[1]+3)/6, 0, 1))[..., :3]
    elevation[~valid] = .18
    views = [('elevation', 'Normalized elevation [-3, 3]', elevation)]
    for ch, az in enumerate((315, 75, 195)):
        a = np.repeat(np.clip(image[ch+4]/2+.5, 0, 1)[..., None], 3, axis=-1)
        a[~valid] = .18
        views.append((f'sun{az}', f'Hillshade: azimuth {az}, altitude 45', a))
    coverage = base*.25; coverage[valid & ~known] = .42; coverage[known] = [.12, .65, .38]
    views.extend([('known', 'Supervision: green=known, gray=unknown', coverage),
                  ('labels', 'Catalog label: k0 < 0.01 (pink)', overlay(truth, [1, .15, .6]))])
    for name, display in [('student', 'Student'), ('teacher', 'EMA')]:
        views.append((name+'_probability', f'Best {display} e{epochs[name]}: probability [0, 1]', heat(probs[name])))
    for name, display in [('student', 'Student'), ('teacher', 'EMA')]:
        pred = (probs[name] >= .5) & valid
        views.append((name+'_prediction', f'{display}: prediction >= 0.5', overlay(pred, [.05, .85, 1])))
    for name, display in [('student', 'Student'), ('teacher', 'EMA')]:
        pred = (probs[name] >= .5) & valid
        a = base*.3; a[valid & ~known] = .42
        a[known & pred & truth] = [.15, .85, .4]
        a[known & ~pred & truth] = [1, .7, .05]
        a[known & pred & ~truth] = [1, .2, .65]
        views.append((name+'_agreement', f'{display} vs catalog: known area only', a))
    size, gap, top, cellh = 384, 14, 68, 414
    canvas = Image.new('RGB', (3*(size+gap)+gap, top+4*cellh+75), '#ffffff')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    heading = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 17)
    draw.text((gap, 10), f'V9 | {key} | {split} | Student e{epochs["student"]} / EMA e{epochs["teacher"]}', font=heading, fill='#173448')
    draw.text((gap, 38), '30 m / full 1024 x 1024 inference / North up / preview panels downsampled to 384 px', font=font, fill='#526575')
    for i, (name, label, a) in enumerate(views):
        x, y = gap+i%3*(size+gap), top+i//3*cellh
        draw.text((x, y), label, font=font, fill='#173448')
        im = Image.fromarray(rgb(a))
        im.save(dest/f'{name}.png', compress_level=3)
        canvas.paste(im.resize((size, size), Image.Resampling.NEAREST), (x, y+24))
    y = top+4*cellh
    draw.text((gap, y), 'Comparison: green=overlap, yellow=missed catalog, pink=outside catalog, gray=unknown.', font=font, fill='#334455')
    draw.text((gap, y+22), 'Incomplete labels: outside-catalog predictions are NOT confirmed errors or new faults.', font=font, fill='#334455')
    ramp = np.tile(np.linspace(0, 1, 240), (12, 1))
    canvas.paste(Image.fromarray(rgb(colormaps['viridis'](ramp)[..., :3])), (880, y+45))
    draw.text((860, y+40), '0', font=font, fill='black'); draw.text((1130, y+40), '1', font=font, fill='black')
    canvas.save(dest/'comparison.jpg', quality=94, subsampling=0)
    thumb = Image.fromarray(rgb(views[9][2])); thumb.thumbnail((240, 240)); thumb.save(dest/'thumb.jpg', quality=85)


def page(out, report):
    data = json.dumps(report['images'], ensure_ascii=False).replace('<', '\\u003c')
    choices = [('comparison', '十二宫格总览'), ('student_probability', 'Student 概率'), ('teacher_probability', 'EMA 概率'),
               ('student_prediction', 'Student 预测叠加'), ('teacher_prediction', 'EMA 预测叠加'),
               ('student_agreement', 'Student 与标注对照'), ('teacher_agreement', 'EMA 与标注对照'),
               ('labels', '目录标注'), ('known', '监督范围'), ('elevation', '标准化高程'),
               ('sun315', '太阳方位 315°'), ('sun75', '太阳方位 75°'), ('sun195', '太阳方位 195°')]
    opts = ''.join(f'<option value="{v}">{t}</option>' for v, t in choices)
    doc = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V9 · 逐图预测图库</title>
<style>*{box-sizing:border-box}body{margin:0;background:#eef2f5;color:#193344;font:15px/1.65 system-ui,"Microsoft YaHei",sans-serif}header{padding:25px 32px;background:#123847;color:#edf8fa}h1{margin:0;font-size:28px}header p{max-width:1200px;margin:8px 0}header a{color:#85e5dc}.badge{display:inline-block;padding:3px 10px;background:#275563;border-radius:20px;margin:8px 8px 0 0}nav{position:sticky;top:0;z-index:4;background:#fff;padding:12px 24px;border-bottom:1px solid #c7d6dd;display:flex;gap:10px;flex-wrap:wrap;align-items:center}select,input,button{padding:8px;border:1px solid #b7cbd5;border-radius:6px;background:#fff;color:#193344;font:inherit}button{cursor:pointer}button:disabled{opacity:.4;cursor:default}main{display:grid;grid-template-columns:260px minmax(0,1fr);gap:20px;padding:20px;max-width:1700px;margin:auto}aside{max-height:85vh;overflow:auto;position:sticky;top:78px;align-self:start}.card{display:flex;gap:9px;width:100%;padding:8px;text-align:left;margin-bottom:7px}.card img{width:65px;height:65px;object-fit:cover}.active{border:2px solid #087e86;background:#ddf4ef}article{background:white;padding:22px;border-radius:12px;min-width:0}h2{margin:0 0 6px}#full{width:100%;height:auto;display:block;margin-top:14px}a{color:#067982}.facts{display:flex;gap:10px;flex-wrap:wrap}.facts span{padding:6px 12px;background:#eaf3f3;border-radius:6px}.note{color:#586d77;font-size:14px}.warning{padding:12px;background:#fff5dd;border-radius:6px}#links{margin-top:12px}.empty{padding:30px}@media(max-width:780px){main{grid-template-columns:1fr;padding:10px}aside{position:static;max-height:210px;display:flex;gap:7px}.card{min-width:220px}article{padding:12px}header{padding:20px}nav{padding:10px}}</style></head>
<body><header><h1>V9 · Earth_dataset_v7 逐图预测</h1><span class="badge">30 m · 1024 × 1024</span><span class="badge">340 个图块</span><span class="badge">Student 第20轮 · EMA 第250轮</span><span class="badge">训练完成 500 轮</span>
<p>使用各自验证损失最低的模型，完整图块一次推理；阈值固定 0.5。按训练 / 验证 / 测试集浏览，可切换三种太阳方位、标签、概率与预测。单项图片为原始 1024 像素，点击可放大。</p>
<p>北向上；绿色对照表示与目录重合，黄色表示漏掉目录标签，粉色表示超出目录，灰色表示未知。未知区域不参与 F1；目录并非完整真值，超出标注不等于已确认错误或新断层。测试集仅展示，不用于选模或调阈值。</p>
<p><a href="manifest.json">模型来源与逐图统计</a> · <a href="model_config.json">训练配置</a></p></header>
<nav><select id="split"><option value="all">全部集合</option><option value="train">训练集 · 174</option><option value="val">验证集 · 40</option><option value="test">测试集 · 126</option></select><input id="search" placeholder="搜索图块编号"><select id="view">__OPTIONS__</select><button id="prev">← 上一张</button><button id="next">下一张 →</button><span id="count" aria-live="polite"></span></nav>
<main><aside id="cards"></aside><article id="detail"><h2 id="title"></h2><p id="info"></p><div class="facts" id="metrics"></div><p class="note">F1 在 label_known 与有效区域交集内按像素统计，属于目录对照指标；逐图 F1 不等同于训练损失。</p><div id="links"></div><a id="imageLink" target="_blank"><img id="full" alt="当前图块预测对照"></a></article></main>
<script>const rows=__ROWS__;const el=id=>document.getElementById(id);let selected=decodeURIComponent(location.hash.slice(1))||rows[0].key;let filtered=rows;const pct=x=>x===null?'—':(x*100).toFixed(1)+'%';const f1=x=>x===null?'无可评估正类':x.toFixed(3);
function render(){const s=el('split').value,q=el('search').value.toLowerCase().trim();filtered=rows.filter(r=>(s==='all'||r.split===s)&&r.key.toLowerCase().includes(q));if(!filtered.some(r=>r.key===selected))selected=filtered.length?filtered[0].key:'';const i=filtered.findIndex(r=>r.key===selected),r=filtered[i];el('count').textContent=filtered.length?`${i+1} / ${filtered.length} 张`:'0 张';el('prev').disabled=i<=0;el('next').disabled=i<0||i>=filtered.length-1;el('detail').hidden=!r;el('cards').innerHTML=filtered.map(t=>`<button class="card ${t.key===selected?'active':''}" data-key="${t.key}"><img loading="lazy" src="${t.dir}/thumb.jpg" alt="${t.key} 预览"><span>${t.key}<br><small>${t.split_label}</small></span></button>`).join('');for(const b of el('cards').querySelectorAll('button'))b.onclick=()=>{selected=b.dataset.key;render()};if(!r)return;el('title').textContent=r.key+' · '+r.split_label;el('info').textContent=`中心 ${r.lat.toFixed(4)}°N, ${r.lon.toFixed(4)}°E · 有效陆地 ${r.valid_pixels.toLocaleString()} 像素 · 已知监督覆盖 ${pct(r.known_pixels/r.valid_pixels)}`;el('metrics').innerHTML=`<span>Student F1 ${f1(r.student.known_F1)}</span><span>EMA F1 ${f1(r.teacher.known_F1)}</span><span>预测面积 Student ${pct(r.student.predicted_fraction)} / EMA ${pct(r.teacher.predicted_fraction)}</span>`;const v=el('view').value,src=`${r.dir}/${v==='comparison'?'comparison.jpg':v+'.png'}`;el('full').src=src;el('full').alt=r.key+' '+el('view').selectedOptions[0].textContent;el('imageLink').href=src;el('links').innerHTML=`<a href="${src}" target="_blank">打开大图</a> · <a href="${r.dir}/prediction.npz" download>下载原始概率、标签与掩膜</a>`;history.replaceState(null,'','#'+encodeURIComponent(r.key));}
el('split').onchange=render;el('search').oninput=render;el('view').onchange=render;function move(d){let i=filtered.findIndex(r=>r.key===selected)+d;if(i>=0&&i<filtered.length){selected=filtered[i].key;render()}}el('prev').onclick=()=>move(-1);el('next').onclick=()=>move(1);document.addEventListener('keydown',e=>{if(['INPUT','SELECT'].includes(e.target.tagName))return;if(e.key==='ArrowLeft')move(-1);if(e.key==='ArrowRight')move(1)});render();</script></body></html>'''
    doc = doc.replace('__ROWS__', data).replace('__OPTIONS__', opts)
    if len(report['images']) != 340: doc = doc.replace('340 个图块', f'{len(report["images"])} 个图块（试运行）')
    (out/'index.html').write_text(doc, encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=ROOT/'output/v9_per_tile_gallery')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    out = args.output
    if out.exists(): raise FileExistsError(out)
    torch.set_num_threads(4); torch.manual_seed(7)
    config = json.loads((RUN/'model_config.json').read_text())
    assert config['fullTile'] and int(config['mn']) == 1024 and config['poolFirst']
    assert config['computePrecision'] == 'bf16' and config['attentionBackend'] == 'flash'
    assert json.loads((RUN/'launch.json').read_text())['status'] == 'complete'
    snap = next(RUN.glob('auto_finish/*/source_sha256.json'))
    source_hashes = json.loads(snap.read_text())
    for name in ('sam2_lora.py','torchUnet.py','training_memory.py','training_lr.py'):
        assert digest(HERE/name) == source_hashes[name], f'Training source mismatch: {name}'
    vendor = json.loads((HERE/'cloud/vendor/manifest.json').read_text())
    # Match every pinned SAM-2 Python/config file used by the training runtime.
    runtime_hashes = vendor.get('runtime_sha256', vendor.get('source_sha256', {}))
    if not runtime_hashes:
        runtime_hashes = next(v for v in vendor.values() if isinstance(v, dict) and 'build_sam.py' in v)
    for name, expected in runtime_hashes.items():
        assert digest(Path(sam2.__file__).parent/name) == expected, f'SAM-2 mismatch: {name}'
    data_hash = digest(DATA/'faults.h5')
    assert data_hash == '53ee8240f66842a45f8cb541e2cd6c624cfa4147043f510f47ec4faecb9ce920'
    assert digest(HERE/'sam2_hiera_small.pt') == '95949964d4e548409021d47b22712d5f1abf2564cc0c3c765ba599a24ac7dce3'
    models = {n: RUN/f'best_{n}/model.pth' for n in ('student','teacher')}
    hashes = {n: digest(p) for n,p in models.items()}
    cloud_manifest = json.loads(next(RUN.glob('auto_finish/*/results_manifest.json')).read_text())
    for n,p in models.items(): assert hashes[n] == cloud_manifest[p.relative_to(RUN).as_posix()]['sha256']
    epochs = {n: json.loads((p.parent/'metrics.json').read_text())['epoch'] for n,p in models.items()}
    dataset_manifest = json.loads((DATA/'manifest.json').read_text())
    assert dataset_manifest['provenance']['row_order'] == 'north_to_south'
    splits = {s: (DATA/f'key_{s}.lst').read_text().split() for s in LABELS}
    assert {s:len(k) for s,k in splits.items()} == {'train':174,'val':40,'test':126}
    assert len(set().union(*splits.values())) == 340
    tiles = {t['key']:t for t in dataset_manifest['tiles']}
    print('Inputs, model weights and inference source hashes verified; loading models.', flush=True)
    nets = {n:load_run_model(p, 'cuda:0', config=config) for n,p in models.items()}
    out.mkdir(parents=True)
    shutil.copy2(RUN/'model_config.json', out/'model_config.json')
    report = dict(status='exporting', dataset='Earth_dataset_v7', experiment='V9', run=str(RUN),
                  created_utc=datetime.now(timezone.utc).isoformat(), epochs=epochs, threshold=.5,
                  dataset_sha256=data_hash, model_sha256=hashes, source_sha256=source_hashes,
                  runtime=dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0)),
                  protocol='Full 1024px single pass, BF16 + Flash attention, north-up arrays and images; known-mask-only metrics',
                  training_started=False, full_export=args.limit==0, images=[])
    start = time.monotonic()
    with h5py.File(DATA/'faults.h5','r') as f, ThreadPoolExecutor(max_workers=3) as writer:
        report['split_audit'] = split_audit(f,splits)
        keys = [(s,k) for s,kk in splits.items() for k in kk]
        if args.limit: keys = keys[:args.limit]
        pending = []
        def finish_one():
            pending.pop(0).result()
        def write_tile(dest,key,split,image,truth,valid,known,probs):
            np.savez_compressed(dest/'prediction.npz', **probs,truth=truth,known=known,valid=valid,
                                threshold=.5,d0=config['d0'],row_order='north_to_south',
                                transform=tiles[key]['transform'],crs='EPSG:3310')
            paint(dest,key,split,image,truth,valid,known,probs,epochs)
        for index,(split,key) in enumerate(keys):
            image,dist,valid = inputs_for_tile(f[key]); assert image.shape == (7,1024,1024)
            assert np.isfinite(image).all() and valid.any()
            known = (f[key]['label_known'][:] > .5) & valid
            truth = (dist < config['d0']) & valid
            tensor = torch.from_numpy(image[None]).to('cuda:0')
            with torch.inference_mode():
                probs = {n:net(tensor)[0,0].float().cpu().numpy() for n,net in nets.items()}
            for p in probs.values(): assert p.shape==(1024,1024) and np.isfinite(p).all() and 0<=p.min()<=p.max()<=1
            del tensor
            dest = out/split/key; dest.mkdir(parents=True)
            pending.append(writer.submit(write_tile,dest,key,split,image,truth,valid,known,probs))
            if len(pending)>=4: finish_one()
            row = dict(key=key,split=split,split_label=LABELS[split],dir=f'{split}/{key}',
                       lat=tiles[key]['lat_c'],lon=tiles[key]['lon_c'],valid_pixels=int(valid.sum()),known_pixels=int(known.sum()))
            row.update({n:metrics(p,truth,valid,known) for n,p in probs.items()})
            report['images'].append(row)
            elapsed = time.monotonic()-start
            progress = dict(predicted=index+1,written_at_least=index+1-len(pending),total=len(keys),elapsed_seconds=round(elapsed,1),
                            estimated_remaining_seconds=round(elapsed/(index+1)*(len(keys)-index-1)),
                            peak_gpu_gib=round(torch.cuda.max_memory_allocated()/1024**3,2))
            save_json(out/'progress.json',progress)
            if index==0 or (index+1)%10==0 or index+1==len(keys): print(json.dumps(progress),flush=True)
        while pending: finish_one()
    print('Checking every exported image and probability array...',flush=True)
    for row in report['images']:
        dest=out/row['dir']
        for p in list(dest.glob('*.png'))+list(dest.glob('*.jpg')):
            with Image.open(p) as im: im.verify()
        with np.load(dest/'prediction.npz',allow_pickle=False) as a:
            for n in nets: assert metrics(a[n],a['truth'],a['valid'],a['known'])==row[n]
    assert {n:digest(p) for n,p in models.items()}==hashes
    report.update(status='complete',counts=dict(Counter(r['split'] for r in report['images'])),
                  verified_tiles=len(report['images']),peak_gpu_gib=round(torch.cuda.max_memory_allocated()/1024**3,2),
                  elapsed_seconds=round(time.monotonic()-start,1))
    save_json(out/'manifest.json',report); page(out,report)
    save_json(out/'progress.json',dict(status='complete',predicted=len(report['images']),
              written=len(report['images']),verified=len(report['images']),total=len(report['images']),
              elapsed_seconds=report['elapsed_seconds'],estimated_remaining_seconds=0,
              peak_gpu_gib=report['peak_gpu_gib']))
    shutil.copy2(__file__,out/Path(__file__).name)
    print(json.dumps({'status':'complete','tiles':len(report['images']),'index':str(out/'index.html')}),flush=True)


if __name__=='__main__': main()
