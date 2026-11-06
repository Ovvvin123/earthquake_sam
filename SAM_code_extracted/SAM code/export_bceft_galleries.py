"""Read-only checkpoint inference: full-tile Student/EMA galleries, no training."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import html
import json
from pathlib import Path
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import h5py
import torch
from matplotlib import colormaps
from scipy.ndimage import binary_erosion
from evaluate_cv import inputs_for_tile, positions, split_audit
from sam2_lora import load_run_model

HERE = Path(__file__).resolve().parent
RUNS = {'V5': 'california_ds5_bceft_lora8_ema_unimatch_seed7_r01',
        'V6': 'greece_nz_bceft_lora8_ema_unimatch_seed7_r01'}
LABELS = {'train': '实际参与训练', 'val': '用于选模的验证集',
          'train_excluded': '原训练清单 · 无有效监督，未训练',
          'val_excluded': '原验证清单 · 无有效监督，未验证',
          'test': '加州测试集'}

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(4*1024*1024), b''): h.update(part)
    return h.hexdigest()

def predict(net, image, device):
    h, w = image.shape[-2:]
    origins = [(y,x) for y in positions(h,256) for x in positions(w,256)]
    total = np.zeros((h,w),np.float32); count = np.zeros((h,w),np.uint8)
    with torch.inference_mode():
        for start in range(0,len(origins),3):
            chunk = origins[start:start+3]
            batch = np.stack([image[:,y:y+256,x:x+256] for y,x in chunk])
            output = net(torch.from_numpy(batch).to(device))[:,0].cpu().numpy()
            if not np.isfinite(output).all() or output.min()<0 or output.max()>1:
                raise ValueError('Invalid probabilities')
            for (y,x), p in zip(chunk,output):
                total[y:y+256,x:x+256] += p; count[y:y+256,x:x+256] += 1
    if not count.all(): raise ValueError('Incomplete full-tile inference')
    return total/count

def rgb(a): return (np.clip(a,0,1)*255).astype(np.uint8)

def panel(path, image, truth, valid, known, student, teacher, title, masked):
    base = np.repeat(np.clip(image[0]/2+.5,0,1)[...,None],3,axis=-1)
    base[~valid] = .25
    edge = known & ~binary_erosion(known) if masked else np.zeros_like(valid)
    def overlay(mask, color):
        a=base.copy(); a[mask]=a[mask]*.25+np.array(color)*.75
        return a
    ann = overlay(truth,[1,.15,.65]); ann[edge]=[.3,1,.2]
    coverage=base*.25; coverage[known]=[.15,.6,.36]; coverage[~valid]=.25
    def heat(p):
        a=colormaps['viridis'](p)[...,:3]; a[~valid]=.25; a[edge]=[.3,1,.2]; return a
    s=(student>=.5)&valid; t=(teacher>=.5)&valid
    agree=base*.35; agree[valid&~known]=[.38,.38,.38]
    agree[known&s&truth]=[.15,.85,.4]
    agree[known&~s&truth]=[1,.7,.05]
    agree[known&s&~truth]=[1,.2,.65]
    depth=colormaps['terrain'](np.clip((image[1]+3)/6,0,1))[...,:3]; depth[~valid]=.25
    plots=[('CTX / terrain',base),('Normalized elevation [-3, 3]',depth),
           ('Catalog faults; green = known boundary',ann),
           ('label_known (green); unknown (gray)' if masked else 'Catalog-background assumption (NOT surveyed mask)',coverage),
           ('Best Student probability [0, 1]',heat(student)),('Best EMA probability [0, 1]',heat(teacher)),
           ('Student prediction >= 0.5',overlay(s,[.05,.8,1])),
           ('EMA prediction >= 0.5',overlay(t,[.05,.8,1])),
           ('Student vs labels (supervised area only)',agree)]
    size=384; gap=14; top=62; cellh=size+30; width=3*(size+gap)+gap
    canvas=Image.new('RGB',(width,top+3*cellh+76),'white'); draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    heading=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    draw.text((gap,12),title,fill='#223344',font=heading)
    draw.text((gap,37),'Full tile / 256-pixel overlapping windows / shared probability scale / d0=0.01',fill='#556677',font=font)
    for i,(label,arr) in enumerate(plots):
        x=gap+(i%3)*(size+gap); y=top+(i//3)*cellh
        draw.text((x,y),label,fill='#223344',font=font)
        canvas.paste(Image.fromarray(rgb(arr)).resize((size,size),Image.Resampling.NEAREST),(x,y+24))
    y=top+3*cellh+4
    draw.text((gap,y),'Agreement: green=overlap; yellow=missed label; pink=outside label; gray=unknown/invalid.',fill='#334455',font=font)
    draw.text((gap,y+22),'Incomplete labels: outside-label predictions are not confirmed errors or new faults.',fill='#334455',font=font)
    gradient=np.tile(np.linspace(0,1,240),(12,1))
    canvas.paste(Image.fromarray(rgb(colormaps['viridis'](gradient)[...,:3])),(width-300,y+42))
    draw.text((width-320,y+38),'0',fill='black',font=font);draw.text((width-55,y+38),'1',fill='black',font=font)
    canvas.save(path,compress_level=3)

STYLE='''body{margin:0;background:#f1f5f9;color:#203344;font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1450px;margin:auto;padding:25px}header,article{background:white;border:1px solid #dce4ec;border-radius:12px;padding:20px}h1{line-height:1.3}nav{display:flex;gap:10px;flex-wrap:wrap;margin:16px 0}input,select,button{padding:9px;font:inherit;border:1px solid #bccbd6;border-radius:6px}section{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,560px),1fr));gap:18px}article img{width:100%}small,.note{color:#526575}a{color:#087589}.warn{background:#fff4de;padding:10px;border-radius:6px}article p{margin:6px 0}'''

def page(out, report):
    data=json.dumps(report['images'],ensure_ascii=False).replace('<','\\u003c')
    counts='；'.join(f'{LABELS[s]} {n} 张' for s,n in report['counts'].items())
    opts=''.join(f'<option value="{s}">{LABELS[s]} ({n})</option>' for s,n in report['counts'].items())
    note=('V5：绿色边界为 label_known；未知区域不计入监督指标。原清单中被过滤的 tile 单独标记，表示该 tile 未进入加载器，不保证其地理范围未被其他训练 tile 覆盖。训练与验证存在空间重叠。'
          if report['label']=='V5' else 'V6：希腊／新西兰训练与验证，加州为跨区域测试。未启用 label_known；指标依赖原始目录背景假设，可能受漏标影响。')
    doc=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{report['label']} 混合损失逐图图库</title><style>{STYLE}</style><main><header>
<a href="../../../bceft_galleries_20261026/index.html">返回 V5 / V6 总览</a>
<h1>{report['label']} · BCE＋0.5×Focal Tversky · 逐图预测</h1>
<p>最佳 Student：第 {report['epochs']['student']} 轮；最佳 EMA：第 {report['epochs']['teacher']} 轮。按各自验证损失选定，不是最后一轮权重。阈值固定 0.5；LoRA r=8、alpha=16，seed=7，d0=0.01。</p>
<p>{counts}。每张包含完整 tile 的九宫格对比，可点击放大并下载原始预测 NPZ。</p>
<p class="warn">{note} 灰色未知区域不能用来判断预测正确与否；粉色表示超出标注，不等于已确认误报或新发现断层。</p>
<p class="note">完整 tile 滑窗预测与训练时固定验证裁剪不同；逐图统计含重复地理像素，不能直接等同于训练日志的验证指标。测试图仅展示，不用于选权重或阈值。</p>
<a href="manifest.json">模型来源、核验记录与完整逐图统计</a></header>
<nav><select id="split"><option value="all">所有集合</option>{opts}</select><input id="search" placeholder="搜索 tile / greece / new_zealand"><button id="prev">上一页</button><button id="next">下一页</button><span id="count"></span></nav><section id="cards"></section></main><script>
const rows={data};let page=0;const limit=24;
function render(){{let q=document.querySelector('#search').value.toLowerCase(),s=document.querySelector('#split').value;let filtered=rows.filter(r=>(s==='all'||r.split===s)&&r.key.toLowerCase().includes(q));page=Math.max(0,Math.min(page,Math.ceil(filtered.length/limit)-1));document.querySelector('#cards').innerHTML=filtered.slice(page*limit,(page+1)*limit).map(r=>`<article><h3>${{r.key}}</h3><p>${{r.split_label}}</p><a href="${{r.file}}" target="_blank"><img loading="lazy" src="${{r.file}}" alt="${{r.key}}"></a><p>有效区域预测正类面积：Student ${{(r.student.predicted_fraction*100).toFixed(2)}}% / EMA ${{(r.teacher.predicted_fraction*100).toFixed(2)}}%</p><p>可监督像素：${{r.known_pixels}}；<a href="${{r.npz}}">下载概率 / 标签 / 掩膜</a></p></article>`).join('');document.querySelector('#count').textContent=`共 ${{filtered.length}} 张 · 第 ${{filtered.length?page+1:0}} / ${{Math.ceil(filtered.length/limit)}} 页`;document.querySelector('#prev').disabled=page===0;document.querySelector('#next').disabled=(page+1)*limit>=filtered.length;}}
document.querySelector('#search').oninput=()=>{{page=0;render()}};document.querySelector('#split').onchange=()=>{{page=0;render()}};document.querySelector('#prev').onclick=()=>{{page--;render()}};document.querySelector('#next').onclick=()=>{{page++;render()}};render();</script></html>'''
    (out/'index.html').write_text(doc,encoding='utf-8')

def stats(prob, truth, valid, known):
    pred=(prob>=.5)&valid; n=int(valid.sum()); tp=int((pred&truth&known).sum())
    fp=int((pred&~truth&known).sum()); fn=int((~pred&truth&known).sum())
    return dict(predicted_fraction=float(pred.sum()/n),probability_min=float(prob[valid].min()),
                probability_max=float(prob[valid].max()),all_positive=bool(pred.sum()==n),
                all_negative=bool(pred.sum()==0),tp=tp,fp=fp,fn=fn,
                known_F1=2*tp/(2*tp+fp+fn) if known.any() and 2*tp+fp+fn else None)

def export(label, suffix, limit):
    run=HERE/'runs'/RUNS[label]; out=run/suffix
    if out.exists(): raise FileExistsError(out)
    config=json.loads((run/'model_config.json').read_text())
    data=Path(config['dataFile']).parent
    models={name:run/('best_'+name)/'model.pth' for name in ('student','teacher')}
    epochs={name:json.loads((path.parent/'metrics.json').read_text())['epoch'] for name,path in models.items()}
    sources={name:digest(p) for name,p in models.items()}
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    nets={name:load_run_model(p,device,config=config) for name,p in models.items()}
    inventory=json.loads((run/'known_crop_inventory.json').read_text()) if label=='V5' else None
    groups=[('train',data),('val',data),('test',data if label=='V5' else HERE/'earth_dataset_v3')]
    out.mkdir()
    report=dict(label=label,status='exporting',epochs=epochs,models={k:str(v) for k,v in models.items()},
                model_sha256=sources,threshold=.5,d0=.01,limit=limit,full_export=not limit,
                protocol='Full-tile 256px overlapping-window mean; known-only label comparison; no test selection',
                counts={},images=[],inputs=[])
    if label=='V5':
        with h5py.File(data/'faults.h5','r') as f:
            report['split_audit']=split_audit(f,{s:(data/f'key_{s}.lst').read_text().split() for s in ('train','val','test')})
    else:
        from prepare_greece_nz import overlap_audit
        report['split_audit']=overlap_audit(json.loads((data/'manifest.json').read_text())['tiles'])
    pending=[]
    with ThreadPoolExecutor(max_workers=2) as writer:
        for split,folder in groups:
            keys=(folder/f'key_{split}.lst').read_text().split()
            if len(keys)!=len(set(keys)): raise ValueError('Duplicate keys')
            if limit: keys=keys[:limit]
            source=folder/'faults.h5'; before=source.stat()
            report['inputs'].append(dict(file=str(source),bytes=before.st_size,mtime_ns=before.st_mtime_ns,split=split))
            with h5py.File(source,'r') as f:
                for i,key in enumerate(keys):
                    actual=split
                    if inventory and split in inventory and key not in inventory[split]['keys']: actual += '_excluded'
                    dest=out/actual;dest.mkdir(exist_ok=True)
                    image,dist,valid=inputs_for_tile(f[key])
                    if not np.isfinite(image).all() or not valid.any(): raise ValueError(f'Invalid input: {key}')
                    truth=(dist<.01)&valid;masked=label=='V5'
                    if masked:
                        mask=f[key]['label_known'][:]
                        if mask.shape!=valid.shape or not np.isin(mask,[0,1]).all(): raise ValueError('Mask mismatch')
                        known=(mask>.5)&valid
                    else: known=valid.copy()
                    probs={name:predict(net,image,device) for name,net in nets.items()}
                    np.savez_compressed(dest/f'{key}.npz',student=probs['student'],teacher=probs['teacher'],
                                        truth=truth,valid=valid,known=known,threshold=.5,d0=.01)
                    title=f"{label} | {key} | {actual} | Student e{epochs['student']} / EMA e{epochs['teacher']}"
                    pending.append(writer.submit(panel,dest/f'{key}.png',image,truth,valid,known,
                                                 probs['student'],probs['teacher'],title,masked))
                    if len(pending)>=4: pending.pop(0).result()
                    row=dict(key=key,split=actual,split_label=LABELS[actual],file=f'{actual}/{key}.png',npz=f'{actual}/{key}.npz',
                             valid_pixels=int(valid.sum()),known_pixels=int(known.sum()),shape=list(valid.shape),
                             **{name:stats(prob,truth,valid,known) for name,prob in probs.items()})
                    report['images'].append(row)
                    if (i+1)%20==0 or i+1==len(keys): print(f'{label} {split}: {i+1}/{len(keys)}',flush=True)
            if (source.stat().st_size,source.stat().st_mtime_ns)!=(before.st_size,before.st_mtime_ns): raise ValueError('Input changed')
        for future in pending: future.result()
    # Read every saved artifact, not just the first few panels.
    for row in report['images']:
        with Image.open(out/row['file']) as im: im.verify()
        with np.load(out/row['npz']) as arrays:
            valid,known,truth=(arrays[k] for k in ('valid','known','truth'))
            assert not (known&~valid).any()
            for name in nets:
                assert arrays[name].shape==valid.shape and np.isfinite(arrays[name]).all()
                assert stats(arrays[name],truth,valid,known)==row[name]
    assert {k:digest(p) for k,p in models.items()}==sources
    report['counts']=dict(Counter(r['split'] for r in report['images']))
    report['summary']={name:dict(all_positive_tiles=sum(r[name]['all_positive'] for r in report['images']),
                               all_negative_tiles=sum(r[name]['all_negative'] for r in report['images'])) for name in nets}
    report.update(status='complete',verified_panels=len(report['images']),inputs_and_weights_unchanged=True)
    (out/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    page(out,report)
    del nets
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(json.dumps(dict(label=label,counts=report['counts'],summary=report['summary'],index=str(out/'index.html'))),flush=True)
    return out,report

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--suffix',default='gallery_best_student_ema')
    args=ap.parse_args()
    if args.limit<0 or Path(args.suffix).name!=args.suffix: ap.error('Invalid arguments')
    if any((HERE/'runs'/name/args.suffix).exists() for name in RUNS.values()): raise FileExistsError('Gallery exists; choose a new suffix')
    torch.set_num_threads(4); torch.manual_seed(7)
    results=[export(label,args.suffix,args.limit) for label in RUNS]
    if not args.limit:
        hub=HERE/'bceft_galleries_20261026';hub.mkdir(exist_ok=False)
        cards=[]
        for out,r in results:
            link=os.path.relpath(out/'index.html',hub)
            cards.append(f'<article><h2>{r["label"]}</h2><p>{len(r["images"])} 张逐图对比 · Student 第 {r["epochs"]["student"]} 轮 / EMA 第 {r["epochs"]["teacher"]} 轮</p><p>全正预测 tile：Student {r["summary"]["student"]["all_positive_tiles"]}；EMA {r["summary"]["teacher"]["all_positive_tiles"]}（阈值 0.5）</p><a href="{html.escape(link)}">打开 {r["label"]} 图库 →</a></article>')
        (hub/'index.html').write_text(f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V5 / V6 混合损失逐图图库</title><style>{STYLE}</style><main><h1>V5 / V6 新训练结果 · 逐图图库</h1><p>2026-10-26 导出 · 冻结 SAM-2 + LoRA r8 / alpha16 · BCE + 0.5 Focal Tversky · EMA + UniMatch</p><p>V5 训练 500 轮、V6 训练 200 轮；各展示最佳 Student 和最佳 EMA。新图库与旧塌缩实验分开保存。</p><section>{"".join(cards)}</section><p class="warn">V5 的 label_known 是批准的几何近似，验证与训练存在空间重叠；V6 原标注仍可能漏标。全正 tile 数不是模型准确率，仍需逐图检查。</p></main></html>',encoding='utf-8')
        (hub/'verification.json').write_text(json.dumps([dict(label=r['label'],counts=r['counts'],summary=r['summary'],verified_panels=r['verified_panels']) for _,r in results],indent=2),encoding='utf-8')
        print('HUB',hub/'index.html',flush=True)

if __name__=='__main__': main()
