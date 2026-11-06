"""Same-tile epoch-500 baseline versus final epoch-600 A/B, Student and EMA.

No best-checkpoint selection: final checkpoints are the primary equal-budget comparison.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import h5py
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from matplotlib import colormaps
from scipy.ndimage import binary_erosion
from sam2_lora import build_model
from export_bceft_galleries import predict, digest, rgb, STYLE, LABELS
from evaluate_cv import inputs_for_tile
from run_v5_feedback_ablation import HERE, SOURCE, NAMES, CONTROL

OUT = HERE/'v5_feedback_comparison_20261028'
GROUPS = {'baseline': SOURCE, **{mode:HERE/'runs'/name for mode,name in NAMES.items()}}
TITLES = {'baseline':'Baseline / epoch 500', 'keep':'A: original feedback / epoch 600',
          'disable':'B: EMA only / epoch 600'}


def measures(p, truth, valid, known):
    pred=(p>=.5)&valid; unknown=valid&~known
    tp=int((pred&truth&known).sum()); fp=int((pred&~truth&known).sum()); fn=int((~pred&truth&known).sum())
    return dict(tp=tp,fp=fp,fn=fn,known_pixels=int(known.sum()),unknown_pixels=int(unknown.sum()),
                unknown_positive=int((pred&unknown).sum()),known_positive=int((pred&known).sum()),
                precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                unknown_positive_fraction=float(pred[unknown].mean()) if unknown.any() else None)


def draw_panel(path, key, split, image, truth, valid, known, probs, epochs):
    base=np.repeat(np.clip(image[0]/2+.5,0,1)[...,None],3,axis=-1); base[~valid]=.25
    edge=known&~binary_erosion(known)
    annotation=base.copy(); annotation[truth]=[1,.15,.65]; annotation[edge]=[.2,1,.3]
    coverage=base*.25; coverage[known]=[.15,.6,.36]
    cells=[('CTX / terrain',base),('Catalog labels / known boundary',annotation),('Known (green) / unknown (dark)',coverage)]
    for group in GROUPS:
        for model in ('student','teacher'):
            p=probs[f'{group}_{model}']
            heat=colormaps['viridis'](p)[...,:3].copy();heat[~valid]=.25;heat[edge]=[.2,1,.3]
            cells.append((f'{group} e{epochs[group]} {model} probability',heat))
        pred=probs[f'{group}_teacher']>=.5
        overlay=base*.5
        overlay[pred&valid]=[.05,.8,1];overlay[truth&known]=[1,.7,.05]
        overlay[pred&truth&known]=[.15,.85,.4];overlay[pred&~truth&known]=[1,.2,.65]
        overlay[edge]=[.2,1,.3]
        cells.append((f'{group}: EMA prediction / labels',overlay))
    size=384;gap=14;ch=418
    canvas=Image.new('RGB',(3*(size+gap)+gap,4*ch+135),'white');draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    draw.text((gap,10),f'V5 | {key} | {split} | threshold=0.5 | d0=0.01',fill='#223344',font=font)
    draw.text((gap,32),'Rows: data / baseline e500 / A keep feedback / B EMA only. All probabilities share [0, 1].',fill='#223344',font=font)
    for i,(label,arr) in enumerate(cells):
        x=gap+i%3*(size+gap);y=64+i//3*ch
        draw.text((x,y),label,fill='#223344',font=font)
        canvas.paste(Image.fromarray(rgb(arr)).resize((size,size),Image.Resampling.NEAREST),(x,y+23))
    draw.text((gap,4*ch+72),'EMA overlay: green=overlap; yellow=missed label; pink=outside label IN known area; cyan=unknown prediction.',fill='#223344',font=font)
    draw.text((gap,4*ch+94),'Unknown area cannot establish errors. Sliding-window results include overlapping geographic pixels.',fill='#223344',font=font)
    canvas.save(path,compress_level=3)


def make_page(out,report):
    data=json.dumps(report['images'],ensure_ascii=False).replace('<','\\u003c')
    options=''.join(f'<option value="{key}">{LABELS[key]} ({value})</option>' for key,value in report['counts'].items())
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V5 继续训练 A/B 对比</title><style>STYLE</style><main><header><h1>V5：第 500 轮基线 → 两组各继续 100 轮</h1>
<p>A：保留 EMA、在线伪标签与一致性；B：保留 EMA，关闭后两项。主对比固定使用第 600 轮，避免最佳轮次不同影响训练时长对照。小规模检查页面除外，以图中实际轮次为准。</p>
<p>每图四行：输入与标签、第 500 轮基线、A、B；后三行依次显示 Student 概率、EMA 概率、EMA 预测与标注对照。阈值 0.5，d0=0.01，同一滑窗推理方式。</p>
<p class="warn">仅 label_known 内计算标注一致性。未知区域预测面积不等于错误率。V5 训练/验证存在空间重叠；原清单中未进入加载器的 tile 单列。测试集仅展示，不参与选模或调参。</p>
<p><a href="manifest.json">检查点、哈希与逐图统计</a> · <a href="training_curves.png">训练/验证对比曲线</a></p></header>
<nav><select id="split"><option value="all">所有集合</option>OPTIONS</select><input id="search" placeholder="搜索 tile"><button id="prev">上一页</button><button id="next">下一页</button><span id="count"></span></nav><section id="cards"></section></main><script>
const rows=DATA;let page=0;const limit=12;
function render(){let q=document.querySelector('#search').value.toLowerCase(),s=document.querySelector('#split').value;let found=rows.filter(r=>(s==='all'||r.split===s)&&r.key.toLowerCase().includes(q));page=Math.max(0,Math.min(page,Math.ceil(found.length/limit)-1));document.querySelector('#cards').innerHTML=found.slice(page*limit,(page+1)*limit).map(r=>`<article><h3>${r.key}</h3><p>${r.split_label}</p><a href="${r.file}" target="_blank"><img loading="lazy" src="${r.file}" alt="${r.key}"></a><p><a href="${r.npz}">下载六组概率及标签、mask</a></p></article>`).join('');document.querySelector('#count').textContent=`共 ${found.length} 张 · 第 ${found.length?page+1:0} / ${Math.ceil(found.length/limit)} 页`;document.querySelector('#prev').disabled=page===0;document.querySelector('#next').disabled=(page+1)*limit>=found.length;}
document.querySelector('#search').oninput=()=>{page=0;render()};document.querySelector('#split').onchange=()=>{page=0;render()};document.querySelector('#prev').onclick=()=>{page--;render()};document.querySelector('#next').onclick=()=>{page++;render()};render();</script></html>'''
    (out/'index.html').write_text(html.replace('STYLE',STYLE).replace('OPTIONS',options).replace('DATA',data),encoding='utf-8')


def curves(out, groups):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for group,folder in groups.items():
        if group=='baseline':continue
        rows=[json.loads(s) for s in (folder/'fixed_val.jsonl').read_text().splitlines() if s.strip()]
        for model,style in [('student','-'),('teacher','--')]:
            for ax,metric in zip(axes.flat,['f1','precision','recall','prediction_positive_fraction']):
                ax.plot([r['epoch'] for r in rows],[r['health'][model][metric] for r in rows],style,label=f'{group} / {model}')
                ax.set(title=metric,xlabel='Epoch');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(out/'training_curves.png',dpi=150);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--b-run',type=Path,help='Explicit fresh B output for an authorized recovery')
    cli=parser.parse_args();torch.set_num_threads(4)
    groups=dict(GROUPS)
    if cli.smoke:
        groups.update({mode:HERE/'runs'/(name+'_smoke') for mode,name in NAMES.items()})
    if cli.b_run:
        if cli.smoke:raise ValueError('Do not combine recovery and smoke outputs')
        groups['disable']=cli.b_run.resolve()
    out=OUT.with_name(OUT.name+'_smoke') if cli.smoke else OUT
    if out.exists():raise FileExistsError(out)
    paths={key:folder/'ema_checkpoint.pth' for key,folder in groups.items()}
    hashes={key:digest(path) for key,path in paths.items()}
    config=json.loads((SOURCE/'model_config.json').read_text())
    source=Path(config['dataFile']); before=source.stat()
    inventory=json.loads((SOURCE/'known_crop_inventory.json').read_text())
    tiles=[]
    for split in ('train','val','test'):
        keys=(source.parent/f'key_{split}.lst').read_text().split()
        if cli.smoke:
            keys=[k for k in keys if split not in inventory or k in inventory[split]['keys']][:1]
        for key in keys:
            actual=split+('_excluded' if split in inventory and key not in inventory[split]['keys'] else '')
            tiles.append((actual,key))
    out.mkdir()
    epochs={};device='cuda:0' if torch.cuda.is_available() else 'cpu'
    for group,path in paths.items():
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        epochs[group]=checkpoint['next_epoch']
        expected=500 if group=='baseline' else 501 if cli.smoke else 600
        if epochs[group]!=expected:raise ValueError(f'{group}: incomplete checkpoint')
        net=build_model(checkpoint['config']).to(device).eval()
        for model,weights in [('student',checkpoint['student']),('teacher',checkpoint['teacher']['model'])]:
            net.load_state_dict(weights,strict=True)
            dest=out/'probabilities'/f'{group}_{model}';dest.mkdir(parents=True)
            with h5py.File(source,'r') as h5:
                for i,(split,key) in enumerate(tiles):
                    image,dist,valid=inputs_for_tile(h5[key])
                    if not np.isfinite(image).all() or not valid.any():raise ValueError(key)
                    p=predict(net,image,device)
                    np.savez_compressed(dest/f'{key}.npz',probability=p)
                    if (i+1)%20==0 or i+1==len(tiles):print(group,model,i+1,len(tiles),flush=True)
        del net,checkpoint,weights
        if torch.cuda.is_available():torch.cuda.empty_cache()
    report=dict(status='exporting',epochs=epochs,checkpoint_sha256=hashes,
                checkpoints={k:str(p) for k,p in paths.items()},threshold=.5,d0=.01,images=[],
                note='Final equal-budget comparison; known-only metrics; unknown area is not error rate.')
    with h5py.File(source,'r') as h5:
        for split,key in tiles:
            image,dist,valid=inputs_for_tile(h5[key]);truth=(dist<.01)&valid
            mask=h5[key]['label_known'][:]
            if mask.shape!=valid.shape or not np.isin(mask,[0,1]).all():raise ValueError('Bad known mask')
            known=(mask>.5)&valid
            probs={}
            for group in groups:
                for model in ('student','teacher'):
                    name=f'{group}_{model}'
                    with np.load(out/'probabilities'/name/f'{key}.npz') as arrays:probs[name]=arrays['probability']
            folder=out/split;folder.mkdir(exist_ok=True)
            np.savez_compressed(folder/f'{key}.npz',**probs,truth=truth,valid=valid,known=known,threshold=.5,d0=.01)
            draw_panel(folder/f'{key}.png',key,split,image,truth,valid,known,probs,epochs)
            report['images'].append(dict(key=key,split=split,split_label=LABELS[split],file=f'{split}/{key}.png',
                npz=f'{split}/{key}.npz',metrics={name:measures(p,truth,valid,known) for name,p in probs.items()}))
    for row in report['images']:
        with Image.open(out/row['file']) as im:im.verify()
        with np.load(out/row['npz']) as arrays:
            for name in row['metrics']:
                p=arrays[name]
                if not np.isfinite(p).all() or p.shape!=arrays['valid'].shape:raise ValueError('Export validation failed')
                if measures(p,arrays['truth'],arrays['valid'],arrays['known'])!=row['metrics'][name]:raise ValueError('Metrics mismatch')
    if hashes!={key:digest(path) for key,path in paths.items()}:raise ValueError('Weights changed')
    if (before.st_size,before.st_mtime_ns)!=(source.stat().st_size,source.stat().st_mtime_ns):raise ValueError('Dataset changed')
    report.update(status='complete',counts=dict(Counter(r['split'] for r in report['images'])),verified_panels=len(tiles))
    (out/'manifest.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    curves(out,groups);make_page(out,report)
    print('GALLERY COMPLETE',out/'index.html',flush=True)


if __name__=='__main__':main()
