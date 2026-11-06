"""Render fixed-scale prediction comparisons from run_masked_loss_ablation outputs."""
import argparse
import html
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def build_gallery(root):
    root=Path(root)
    report=json.loads((root/'report.json').read_text())
    data=np.load(root/'samples.npz')
    arms=list(report['arms'])
    predictions={a:np.load(root/a/'predictions.npz') for a in arms}
    names={'ft':'Original FT','ft_low_lr':'FT / lower LR','bce':'BCE logits','mixed':'BCE + 0.5 FT'}
    steps=sorted(int(k.split('_')[1]) for k in predictions[arms[0]].files)
    images=root/'comparisons'; images.mkdir(exist_ok=True)
    rows=[]
    for i,key in enumerate(data['keys']):
        known=data['known'][i]; target=data['target'][i]; terrain=data['terrain'][i]
        label=np.full((*known.shape,3),.45)
        label[known]=[.92,.92,.92]; label[known & (target>.5)]=[.85,.05,.2]
        fig,axes=plt.subplots(2,7,figsize=(18,6.8))
        views=[(terrain,'Terrain (normalized)'),(label,'Known: white / fault: red'),
               (predictions[arms[0]]['step_0'][i],'Initialization')]
        views += [(predictions[a][f'step_{steps[-1]}'][i],names[a]) for a in arms]
        for j,(view,title) in enumerate(views):
            for r in range(2):
                ax=axes[r,j]
                if j==0: ax.imshow(view,cmap='terrain',origin='lower')
                elif j==1: ax.imshow(view,origin='lower')
                else:
                    show=view if r==0 else view>=.5
                    ax.imshow(show,cmap='magma',vmin=0,vmax=1,origin='lower')
                    if known.any() and not known.all():
                        ax.contour(known,levels=[.5],colors=['cyan'],linewidths=.6,origin='lower')
                ax.set_title(title if r==0 else ('Binary p >= 0.5' if j>=2 else title),fontsize=9)
                ax.axis('off')
        fig.suptitle(f'{data["split"][i]} / {key} | update {steps[-1]} | probability scale 0..1 | cyan = label_known boundary')
        fig.subplots_adjust(left=.01,right=.99,bottom=.03,top=.88,wspace=.06,hspace=.25)
        fig.savefig(images/f'{key}.png',dpi=130); plt.close(fig)
        fig,axes=plt.subplots(len(arms),len(steps),figsize=(11,10),squeeze=False)
        for a,arm in enumerate(arms):
            for b,step in enumerate(steps):
                ax=axes[a,b]
                ax.imshow(predictions[arm][f'step_{step}'][i],cmap='magma',vmin=0,vmax=1,origin='lower')
                if known.any() and not known.all():
                    ax.contour(known,levels=[.5],colors=['cyan'],linewidths=.6,origin='lower')
                ax.set_title(f'{names[arm]} / {step}',fontsize=9); ax.axis('off')
        fig.suptitle(f'{key}: evolution, fixed probability scale 0..1')
        fig.tight_layout(); fig.savefig(images/f'{key}_evolution.png',dpi=110); plt.close(fig)
        rows.append(f'<h2>{html.escape(str(data["split"][i]))} / {html.escape(str(key))}</h2>'
                    f'<a href="comparisons/{key}.png"><img loading="lazy" src="comparisons/{key}.png"></a>'
                    f'<details><summary>0 / 20 / 50 / 100 更新过程</summary>'
                    f'<img loading="lazy" src="comparisons/{key}_evolution.png"></details>')
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    metric_names=[('f1','F1, threshold 0.5'),('prediction_positive_fraction','Predicted fault fraction'),
                  ('probability_separation','Mean p(fault) - mean p(background)')]
    for arm in arms:
        evaluations=[json.loads(line) for line in (root/arm/'evaluation.jsonl').read_text().splitlines()]
        for r,split in enumerate(('train','val')):
            for c,(metric,title) in enumerate(metric_names):
                ax=axes[r,c]
                ax.plot([e['update'] for e in evaluations],[e[split][metric] for e in evaluations],label=names[arm])
                ax.set_title(f'{split}: {title}'); ax.set_xlabel('Optimizer update'); ax.grid(alpha=.2)
                if metric!='probability_separation': ax.set_ylim(-.02,1.02)
    axes[0,0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(root/'metrics.png',dpi=140); plt.close(fig)
    table=[]
    for arm in arms:
        for split in ('train','val'):
            m=report['arms'][arm]['final'][split]
            precision=f'{m["precision"]:.4f}' if m['precision'] is not None else '—'
            table.append(f'<tr><td>{names[arm]}</td><td>{split}</td><td>{m["f1"]:.4f}</td>'
                         f'<td>{precision}</td>'
                         f'<td>{m["recall"]:.4f}</td><td>{m["prediction_positive_fraction"]:.2%}</td>'
                         f'<td>{m["probability_separation"]:.4f}</td><td>{m["collapse"] or "未发生全正/全负"}</td></tr>')
    document='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>V5 masked loss ablation</title>
<style>body{font-family:system-ui;margin:24px;background:#f6f7fb;color:#16202a}img{width:100%;height:auto;background:white}table{border-collapse:collapse;background:white}td,th{padding:8px;border:1px solid #bbb}p{max-width:1100px;line-height:1.6}summary{cursor:pointer;margin:12px}a{color:#165aaa}</style>
<h1>earth_dataset_v5：四组监督损失对照</h1>
<p>8 个固定训练裁剪、4 个诊断验证裁剪，每组 100 次优化器更新；batch=2 × 累积 4。
固定 seed=7、相同初始化与样本顺序，冻结 SAM-2，训练 LoRA 和自定义网络。
启用 label_known，关闭 EMA、伪标签、一致性训练与数据增强。原损失指 Focal Tversky；混合损失为 BCE + 0.5 × Focal Tversky。</p>
<p>指标仅统计有效 label_known 内的像素。灰色标签区域未知，不能当作负类；预测图仍展示整张裁剪。
青色线为已知范围边界，概率图统一使用 0–1 色阶（黑→紫→橙→黄）。训练/验证有空间重叠，此结果是优化诊断，不代表独立地理泛化能力；几何范围也不等于已证实的完整调查范围。</p>
<p><a href="report.json">完整报告</a> · <a href="selection.json">固定样本与空间重叠记录</a> · <a href="schedule.json">逐次更新样本顺序</a></p>
<table><tr><th>组别</th><th>集合</th><th>F1</th><th>Precision</th><th>Recall</th><th>预测正类占比</th><th>正负类概率差</th><th>状态</th></tr>'''
    document+=''.join(table)+'</table><img src="metrics.png">'+''.join(rows)+'</html>'
    (root/'index.html').write_text(document,encoding='utf-8')


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('directory',type=Path)
    build_gallery(ap.parse_args().directory)
