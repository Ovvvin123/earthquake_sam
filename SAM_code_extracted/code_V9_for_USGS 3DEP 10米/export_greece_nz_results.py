"""Export the Greece/NZ run's loss figure and full-tile gallery, without training."""
import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import re

import numpy as np

from evaluate_cv import inputs_for_tile
from export_fault_predictions import predict_tile, save_panel
from prepare_greece_nz import DEFAULT_DATA_DIR, HERE, overlap_audit, overlaps


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plot_losses(run, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s=np.loadtxt(run/'loss.txt',ndmin=2)
    t=np.loadtxt(run/'ema_loss.txt',ndmin=2)
    config=json.loads((run/'ema_config.json').read_text())
    best=json.loads((run/'best_metrics.json').read_text())
    assert np.isfinite(s).all() and np.isfinite(t).all()
    assert np.array_equal(s[:,0],np.arange(config['epochs']))
    assert np.array_equal(s[:,0],t[:,0])
    for name,arr in [('student',s),('teacher',t)]:
        row=best[name]
        assert np.isclose(arr[row['epoch']-1,2],row['loss'],atol=2e-7,rtol=0)
    fig,ax=plt.subplots(figsize=(12,6.3))
    ax.axvspan(1,40,color='#e8edf2',alpha=.7,label='Warm-up (epochs 1-40)')
    ax.plot(s[:,0]+1,s[:,1],color='#687d91',lw=1.2,label='Student training loss (total)')
    ax.plot(s[:,0]+1,s[:,2],color='#1664b4',lw=2,label='Student validation loss')
    ax.plot(t[:,0]+1,t[:,2],color='#e48320',lw=1.5,ls='--',label='EMA teacher validation loss')
    ax.axvline(41,color='#585e66',ls='--',lw=1)
    ax.axvline(80,color='#90969d',ls=':',lw=1)
    for name,color in [('student','#1664b4'),('teacher','#e48320')]:
        row=best[name]
        ax.scatter(row['epoch'],row['loss'],marker='*',s=115,color=color,zorder=5)
    ax.set(xlim=(1,config['epochs']),ylim=(0,1),xlabel='Epoch (1-based)',ylabel='Loss',
           title='Greece + New Zealand | UniMatch, d0=0.01 | 312 train / 69 validation tiles')
    ax.grid(alpha=.18)
    ax.legend(loc='lower left',ncol=2,fontsize=9)
    fig.text(.5,.045,'Raw epoch values, no smoothing. Dashed vertical line: activation at epoch 41; dotted: full weight at epoch 80.\n'
             'Training includes consistency loss after activation; validation uses original labels only.\n'
             f"Best student: epoch {best['student']['epoch']} ({best['student']['loss']:.6f}); "
             f"best teacher: epoch {best['teacher']['epoch']} ({best['teacher']['loss']:.6f}). Validation remains near 0.860302.",
             ha='center',fontsize=9,color='#424c58')
    fig.tight_layout(rect=(0,.14,1,1))
    fig.savefig(output/'loss_curve.png',dpi=180)
    plt.close(fig)
    np.savetxt(output/'loss_history.csv',np.column_stack([s[:,0]+1,s[:,1],s[:,2],t[:,2]]),delimiter=',',
               header='epoch,student_train_total,student_validation,ema_teacher_validation',comments='')
    return dict(epochs=len(s),best=best,smoothing=False,
                source_sha256={name:sha(run/name) for name in ['loss.txt','ema_loss.txt','best_metrics.json']})


def save_gallery(out,report,style):
    labels={'train_seen':'希腊／新西兰 · 训练集','validation':'希腊／新西兰 · 验证集',
            'test_unseen':'加州 · 独立测试集'}
    cards=[]
    for row in report['images']:
        split,key=row['split'],row['key']
        key_html=html.escape(key)
        relative=f'{split}/{key}'
        cards.append(f'<article data-split="{split}" data-key="{key_html}"><h3>{key_html} '
                     f'<small>{labels[split]}</small></h3><a href="{relative}.png" target="_blank">'
                     f'<img loading="lazy" src="{relative}.png" alt="{key_html}"></a>'
                     f'<p>预测为断层：{row["predicted_fraction"]:.2%}；原标注：{row["annotated_fraction"]:.2%}。'
                     f'<a href="{relative}.png" target="_blank">完整对比图</a> · '
                     f'<a href="{relative}.npz">概率与掩膜数据</a></p></article>')
    buttons=''.join(f'<button data-filter="{k}">{v}（{report["counts"][k]}）</button>' for k,v in labels.items())
    saturated=sum(row['predicted_fraction']>=.99 for row in report['images'])
    report['tiles_with_at_least_99_percent_predicted_positive']=saturated
    doc=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>希腊＋新西兰 UniMatch 逐图结果</title>
<style>{style}</style><main><header><h1>希腊＋新西兰：UniMatch 逐图结果</h1>
<p>模型：最佳 Student，第 {report['epoch']} 轮，由验证 loss 选定。标签距离阈值 d0=0.01；
预测阈值统一为 0.5，未在测试集上调参。每块 384×384，使用 9 个 256×256 滑窗平均覆盖整块。</p>
<p>训练 {report['counts']['train_seen']} 张，验证 {report['counts']['validation']} 张，加州独立测试
{report['counts']['test_unseen']} 张。<a href="../results/loss_curve.png">打开损失曲线</a> · <a href="manifest.json">模型与导出记录</a></p>
<p style="color:#a63228"><strong>结果提示：</strong>{saturated}/{len(report['images'])} 张图中，至少 99% 的有效像素被预测为断层。
验证 loss 后期约为 0.860302，接近“全部预测为断层”的参照值；最佳轮次不表示已经获得有效识别能力。</p>
<p class="note">绿色：预测与原标注重合；黄色：原标注中未被预测的部分；粉色：预测超出原标注的部分。
标签存在漏标，粉色不能直接认定为错误或真实新断层。各集合内部图块有重叠，张数不等于独立地理样本数。</p>
<p class="note">训练与验证按完整空间块分组，足迹无重叠；验证集用于选模型，不能当作最终独立测试。
加州不参与本次训练或验证，标签来自 USGS；希腊／新西兰标签来自 GEM。图集用于定性检查，不代替正式评估。</p>
</header><nav><button class="active" data-filter="all">全部图像</button>{buttons}
<input id="search" placeholder="搜索图块编号或 greece / new_zealand"><span id="count"></span></nav>
<section>{''.join(cards)}</section></main><script>
let selected='all';const articles=[...document.querySelectorAll('article')];
function update(){{const q=document.querySelector('#search').value.trim().toLowerCase();let n=0;
for(const a of articles){{a.hidden=!((selected==='all'||a.dataset.split===selected)&&a.dataset.key.toLowerCase().includes(q));if(!a.hidden)n++;}}
document.querySelector('#count').textContent='显示 '+n+' 张';}}
document.querySelectorAll('[data-filter]').forEach(b=>b.onclick=()=>{{selected=b.dataset.filter;
document.querySelectorAll('[data-filter]').forEach(x=>x.classList.toggle('active',x===b));update();}});
document.querySelector('#search').oninput=update;update();</script></html>'''
    (out/'index.html').write_text(doc,encoding='utf-8')


def main():
    import h5py
    import torch
    from torchUnet import conv_sam
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=HERE/'train_greece_nz_stride64_unimatch_d001')
    args=parser.parse_args()
    run=args.run_dir.resolve()
    out=run/'predictions_seen_unseen'
    if out.exists() and any(out.iterdir()):
        parser.error('Gallery directory is not empty; existing export is protected.')
    config=json.loads((run/'ema_config.json').read_text())
    assert config['d0']==.01 and config['unimatch'] and config['k0Only']
    data=Path(config['dataset']).parent
    assert data.resolve()==DEFAULT_DATA_DIR.resolve()
    results=run/'results';results.mkdir(exist_ok=True)
    losses=plot_losses(run,results)
    print('Loss figure saved:',results/'loss_curve.png',flush=True)
    metadata=json.loads((run/'best_student/metrics.json').read_text())
    man=json.loads((data/'manifest.json').read_text())
    ca=HERE/'earth_dataset_v3'
    ca_man=json.loads((ca/'manifest.json').read_text())
    test_keys=(ca/'key_test.lst').read_text().split()
    extent=ca_man['provenance']['tile']*ca_man['provenance']['res_deg']
    ca_rows=[dict(t,extent_deg=extent) for t in ca_man['tiles'] if t['key'] in set(test_keys)]
    assert len(ca_rows)==len(test_keys)
    audit=overlap_audit(man['tiles'])
    assert audit['train_val_overlap_pairs']==0
    assert not any(overlaps(a,b) for a in man['tiles'] for b in ca_rows)
    torch.set_num_threads(4);torch.manual_seed(0)
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    model_path=run/'best_student/model.pth'
    from sam2_lora import load_run_model
    net=load_run_model(model_path, device)
    report=dict(model=str(model_path),model_sha256=sha(model_path),epoch=metadata['epoch'],d0=.01,threshold=.5,
                threshold_selection='fixed in advance; not tuned on test',crop=256,windows_per_tile=9,
                device=device,loss_provenance=losses,train_val_audit=audit,california_overlap_pairs=0,counts={},images=[],
                protocol='Full-tile nine-window average; repeated pixels across tiles are not deduplicated metrics.')
    out.mkdir(parents=True,exist_ok=True)
    for name,folder,split in [('train_seen',data,'train'),('validation',data,'val'),('test_unseen',ca,'test')]:
        keys=(folder/f'key_{split}.lst').read_text().split()
        assert keys and len(set(keys))==len(keys)
        destination=out/name;destination.mkdir()
        report['counts'][name]=len(keys)
        with h5py.File(folder/'faults.h5','r') as h5:
            for i,key in enumerate(keys):
                image,distance,valid=inputs_for_tile(h5[key])
                truth=(distance<.01)&valid
                probability=predict_tile(net,image,256,device)
                prediction=(probability>=.5)&valid
                np.savez_compressed(destination/f'{key}.npz',probability=probability,truth=truth,valid=valid,
                                    prediction=prediction,threshold=.5,d0=.01)
                save_panel(destination/f'{key}.png',image,truth,valid,probability,.5,
                           f'{name} | {key} | best student epoch {metadata["epoch"]} | d0=0.01')
                n=int(valid.sum());assert n>0
                report['images'].append(dict(split=name,key=key,shape=list(valid.shape),valid_pixels=n,
                    annotated_pixels=int(truth.sum()),predicted_pixels=int(prediction.sum()),
                    predicted_fraction=float(prediction.sum()/n),annotated_fraction=float(truth.sum()/n),
                    probability_mean=float(probability[valid].mean()),probability_min=float(probability[valid].min()),
                    probability_max=float(probability[valid].max())))
                if (i+1)%10==0 or i+1==len(keys): print(f'{name}: {i+1}/{len(keys)}',flush=True)
    reference=HERE/'train_v5_unimatch/predictions_seen_unseen/index.html'
    style=re.search(r'<style>(.*?)</style>',reference.read_text(encoding='utf-8'),re.S).group(1)
    save_gallery(out,report,style)
    report['status']='complete'
    (out/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'gallery':str(out/'index.html'),'counts':report['counts'],
                     'tiles_at_least_99_percent_positive':report['tiles_with_at_least_99_percent_predicted_positive']},indent=2),flush=True)


if __name__=='__main__':
    main()
