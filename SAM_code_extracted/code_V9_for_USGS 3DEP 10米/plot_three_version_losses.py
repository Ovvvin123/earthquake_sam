"""Plot original epoch losses without smoothing or mixing warm-up runs."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    base = Path(__file__).resolve().parent
    out = base/'three_version_loss_curves'
    out.mkdir(exist_ok=True)
    runs = [('train_v5_online', '1. Online correction (d0=0.02)'),
            ('train_v5_unimatch', '2. Online + UniMatch (d0=0.02)'),
            ('train_v5_unimatch_d001', '3. UniMatch (d0=0.01)')]
    data = []
    provenance = dict(epoch_numbering='loss.txt and ema_loss.txt first column + 1',
                      smoothing=False, source='saved original training logs', runs=[])
    for folder, title in runs:
        root=base/folder
        student=np.loadtxt(root/'loss.txt')
        teacher=np.loadtxt(root/'ema_loss.txt')
        assert len(student)==len(set(student[:,0]))
        assert len(teacher)==len(set(teacher[:,0]))
        assert np.isfinite(student).all() and np.isfinite(teacher).all()
        assert np.array_equal(student[:,0],teacher[:,0])
        best=json.loads((root/'best_metrics.json').read_text())
        for name, array, column in [('student',student,2),('teacher',teacher,2)]:
            saved=array[array[:,0]+1==best[name]['epoch'],column]
            assert len(saved)==1 and np.isclose(saved[0],best[name]['loss'],atol=2e-6,rtol=0)
        data.append((folder,title,student,teacher,best))
        provenance['runs'].append(dict(folder=folder, first_epoch=int(student[0,0]+1),
            last_epoch=int(student[-1,0]+1), best=best,
            source_sha256={f:hashlib.sha256((root/f).read_bytes()).hexdigest()
                           for f in ['loss.txt','ema_loss.txt','best_metrics.json']}))

    def panel(ax, item):
        folder,title,s,t,best=item
        x=s[:,0]+1
        ax.plot(x,s[:,1],color='#78899b',lw=1,alpha=.75,label='Student training loss')
        ax.plot(x,s[:,2],color='#226ab3',lw=1.25,label='Student validation loss')
        ax.plot(t[:,0]+1,t[:,2],color='#df8b27',lw=1.5,label='EMA teacher validation loss')
        ax.axvline(41,color='#707070',ls='--',lw=.8,alpha=.7)
        for name,color in [('student','#226ab3'),('teacher','#df8b27')]:
            r=best[name]
            ax.scatter(r['epoch'],r['loss'],s=52,marker='*',c=color,zorder=5)
        ax.set_title(title+'\nBest student: epoch '+str(best['student']['epoch'])+
                     '; teacher: epoch '+str(best['teacher']['epoch']),fontsize=11)
        ax.set_xlim(0,203); ax.set_ylim(.1,.95)
        ax.set_xlabel('Epoch (1-based)'); ax.set_ylabel('Loss')
        ax.grid(alpha=.18); ax.legend(fontsize=8,loc='upper right')
        if folder=='train_v5_online':
            ax.axvspan(0,40,color='#eeeeee',zorder=-1)
            ax.text(20,.5,'Epochs 1-40\nnot in this log',ha='center',fontsize=8,color='#666666')

    note=('Dashed line: correction/UniMatch begins at epoch 41. Raw values; no smoothing.\n'
          'Training objectives differ with UniMatch; d0=0.01 also changes validation labels. '
          'Loss heights are not a common performance ranking.')
    for item in data:
        fig,ax=plt.subplots(figsize=(10,5.8))
        panel(ax,item)
        fig.text(.5,.025,note,ha='center',fontsize=8,color='#4b5563')
        fig.tight_layout(rect=(0,.09,1,1))
        fig.savefig(out/(item[0]+'_loss.png'),dpi=170)
        plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(18,6),sharey=True)
    for ax,item in zip(axes,data): panel(ax,item)
    fig.suptitle('Original loss histories: three experiment versions',fontsize=17)
    fig.text(.5,.025,note,ha='center',fontsize=10,color='#4b5563')
    fig.tight_layout(rect=(0,.10,1,.94))
    fig.savefig(out/'three_versions_loss.png',dpi=180)
    plt.close(fig)
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    lines=['# 三版本原始 loss 曲线','',
           '- 灰色：Student训练loss；蓝色：Student验证loss；橙色：EMA Teacher验证loss。',
           '- 原始逐轮数据，不平滑。横轴已由日志的0起始编号转换为1起始轮次。星号表示保存的最佳模型轮次。',
           '- 第一版当前目录只有第41–200轮日志；不把另一个预热目录的曲线拼进来，避免混用不同验证流程。',
           '- 第二、三版为第1–200轮。虚线标出第41轮开始在线补标/UniMatch的阶段。',
           '- UniMatch的训练loss包含额外一致性项，不能直接和第一版训练loss比较高低。第三版d0=0.01改变了原标签范围，验证loss也不能作为三组统一性能排名。',
           '- 曲线均来自本地loss.txt和ema_loss.txt，最佳轮次与best_metrics.json交叉核验；原数据和模型不改动。','',
           '| 版本 | 最佳Student轮次 / 验证loss | 最佳Teacher轮次 / 验证loss |','|---|---|---|']
    for folder,title,s,t,b in data:
        lines.append(f'| {folder} | {b["student"]["epoch"]} / {b["student"]["loss"]:.4f} | {b["teacher"]["epoch"]} / {b["teacher"]["loss"]:.4f} |')
    (out/'曲线说明.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Saved:',out,flush=True)


if __name__=='__main__':
    main()
