from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from matplotlib.font_manager import FontProperties

OUT=Path(__file__).resolve().parent
F=FontProperties(fname='/mnt/c/Windows/Fonts/msyh.ttc')
fig,ax=plt.subplots(figsize=(18,11.5),facecolor='white')
ax.set_xlim(0,18);ax.set_ylim(0,11.5);ax.axis('off')
ink='#193b52';muted='#536c7c';blue='#337ca5';green='#368d79';violet='#8561ac';orange='#c47a49'
def text(x,y,s,size=18,color=ink,ha='center'):
    ax.text(x,y,s,fontproperties=F,fontsize=size,color=color,ha=ha,va='center',linespacing=1.6,zorder=5)
def box(x,y,w,h,title,body,color,fill):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.025,rounding_size=.16',fc=fill,ec=color,lw=1.5,zorder=3))
    text(x+w/2,y+h*.68,title,22)
    text(x+w/2,y+h*.27,body,15,muted)
def arrow(points,color=blue,dashed=False):
    for a,b in zip(points[:-2],points[1:-1]):
        ax.plot([a[0],b[0]],[a[1],b[1]],color=color,lw=2.4,ls='--' if dashed else '-',zorder=2)
    ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=20,lw=2.4,color=color,linestyle='--' if dashed else '-',zorder=4))

text(.6,10.9,'让模型学会找断层：加入 AIO2 和 UniMatch 后',29,ha='left')
text(.6,10.24,'AIO2：帮助寻找漏标候选     ｜     UniMatch：让同一个模型在不同扰动下练习识别',17,muted,ha='left')

box(.65,7.6,3.75,1.6,'① 输入地形图','前面介绍的七个通道',blue,'#eef6fb')
box(5.55,7.6,5.0,1.6,'② 模型尝试找断层','给每个位置一个“像断层”的分数',blue,'#eef6fb')
box(12.4,7.6,4.9,1.6,'③ 比较结果，调整模型','根据预测与训练标签的差异学习',orange,'#fff4ea')
arrow([(4.4,8.4),(5.55,8.4)])
arrow([(10.55,8.4),(12.4,8.4)])
arrow([(14.85,9.2),(14.85,9.67),(8.05,9.67),(8.05,9.2)],orange,True)
text(11.5,9.83,'调整后继续练习',13,orange)

box(.65,4.4,4.5,1.7,'老师找可能漏标的位置','由学生模型逐步更新，给出候选',green,'#eef8f3')
box(6.55,4.4,4.95,1.7,'筛选后，临时加入标签','保留 USGS 原标注；新增部分权重较低',green,'#eef8f3')
arrow([(2.525,7.6),(2.525,6.1)],green)
text(2.8,6.9,'同一份地形',13,green,ha='left')
arrow([(5.15,5.25),(6.55,5.25)],green)
arrow([(11.5,5.25),(14.85,5.25),(14.85,7.6)],green)
text(15.1,6.45,'用于比较的标签',14,green,ha='left')

box(.65,1.35,4.5,1.65,'把同一张图变出两个版本','稍微改变明暗、对比度，加入少量噪声',violet,'#f6f0fb')
box(6.55,1.35,6.2,1.65,'同一个模型再认两遍','内部遮住不同部分的特征；仍要认出标注的断层',violet,'#f6f0fb')
arrow([(5.15,2.175),(6.55,2.175)],violet)
arrow([(12.75,2.175),(16.75,2.175),(16.75,7.6)],violet)
text(16.5,3.35,'这两次练习的错误\n也用于调整模型',14,violet,ha='right')
arrow([(9.1,4.4),(9.1,3.0)],green)
text(9.35,3.7,'沿用同一组正标签',12.5,green,ha='left')

text(3.05,6.43,'AIO2 思路：补充可能漏掉的标注',15,green,ha='left')
text(.72,3.45,'UniMatch 思路：换个样子，也要认得出来',15,violet,ha='left')
text(.65,.67,'这里的“老师”也是模型：用学生历次参数的加权平均更新。候选标注仅供当前训练使用，不修改原始目录。',12.5,muted,ha='left')
text(.65,.25,'额外练习只要求认出已有标注和筛选后新增的断层位置；这些改进思路并不保证效果一定提高。',12,muted,ha='left')
fig.subplots_adjust(left=.01,right=.99,top=.99,bottom=.01)
fig.savefig(OUT/'model_training_flow_AIO2_UniMatchV2_simple.png',dpi=220,facecolor='white')
plt.close(fig)
print('Saved simple PNG')

