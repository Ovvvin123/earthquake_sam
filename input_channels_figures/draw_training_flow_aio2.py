from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from matplotlib.font_manager import FontProperties

OUT=Path(__file__).resolve().parent
F=FontProperties(fname='/mnt/c/Windows/Fonts/msyh.ttc')
fig,ax=plt.subplots(figsize=(20,16),facecolor='white')
ax.set_xlim(0,20);ax.set_ylim(0,16);ax.axis('off')
navy='#19344b';blue='#2476a7';orange='#cc6632';teal='#2c8c7e';purple='#8271a8';muted='#556b7c'
def text(x,y,s,size=15,color=navy,ha='center',weight='normal'):
    ax.text(x,y,s,fontproperties=F,fontsize=size,color=color,ha=ha,va='center',weight=weight,linespacing=1.65,zorder=6)
def box(x,y,w,h,title,body,fc='#edf6fb',ec='#b9d5e7',ts=17,bs=13):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.025,rounding_size=.12',fc=fc,ec=ec,lw=1.4,zorder=3))
    text(x+w/2,y+h*.72,title,ts)
    text(x+w/2,y+h*.32,body,bs,muted)
def arrow(points,color=blue,dashed=False):
    for a,b in zip(points[:-2],points[1:-1]):
        ax.plot([a[0],b[0]],[a[1],b[1]],color=color,lw=2,ls='--' if dashed else '-',zorder=4)
    ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=16,color=color,lw=2,linestyle='--' if dashed else '-',zorder=4))

text(.45,15.43,'从七通道地形到断层预测：加入 AIO2 思路的训练流程',27,ha='left',weight='bold')
text(.45,14.78,'本项目适配：EMA 教师 + 在线临时补标',17,muted,ha='left')
ax.add_patch(FancyBboxPatch((3.5,10.1),12.7,3.65,boxstyle='round,pad=0.03,rounding_size=.17',fc='#f8fafc',ec='#a9becc',lw=1.4,zorder=1))
text(3.75,13.43,'学生模型  /  conv_sam',15,ha='left')
box(.45,10.7,2.45,1.9,'七通道输入','地形 × 2\n掩膜 × 2 · 晕渲 × 3',bs=12)
box(3.85,10.7,2.15,1.9,'卷积特征融合','整合七通道\n提取局部特征',bs=12)
box(6.5,10.7,2.1,1.9,'三通道适配','可学习的卷积\n生成三通道表征',bs=12)
box(9.1,10.7,2.75,1.9,'SAM-2 编码器','预训练图像编码器\n提取多尺度特征',fc='#e9f1fa',bs=12)
box(12.45,10.7,3.3,1.9,'特征融合与解码','融合局部与 SAM 特征\n逐级上采样 + 输出层',bs=12)
box(17,10.7,2.55,1.9,'学生断层概率图','逐像素断层概率\nk0 主任务通道',fc='#e9f6ef',ec='#abcdbb',ts=16,bs=12)
for a,b in [(2.9,3.85),(6,6.5),(8.6,9.1),(11.85,12.45),(15.75,17)]:arrow([(a,11.65),(b,11.65)])
arrow([(4.93,12.6),(4.93,12.99),(14.1,12.99),(14.1,12.6)])
text(10,13.18,'局部特征跳跃连接',12,muted)
text(9.8,10.32,'图中合并展示多个尺度的融合与上采样',11,muted)

box(17,7.5,2.55,1.55,'计算损失','加权 Focal Tversky\n比较预测与临时标签',fc='#fff2e8',ec='#e9bd99',bs=11.5)
arrow([(18.28,10.7),(18.28,9.05)])
text(18.7,9.78,'预测值',12,muted)
box(10,7.5,4.75,1.55,'反向传播 → 更新学生','计算梯度，再由 AdamW 更新参数',fc='#fff2e8',ec='#e9bd99',bs=12)
arrow([(17,8.275),(14.75,8.275)],orange)
text(15.86,8.56,'损失',12,orange)
arrow([(12,9.05),(12,10.1)],orange,True)
text(12.25,9.58,'更新学生参数',12,orange,ha='left')

box(.65,7.4,4.6,1.65,'EMA 教师模型','与学生同结构，初始化复制学生\n只推理，不参与反向传播',fc='#edf8f5',ec='#94c7bb',bs=12)
arrow([(1.67,10.7),(1.67,9.05)])
text(1.94,9.84,'同一批输入',12,blue,ha='left')
arrow([(10,8.275),(5.25,8.275)],teal,True)
text(7.62,8.89,'每次学生参数更新后',12,teal)
text(7.62,8.59,'EMA 更新教师',13,teal)
text(7.62,7.89,'教师 ← α × 旧教师 + (1−α) × 新学生',10.5,teal)

box(6.5,4.2,5,2.1,'疑似漏标候选筛选','高置信种子 + 连通区域\n排除原标签邻域，限制候选面积',fc='#edf8f5',ec='#94c7bb',bs=13)
arrow([(2.95,7.4),(2.95,5.25),(6.5,5.25)],teal)
text(4.85,5.58,'教师预测',12,teal)
box(13,4.2,6.55,2.1,'本批次临时训练标签','保留原始正标签 + 接纳候选\n候选使用较低权重；逐批重建，不改写原文件',fc='#edf8f5',ec='#94c7bb',bs=12.5)
arrow([(11.5,5.25),(13,5.25)],teal)
text(12.25,5.58,'接纳候选',11,teal)
arrow([(18.28,6.3),(18.28,7.5)],teal)
text(18.64,6.93,'临时监督',12,teal,ha='left')

box(.65,1.35,4.6,1.55,'USGS 断层目录','已有断层矢量线',fc='#f1effa',ec='#c5bfdf',bs=12)
box(6.5,1.35,5,1.55,'原始训练标签','空间对齐 → 距离场 → 阈值化',fc='#f1effa',ec='#c5bfdf',bs=12)
arrow([(5.25,2.125),(6.5,2.125)],purple)
arrow([(9,2.9),(9,4.2)],purple)
text(9.27,3.56,'提供排除邻域',11.5,purple,ha='left')
arrow([(11.5,2.125),(15.05,2.125),(15.05,4.2)],purple)
text(15.33,3.3,'保留原始正标签',12,purple,ha='left')

text(.7,4.17,'AIO2 思路新增分支',15,teal,ha='left',weight='bold')
text(.7,3.56,'教师提供候选，学生从临时监督中学习',11.5,muted,ha='left')
text(.45,.65,'预热后启用补标；无合格候选时沿用原标签；验证仍使用原始标签。',12,muted,ha='left')
text(.45,.22,'当前为 AIO2 思路适配，未完整实现原论文 ACT / O2C。本图不展开 UniMatch；梯度可累积后再更新学生。',11,muted,ha='left')
fig.subplots_adjust(left=.01,right=.99,top=.99,bottom=.01)
fig.savefig(OUT/'model_training_flow_AIO2.png',dpi=220,facecolor='white')
plt.close(fig)
print('Saved model_training_flow_AIO2.png')
