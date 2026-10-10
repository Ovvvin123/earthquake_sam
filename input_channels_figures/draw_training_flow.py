from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from matplotlib.font_manager import FontProperties

OUT=Path(__file__).resolve().parent
F=FontProperties(fname='/mnt/c/Windows/Fonts/msyh.ttc')
fig,ax=plt.subplots(figsize=(20,11.5),facecolor='white')
ax.set_xlim(0,20);ax.set_ylim(0,11.5);ax.axis('off')
navy='#19344b'; blue='#2476a7'; orange='#cc6632';muted='#556b7c'
def text(x,y,s,size=15,color=navy,ha='center',weight='normal'):
    return ax.text(x,y,s,fontproperties=F,fontsize=size,color=color,ha=ha,va='center',weight=weight,linespacing=1.6)
def box(x,y,w,h,title,body,fc='#edf6fb',ec='#b9d5e7',ts=17,bs=13):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.025,rounding_size=0.12',facecolor=fc,edgecolor=ec,linewidth=1.4))
    text(x+w/2,y+h*.68,title,ts)
    text(x+w/2,y+h*.29,body,bs,muted)
def arrow(points,color=blue,dashed=False):
    for a,b in zip(points[:-2],points[1:-1]):
        ax.plot([a[0],b[0]],[a[1],b[1]],color=color,lw=2,ls='--' if dashed else '-',zorder=4)
    ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=16,color=color,lw=2,linestyle='--' if dashed else '-',zorder=4))

text(.45,10.95,'从七通道地形到断层预测：模型与训练循环',27,ha='left',weight='bold')
text(.45,10.32,'基础监督流程  ·  先预测，再比较标签，最后根据梯度更新参数',16,muted,ha='left')
ax.add_patch(FancyBboxPatch((3.5,5.95),12.7,3.65,boxstyle='round,pad=0.03,rounding_size=.17',facecolor='#f8fafc',edgecolor='#a9becc',linewidth=1.4))
text(3.75,9.3,'模型内部  /  conv_sam',15,ha='left')
box(.45,6.55,2.45,1.9,'七通道输入','地形 × 2\n掩膜 × 2 · 晕渲 × 3',bs=12)
box(3.85,6.55,2.15,1.9,'卷积特征融合','整合七通道\n提取局部特征',bs=12)
box(6.5,6.55,2.1,1.9,'三通道适配','可学习的卷积\n生成三通道表征',bs=12)
box(9.1,6.55,2.75,1.9,'SAM-2 编码器','预训练图像编码器\n提取多尺度特征',fc='#e9f1fa',bs=12)
box(12.45,6.55,3.3,1.9,'特征融合与解码','融合局部与 SAM 特征\n逐级上采样 + 输出层',bs=12)
box(17,6.55,2.55,1.9,'断层概率图','每个像素的断层分数\n取 k0 主任务通道',fc='#e9f6ef',ec='#abcdbb',bs=12)
for a,b in [(2.9,3.85),(6,6.5),(8.6,9.1),(11.85,12.45),(15.75,17)]:arrow([(a,7.5),(b,7.5)])
arrow([(4.93,8.45),(4.93,8.82),(14.1,8.82),(14.1,8.45)])
text(10.0,9.02,'局部特征通过跳跃连接参与融合',12,muted)
text(9.8,6.19,'图中合并展示多个尺度的融合与上采样',11,muted)

box(17,3.65,2.55,1.55,'计算损失','预测与标签比较\nFocal Tversky',fc='#fff2e8',ec='#e9bd99',bs=12)
arrow([(18.28,6.55),(18.28,5.2)])
text(18.28,5.8,'预测值',12,muted)
box(10.0,3.65,4.75,1.55,'反向传播 → 参数更新','计算梯度，再由 AdamW 更新参数',fc='#fff2e8',ec='#e9bd99',bs=12)
arrow([(17,4.42),(14.75,4.42)],orange)
text(15.86,4.72,'损失',12,orange)
arrow([(12.0,5.2),(12.0,5.95)],orange,True)
text(11.72,5.59,'更新参与前向计算的可训练参数',12,orange,ha='right')

box(10.0,.85,3.5,1.55,'USGS 断层目录','已有断层矢量线',fc='#f1effa',ec='#c5bfdf',bs=12)
box(15.1,.85,4.45,1.55,'构建训练标签','空间对齐 → 距离场 → 阈值化',fc='#f1effa',ec='#c5bfdf',bs=12)
arrow([(13.5,1.62),(15.1,1.62)],'#8271a8')
arrow([(18.28,2.4),(18.28,3.65)],'#8271a8')
text(18.7,3.04,'目标值',12,'#8271a8')

text(.6,4.65,'如何理解模型中间这一段？',19,ha='left',weight='bold')
text(.6,3.48,'融合输入：把七个通道的信息结合起来\n提取特征：识别不同尺度的地形结构\n融合解码：把特征还原为逐像素的断层分数',15,ha='left')
text(.6,1.62,'每批数据重复这一循环。\n梯度可先累积，再执行一次参数更新。',14,muted,ha='left')
text(.45,.22,'按当前 mars_sam + k0Only 路径概括；EMA 教师、在线补标和 UniMatch 可在此基础上另行展开。',11,muted,ha='left')
fig.subplots_adjust(left=.01,right=.99,top=.99,bottom=.01)
fig.savefig(OUT/'model_training_flow.png',dpi=200)
fig.savefig(OUT/'model_training_flow.svg')
plt.close(fig)
(OUT/'model_training_flow_notes.md').write_text('''# 模型与基础训练流程

图对应当前 `mars_sam`（`isvit=False`）与 `k0Only` 主任务。为便于介绍，未展开可选对齐、EMA、在线补标、PU 或 UniMatch 分支。

模型：七通道 → head 卷积/残差块 → input_sam 三通道适配 → SAM-2 forward_image → 多尺度特征与局部跳跃特征相加 → 自定义上采样解码与 tail → Softmax。完整输出为三通道，k0Only 损失取首个主任务通道，不应把图理解为原生 SAM 提示分割或单通道 Sigmoid。

监督：USGS 线生成距离场，训练时以 target < d0 形成标签；损失使用连续预测和像素权重，不必先把预测变成硬二值图。图中损失按 FocalTversky 配置标注。

更新：loss.backward() 计算梯度，梯度累积到更新时点后裁剪，再 opt.step()（AdamW），最后清零梯度。当前代码未显式冻结 SAM-2 图像编码器，SAM 参数组使用较小但非零学习率；eval() 不等于冻结参数。只有参与前向计算并得到梯度的参数会更新。旧说明中的“冻结主干”与当前代码不一致。

代码依据：torchUnet.py 中 conv_sam.__init__ / forward / FocalTversky；trainV10.py 模型构建、优化器和训练循环。
''',encoding='utf-8')
print('Saved model_training_flow.png and .svg')
