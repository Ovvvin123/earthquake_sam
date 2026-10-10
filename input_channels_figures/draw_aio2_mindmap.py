from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.path import Path as MPath

OUT=Path(__file__).resolve().parent
FONT=FontProperties(fname='/mnt/c/Windows/Fonts/msyh.ttc')
fig,ax=plt.subplots(figsize=(24,14.6),facecolor='#ffffff')
ax.set_xlim(0,24);ax.set_ylim(0,14.6);ax.axis('off')
def txt(x,y,s,size=17,color='#21394d',ha='left',weight='normal'):
    ax.text(x,y,s,fontproperties=FONT,fontsize=size,color=color,ha=ha,va='center',weight=weight,linespacing=1.75,zorder=5)
def rounded(x,y,w,h,fc,ec,lw=1.5):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.025,rounding_size=.19',fc=fc,ec=ec,lw=lw,zorder=3))
def branch(x1,y1,x2,y2,color):
    mid=(x1+x2)/2
    p=MPath([(x1,y1),(mid,y1),(mid,y2),(x2,y2)],[MPath.MOVETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4])
    ax.add_patch(PathPatch(p,fc='none',ec=color,lw=2.8,zorder=1))
def card(x,y,num,title,body,color,fc,bs=16):
    rounded(x,y,7.5,2.8,fc,color)
    rounded(x+.24,y+2.02,.53,.48,color,color)
    txt(x+.505,y+2.26,str(num),16,'white','center')
    txt(x+.95,y+2.26,title,20,color,weight='bold')
    txt(x+.3,y+1.14,body,bs)

txt(.6,14.02,'Earthquake-SAM｜加入 AIO2 思路的训练导图',29,weight='bold')
txt(.6,13.4,'本项目适配版：EMA 教师 + 在线临时补标；采用固定轮次触发，尚未完整实现原论文 ACT / O2C。',16,'#5a6c7a')

left=.6;right=15.9
ys=[10.05,6.55,3.05]
colors=['#247ca8','#3768ab','#4d8c7e','#997232','#9861a8','#c56a43']
for y,c in zip(ys,colors[:3]):branch(8.1,y+1.4,9.15,8.13,c)
for y,c in zip(ys,colors[3:]):branch(14.85,8.13,15.9,y+1.4,c)

card(left,ys[0],1,'数据与原始监督',
     '输入：七通道地形、掩膜与晕渲\nUSGS 断层线 → 距离场 → 原始标签\n未标注区域可能仍有真实断层',colors[0],'#eff7fc')
card(left,ys[1],2,'学生模型：负责学习与预测',
     '卷积融合 → 三通道适配 → SAM-2 编码器\n多尺度特征融合与解码 → 断层概率图\n根据训练损失反向传播，更新模型参数',colors[1],'#f0f4fb')
card(left,ys[2],3,'EMA 教师：提供候选预测',
     '初始化复制学生；推理不参与反向传播\n每次学生参数更新后，进行 EMA 更新\n教师参数 ← α × 旧教师 + (1−α) × 新学生',colors[2],'#edf7f3',bs=15)

card(right,ys[0],4,'筛选疑似漏标区域',
     '默认：以 p ≥ 0.5 建连通区域，含 p ≥ 0.9 种子\n整片不接触原标签及其 3 像素缓冲区\n至少 16 像素；单区与总面积均受 5% 预算限制',colors[3],'#fff8eb',bs=14)
card(right,ys[1],5,'生成本批次的临时监督',
     '保留原始正标签，接纳候选作为临时正标签\n逐 batch 重新生成；不改写原始标签文件\n预热后启用，候选像素权重逐步提高',colors[4],'#f8f0fb',bs=15)
card(right,ys[2],6,'计算损失，完成训练闭环',
     '学生概率 + 临时标签 → 加权 Focal Tversky\n反向传播 → 梯度累积／裁剪 → AdamW 更新\n验证仍使用原始标签；不加入教师伪标签',colors[5],'#fff3ed',bs=14.5)

rounded(9.15,6.95,5.7,2.4,'#193b52','#193b52')
txt(12,8.78,'AIO2 思路适配',26,'white','center',weight='bold')
txt(12,7.97,'EMA 教师\n+ 在线临时补标',21,'white','center')
txt(12,7.28,'用于缓解断层目录漏标',14,'#d5e5ed','center')
txt(12,5.96,'默认补标时间表',16,'#475e71','center',weight='bold')
txt(12,4.93,'1–40 轮：预热，不补标\n41–50 轮：候选权重 0.05\n51–80 轮：逐步增至 0.2\n81 轮起：维持 0.2，继续补标',13,'#536b7d','center')
txt(12,3.51,'权重值是像素系数\n不代表候选占总损失的比例',11,'#71818d','center')

rounded(.6,1.35,22.8,1.02,'#f3f6f9','#d7e0e7')
txt(12,1.86,'教师预测 → 筛选候选 → 临时标签 → 学生预测与损失 → 更新学生 → EMA 更新教师 → 下一批',18,'#25445b','center',weight='bold')
txt(.6,.73,'参数为当前实现的默认实验设置，可随训练配置改变。无合格候选时沿用原标签；候选是否为真实断层仍需核查。',12,'#657887')
txt(.6,.28,'依据：aio2_online.py、aio2_ema.py、trainV10.py 及在线临时补标说明。此图不展开 UniMatch 分支。',11,'#7b8c99')
fig.subplots_adjust(left=.01,right=.99,top=.99,bottom=.01)
fig.savefig(OUT/'AIO2_模型与训练思维导图.png',dpi=200,facecolor='white')
plt.close(fig)
print('Saved AIO2_模型与训练思维导图.png')
