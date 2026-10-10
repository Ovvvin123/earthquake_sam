"""Extend the existing editable AIO2 plotting source; keep its layout intact."""
from pathlib import Path
OUT=Path(__file__).resolve().parent
source=(OUT/'draw_training_flow_aio2.py').read_text(encoding='utf-8')
source=source.split('fig.subplots_adjust')[0]
source=source.replace("figsize=(20,16)","figsize=(33,16)")
source=source.replace("ax.set_xlim(0,20)","ax.set_xlim(0,33)")
source=source.replace('从七通道地形到断层预测：加入 AIO2 思路的训练流程','从七通道地形到断层预测：AIO2 + UniMatch V2 思路适配')
source=source.replace('本项目适配：EMA 教师 + 在线临时补标','保留原模型与 AIO2 补标流程；新增双强视图、互补特征遮挡与正标签一致性训练')
source=source.replace("box(17,7.5,2.55,1.55,'计算损失','加权 Focal Tversky\\n比较预测与临时标签',fc='#fff2e8',ec='#e9bd99',bs=11.5)",
"box(17,7.5,3.55,1.55,'合并训练损失','L_total = L_base + λ × L_cons\\n原支路 L_base：加权 Focal Tversky',fc='#fff2e8',ec='#e9bd99',bs=11.5)")
source=source.replace("text(.45,.65,'预热后启用补标；无合格候选时沿用原标签；验证仍使用原始标签。',12,muted,ha='left')", "")
source=source.replace("text(.45,.22,'当前为 AIO2 思路适配，未完整实现原论文 ACT / O2C。本图不展开 UniMatch；梯度可累积后再更新学生。',11,muted,ha='left')", "")
exec(compile(source,str(OUT/'draw_training_flow_aio2.py'),'exec'))

violet='#7751a5'
ax.add_patch(FancyBboxPatch((21.55,3.45),10.85,10.3,boxstyle='round,pad=.03,rounding_size=.17',fc='#faf7fd',ec='#c9b7de',lw=1.5,zorder=1))
text(21.88,13.42,'UniMatch V2 思路新增分支  /  仅在训练时启用',17,violet,ha='left',weight='bold')
box(22.05,11.05,9.85,1.7,'从同一输入独立生成两个强视图','仅扰动 CTX 与三方位晕渲的亮度、对比度和噪声\n水深、有效掩膜与几何坐标保持不变',fc='#f3ecfa',ec='#c7b1dc',ts=18,bs=13)
# Additional input branch is routed above the original student container.
arrow([(1.67,12.6),(1.67,14.1),(32.65,14.1),(32.65,11.9),(31.9,11.9)],violet)

text(21.8,14.34,'同一批七通道输入',12,violet,ha='left')

box(22.05,8.85,4.35,1.5,'强视图 1 → 共享学生','解码器特征 × 2M\n得到预测 p1',fc='#f3ecfa',ec='#c7b1dc',ts=16,bs=12)
box(27.55,8.85,4.35,1.5,'强视图 2 → 共享学生','解码器特征 × 2(1−M)\n得到预测 p2',fc='#f3ecfa',ec='#c7b1dc',ts=16,bs=12)
arrow([(24.22,11.05),(24.22,10.35)],violet)
arrow([(29.72,11.05),(29.72,10.35)],violet)
text(26.97,8.44,'共享上方学生参数；互补遮挡覆盖四个解码输入尺度（含局部跳跃分支）',11.3,violet)
box(24.2,6.05,5.55,1.7,'正标签一致性损失','L_cons = (L_strong1 + L_strong2) / 2\n仅在选中的正标签像素计算 −w log(p)',fc='#f3ecfa',ec='#c7b1dc',ts=17,bs=12)
arrow([(24.22,8.85),(24.22,8.06),(25.55,8.06),(25.55,7.75)],violet)
arrow([(29.72,8.85),(29.72,8.06),(28.4,8.06),(28.4,7.75)],violet)
arrow([(24.2,6.9),(21.07,6.9),(21.07,8.275),(20.55,8.275)],violet)
text(22.64,7.19,'新增 L_cons',12,violet)

# Supervision for strong branches: original positives and accepted temporary positives.
arrow([(19.55,5.25),(26.97,5.25),(26.97,6.05)],teal)
text(26.97,4.91,'有效原始正标签（w=1）+ 接纳的临时正标签（降权）',12,teal)
text(26.97,4.06,'强分支不新增背景监督；原支路保留原有监督范围',12,muted)

box(21.55,1.3,10.85,1.6,'联合优化，保持同一个训练循环','原支路与双强视图依次反向传播并累积梯度；一次优化器更新后，EMA 教师只更新一次。\n默认第 41 轮启用，λ 逐步增至 0.1；推理时关闭强增强与特征遮挡。',fc='#f7f8fa',ec='#ccd5de',ts=16,bs=12)
text(.45,.65,'强视图在同一组正标签上学习；无临时候选时仍可用原始正标签。验证与测试始终使用原始标签。',12,muted,ha='left')
text(.45,.22,'按本项目实际代码概括：AIO2 与 UniMatch V2 均为思路适配，非原论文完整复现；参数值为默认实验设置。',11,muted,ha='left')
fig.subplots_adjust(left=.008,right=.992,top=.99,bottom=.01)
fig.savefig(OUT/'model_training_flow_AIO2_UniMatchV2.png',dpi=200,facecolor='white')

# Second export includes the verified historical comparison without changing the main flow.
fig.set_size_inches(33,19)
ax.set_ylim(-3,16)
ax.add_patch(FancyBboxPatch((.45,-2.5),31.95,2.28,boxstyle='round,pad=.03,rounding_size=.15',fc='#f8fafc',ec='#cbd5e1',lw=1.3,zorder=2))
text(.8,-.61,'已核验实验结果｜d0 = 0.02 · 200 轮 · 同一组 38 个测试图块',18,ha='left',weight='bold')
text(.8,-1.2,'像素 AP：学生 18.82% → 14.76%（−4.06 个百分点）    教师 18.46% → 18.17%（−0.29 个百分点）',18,ha='left')
text(.8,-1.76,'比较方向：原 EMA + 在线补标 → 加入 UniMatch V2 思路适配。本次主要口径下 AP、F1、IoU 均未提升。',14,ha='left')
text(.8,-2.21,'两次历史运行的预热／验证设置不同，并非严格消融；训练与验证区存在空间重叠，不能将变化单独归因于 UniMatch。未纳入 d0=0.01 实验。',12,muted,ha='left')
fig.savefig(OUT/'model_training_flow_AIO2_UniMatchV2_results.png',dpi=200,facecolor='white')
plt.close(fig)
print('Saved flow-only and flow-with-results PNGs.')

