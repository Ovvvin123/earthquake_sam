# 在线临时补标：训练与检查说明

这是固定轮数触发的保守在线补标实验，不是完整复现 AIO2 的 ACT/O2C。
原始标签文件不改写；每次读取训练 batch 都用当前 EMA 教师重新生成临时标签。
现有 PU 功能和 train_wsl.sh 保留，本实验运行时显式使用 --pu 0。

## 默认时间表（epoch 为从 1 开始的人类编号）

| Epoch | 临时标签像素权重 |
|---|---|
| 1–40 | 不补标 |
| 41–50 | 0.05 |
| 51–80 | 从 0.05 线性增加到 0.2，第 80 轮到达 0.2 |
| 81–200 | 保持 0.2；临时标签仍然逐 batch 重新生成 |

第 51 轮为 0.055，第 65 轮为 0.125。
过渡结束不代表停止补标。代码不会保证每轮新增数量增长，也不会根据 val 自动提高权重。
总轮数由 --epochs 指定，不自动启动长训练，不启用早停。

## 从现有第 40 轮开始正式实验

在 WSL 的代码目录执行（train_wsl.sh 会激活原训练环境）：

~~~bash
bash train_wsl.sh \
  --resDir train_v5_online/ \
  --pu 0 \
  --emaTeacher \
  --emaDecay 0.98 \
  --epochs 200 \
  --emaResume train_v5_ema_smoke/ema_checkpoint.pth \
  --emaStageTransition \
  --onlineCorrection \
  --emaPreview \
  --previewEvery 5
~~~

--emaStageTransition 只用于从预热 checkpoint 切换到新实验。
保留学生、教师、优化器、累计 EMA 更新数、随机状态和旧训练循环的待累积梯度。
加载完成后明确将 EMA 系数改成 0.98。
要求新输出目录，不能覆盖已经有模型结果的实验。
源 checkpoint 必须早于 onlineStart；默认预期 next_epoch=40。
原始 train_v5_ema_smoke/ema_checkpoint.pth 不会被改写。
--onlineCorrection 自动启用 --fixedVal，要求 --k0Only、FocalTversky、EMA、PU=0。

可以将 --epochs 200 改成 --epochs 41 先仅跑第 41 轮。
--epochs 是最终总轮数，不是再训练多少轮。

## 正常中断续训

恢复新实验自己的 checkpoint 时，不再传 --emaStageTransition：

~~~bash
bash train_wsl.sh \
  --resDir train_v5_online/ \
  --pu 0 \
  --emaTeacher \
  --emaDecay 0.98 \
  --epochs 200 \
  --emaResume train_v5_online/ema_checkpoint.pth \
  --onlineCorrection \
  --emaPreview \
  --previewEvery 5
~~~

补标起止轮数、阈值、权重等训练参数必须与已保存配置一致。
正常恢复会保留历史最佳分数。更改验证内容会触发指纹检查，避免混用不同考题。
若手动改了候选参数，不应把它当成同一实验无声续训，应建立明确的新实验。
本次阶段切换只开放从未启用在线补标的预热 checkpoint 开始，防止误用。

## 原标签对照组

同样从第 40 轮开始，EMA 同为 0.98、固定验证一致，但不启用在线补标：

~~~bash
bash train_wsl.sh \
  --resDir train_v5_online_control/ \
  --pu 0 \
  --emaTeacher \
  --emaDecay 0.98 \
  --epochs 200 \
  --emaResume train_v5_ema_smoke/ema_checkpoint.pth \
  --emaStageTransition \
  --fixedVal \
  --emaPreview \
  --previewEvery 5
~~~

对照组续训同样去掉阶段切换开关，改为自己的 checkpoint，继续保留 --fixedVal。

## 候选如何筛选

默认值都是实验起点，不是已证实的最佳参数：

- --onlineThreshold 0.9：一个候选区域至少有一个概率达到 0.9 的高置信种子。
- --onlineSupport 0.5：用概率至少 0.5 的区域建立八连通关系。
- --onlineBuffer 3：原标签周围 3 像素作为排除范围。
- --onlineMinPixels 16：候选连通区域至少 16 像素。
- --onlineMaxFraction 0.05：单个区域以及每张裁剪的候选总面积不超过有效面积的 5%。

先在完整教师预测上判断连通区域。整个区域只要碰到原标签或其缓冲区，就整体拒绝。
不能先删除原标签再把剩下的扩张环当成新对象。
中等概率的连接带也参与判断，降低把外扩区域误认为独立区域的机会。
不合格的面积区域整体拒绝，不按面积上限强行切开。
若多个合格区域总面积仍超预算，则该裁剪本次全部不接纳，记录 rejected_budget。
不使用无效像素或非有限预测生成临时标签。
这会漏掉部分真实断层延伸，也不保证独立区域就一定是断层。
不保证每个 batch 或每个 epoch 都有合格候选。
连通关系只在当前裁剪内判断；位于裁剪边缘的区域是否连接到裁剪外的原标注，当前规则无法判断，需重点人工核查。

## 临时监督如何生效

原始正标签保留。接纳区域本次变为硬正标签，其他像素标签不变。
在同一个 FocalTversky 计算中，候选不再同时作为背景产生 FP 惩罚。
候选像素的空间权重 = 当前计划权重 × 该裁剪原空间权重的最大值。
这样远离原标注的候选不会再被“离原标注远”重复削弱。
其他像素的空间权重保持原样；没有候选或计划权重为零时，目标和权重均原样返回。
0.2 是像素权重系数，不表示候选占总 loss 的 20%，也不是候选面积比例。
该规则属于本实验的损失接入方式，不是声称与原论文损失完全一致。

教师先推理，学生再前向与反向传播；每次实际 optimizer.step 后进行 EMA 更新。
保留原训练循环的梯度累积逻辑，避免在同一实验中再改一个变量。

## 验证与模型选择

--fixedVal 使用每个验证 key 的固定中心裁剪，关闭随机增强，不打乱，不丢弃尾部样本。
当前数据是 17 个验证 key，因此每轮评估 17 张固定裁剪，不代表覆盖每张原图的全部区域。
对每张裁剪独立计算 loss，再取平均，避免 batch 分组影响验证结果。
仅在有效像素上评估，使用原始标签、PU=0，绝不向验证标签加入伪标签。
第一次启动在训练之前对第 40 轮 checkpoint 建立固定验证基准。
这个数值不能直接与旧随机裁剪验证曲线比较。
validation_manifest.json 记录验证样本 key、数量和实际输入/标签指纹。
独立验证不会消耗训练随机状态或改变学生训练模式。

主要输出：

- ema_checkpoint.pth：最新完整状态，用于正常续训。
- model.pth、teacher/model.pth：最新学生、教师权重。
- best_student/model.pth、best_teacher/model.pth：各自在固定验证上最优的模型权重。
- best_student/ema_checkpoint.pth、best_teacher/ema_checkpoint.pth：对应时刻的完整训练状态。
- best_student/metrics.json、best_teacher/metrics.json、best_metrics.json：最佳轮数及 loss。
- checkpoints/epoch_0060.pth 等：默认每 20 轮保存完整状态；--checkpointEvery 0 可以关闭。
- fixed_val.jsonl：一开始的固定验证基准，以及逐轮学生、教师结果。
- loss.txt、ema_loss.txt：继续保留原格式，epoch 索引仍从 0 开始。
- ema_config.json：此次命令实际配置。

学生、教师的最佳轮数可以不同。完整 checkpoint 同时包含当时两者，不要混合不同轮数状态。
这些模型按现有验证标注选出；若验证标注也有漏标，最低 loss 不保证漏标恢复效果最好。
最终应结合固定预览和人工核查样本。完全独立的测试集不参与选轮数。
最佳模型和阶段完整状态会额外占用磁盘空间。

## 去哪里看实际补标

online/epoch_0041/summary.json：本轮权重、接受像素、接受区域以及各类拒绝计数、逐 batch loss。
这些数量包含重复抽样和数据增强，不能解释为新增了多少条独立断层。

online/epoch_0041/batch_sample_00.png 等：
左为原标签（绿色），中为教师概率，右为实际参与本次训练的临时正标签（紫红色）。
同时保存 NPZ，可检查原标签、临时标签、接纳掩膜与教师概率。
第 41 轮、预览间隔轮和本次运行最后一轮保存至多两个实际训练裁剪。
其中优先保留带候选的例子；无候选时也保留一个例子供检查。

原 preview/epoch_XXXX/ 仍是固定训练样本的旧诊断筛选（阈值 0.7）。
其紫红色不等于实际接纳的标签；判断本实验用了什么应看 online/。
固定预览用于观察同一样本的教师/学生变化，online 图用于核查实际临时监督。

## 验证代码

~~~bash
python -m unittest test_aio2_ema test_aio2_preview test_aio2_online -v
~~~

覆盖关闭新功能时的行为、EMA 更新、恢复一致性、原标签不变、候选可消失、
候选梯度方向、环状扩张拒绝、无效数据和面积约束、固定验证、最佳 checkpoint 与数据指纹。
真实模型 smoke 结果见同目录 train_v5_online_smoke_console.log；该目录与正式实验分开。

原配置、训练入口和 EMA 文件在 _backup_before_online/ 保留改动前副本。

## 本次实际验证结果

2026-09-09：23 项测试通过；真实 SAM/CUDA 从原第 40 轮状态完成第 41 轮。
输出目录 train_v5_online_smoke/，教师累计更新由 800 增至 820，临时权重 0.05。
本轮候选累计 25,623 像素、36 个被接纳连通区域（含重复/增强裁剪），拒绝 63 个触碰原标注的区域。
固定验证覆盖 17 个中心裁剪：
- 第 40 轮基准：学生 0.709360，教师 0.701710。
- 第 41 轮：学生 0.706482，教师 0.704664。
- 最佳学生保留第 41 轮，最佳教师保留第 40 轮。

这验证了阶段切换、临时监督、固定验证、最新/最佳状态保存可运行，
不证明补标准确或长期效果改善；正式 200 轮训练尚未启动。


## 2026-09-10：修复第 158 轮的 NaN 梯度

console_resume.log 显示，第 158 轮（零起始索引 157）batch 28 的 loss 为 0，
下一批开始出现 NaN，最终被固定验证的非有限数值检查拦截。固定验证并非最初故障来源。

已用原 FocalTversky 函数复现：gamma=0.75 时，零误差的分数次幂导数非有限；
浮点舍入也会使几乎完美的预测得到精确的零误差。因此有限的 loss=0 并不保证梯度有限。
该边界问题在 PU=0 和 PU=0.4 下都能复现，不能归因于关闭 PU 本身。

修复：
- 在分数次幂之前给误差加机器精度下限；常规区间的 loss 和梯度保持原公式结果。
- 默认 float32、gamma=0.75 时，极小 loss 的下限约为 0.0000064，这不表示模型没有收敛。
- 在训练输入、距离标签、教师/学生预测、空间权重和 loss 出现非有限值时立即报告。
- optimizer.step 前检查梯度范数，非有限时终止，不将坏梯度传入学生参数和 EMA 教师。
- 固定验证错误现在报告具体学生/教师、样本位置、key 和 epoch。
- 数值异常写入当前结果目录 numerical_failure.json，不静默跳过异常样本。

已有正常 checkpoint 可按原命令续训，无需增加开关、重置模型或修改 PU。
代码更新不会替换已经运行进程中的函数，需重新启动训练进程才能生效。
原文件备份位于 _backup_before_nanfix/。

测试命令：
~~~bash
python -m unittest test_aio2_ema test_aio2_preview test_aio2_online test_training_numerics -v
~~~
28 项测试通过，覆盖 CPU/CUDA 零误差、PU兼容性、普通梯度不变、
坏梯度阻止真实训练分支的参数更新，以及验证失败样本定位。
独立实机重跑日志位于 train_v5_online_nanfix_smoke_console.log。
