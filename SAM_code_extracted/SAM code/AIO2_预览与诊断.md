# AIO2 固定预览与训练诊断

本步骤仅观察模型，不自动补标签、不改变损失函数或 PU 参数。EMA 本身继续按既有方式更新。

## 不训练，直接查看已有权重

在 WSL 的 SAM code 目录中、激活 earthquake-sam 环境后运行：

```bash
python preview_ema.py \
  --checkpoint train_v5_ema_smoke/ema_checkpoint.pth
```

默认读取最近完成轮次的检查点，输出到同一运行目录下的 preview/epoch_XXXX。学生和教师均仅做前向预测，检查点不会被覆盖。可选参数：--outDir、--count（默认 6）、--threshold（默认 0.7）、--buffer（默认 2 像素）、--minPixels（默认 8 像素）。脚本目前支持 mars_sam/mars_sam_vit 和空间 key_train.lst 划分。

## 在下一轮训练中启用

从现有第 4 轮继续到第 5 轮：

```bash
bash train_wsl.sh \
  --resDir train_v5_ema_smoke/ \
  --pu 0 \
  --emaTeacher \
  --emaDecay 0.99 \
  --epochs 5 \
  --emaResume train_v5_ema_smoke/ema_checkpoint.pth \
  --emaPreview
```

--epochs 是总目标轮次。新加的预览参数允许在旧 EMA 检查点上开启或调整，其他训练参数仍检查一致性。不加 --emaPreview 时不增加训练诊断或预览开销。

训练时的可选参数：

| 参数 | 默认值 | 用途 |
|---|---:|---|
| --previewCount | 6 | 固定训练样本数量 |
| --previewEvery | 1 | 图像每隔多少轮输出；最后一轮也输出 |
| --previewThreshold | 0.7 | 仅用于候选预览和概率统计的阈值 |
| --previewBuffer | 2 | 原始标签周围排除缓冲区的宽度（像素） |
| --previewMinPixels | 8 | 候选连通区域最少像素数 |

这些阈值是诊断起点，不是经验证的断层补标阈值。不要把它们解读为完整 O2C 的筛选规则。

## 固定样本如何选取

- 仅从实际训练 keys 中选样，不使用测试集。
- 按中心裁剪中是否含原始 k0 正标签，尽可能各选一半；每组按 key 排序均匀选取，不抽随机数。
- 使用独立的 Generator_new(fixed_crop=True, training=False)：固定中心裁剪，无随机缩放、翻转或旋转。原训练/验证加载器的行为不变。
- samples.json 保存 key、尺寸、标签比例和输入/标签摘要。再次运行必须匹配同一份样本，防止把不同裁剪误当作模型变化。修改样本数量或数据后，请使用新的预览输出目录。
- 预览保护随机数状态，恢复学生各子模块原有 train/eval 状态，不更新梯度或 EMA 次数。

## 输出和读图

preview/samples.json：固定样本清单。

preview/epoch_0005/ 等轮次目录中：

- sample_00.png 等：地形、原始标签、学生概率、教师概率、候选叠加、概率直方图。
- sample_00.npz 等：对应的完整概率数组、原始二值标签、有效像素及候选掩膜，方便后续跨轮比较。
- metrics.json：每个样本的正标签比例、学生/教师概率均值、分位数、20 档直方图，以及候选数量/面积。
- train_batches.csv：本轮每个真实增强训练 batch 的 k0 标签占比、预测分布、有效像素数和原始损失。
- train_summary.json：区分“整个 batch 的原始 k0 标签为空”和“含原始正标签”，分别统计近 1 的损失（>=0.9999）次数及平均损失。它能帮助区分空标签批次与其他问题，不会自动把 loss=1 判定为正常。

独立 preview_ema.py 没有训练 batch，因此不会为历史轮次生成 train_batches.csv/train_summary.json。它们只在以后开启 --emaPreview 的训练轮次中产生。

候选图：绿色是原始正标签；青色是教师高于阈值、且在原标签缓冲区之外的有效像素；洋红色是其中通过 8 邻域连通面积筛选的区域。原标签不会被修改。连通面积筛选不等于断层段识别，海脊、峡谷等可能成为误报。

图上的学生/教师概率固定显示 0–1，便于跨轮比较。若教师概率几乎全图很高，即使验证 loss 有下降，也不能直接启用补标。

## 本次验证结果（2026-09-09）

- 14 项测试通过：EMA、CUDA、续训、固定裁剪、候选排除规则、无效像素、旧检查点兼容、开启预览不改变学生结果。
- 已使用真实 SAM-2 第 4 轮检查点（EMA 更新 80 次）生成 6 张预览；没有继续训练或更新权重。
- 这 6 个固定样本的教师平均概率约 0.914–0.964，候选覆盖大部分图像。当前候选质量不足以直接补标，需要继续观察训练与排查预测分布，不能仅依据平均 loss 判断。
- 终端轮末训练 loss 的重复除法已修正；loss.txt 原有数据和损失计算未变。
