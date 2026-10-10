# SAM-2 + UniMatch V2 思路：完整实现与运行说明

这是针对断层漏标任务的适配，不是 UniMatch V2 论文的完整复现。保留 SAM-2、EMA 教师、现有 FocalTversky 和每批临时补标；新增双强视图、互补特征遮挡、正标签一致性损失。本实现不使用 DINOv2、CutMix、伪背景或额外无标签数据集。

## 现在每批训练做什么

1. 读取原有训练裁剪。数据加载器已有的几何增强，是所有视图共同的几何基础。
2. 达到 onlineStart 后，教师在这份输入上预测，沿用原来的连通区域筛选规则生成临时正标签。
3. 学生原有支路照常对“原标签 + 临时标签”计算 FocalTversky，并先反向传播。
4. 达到 unimatchStart 后，对同一输入独立生成两个显示扰动版本。
5. 在两个版本的对应内部特征通道上施加互补遮挡。每个尺度使用 M*2 与 (1-M)*2；共享同一学生，依次前向/反向，不同时保存三个分支的反向图。
6. 仅在有效的原始正标签和接纳的临时正标签上，要求两个强视图识别断层。
7. 梯度按原先的 stepPerMiniBatch 累积。一次 optimizer.step 后只更新一次 EMA，强视图不会额外更新教师。
8. 下次读到这片区域时重新判断临时标签；原始 HDF5 不改写。

原训练支路仍保留未被接纳区域的原始背景监督。新增支路没有伪背景，也不表示整个训练已经忽略所有疑似漏标像素。教师/学生在多个视图下一致，不能单独证明补标正确。

## 七通道增强与模型接口

- 0：CTX 显示通道；1：水深；2、3：有效区域掩膜；4、5、6：不同方位地形阴影。
- 只改变 0、4、5、6，四个显示通道共享每张图的对比度、亮度变化及同一空间噪声；截断到 [-1, 1]。
- 水深、掩膜及无效位置逐元素不变。不新增几何变换，因此标签与预测保持相同坐标。
- 默认对比度幅度 0.2、亮度幅度 0.1、噪声标准差 0.03。这是待验证的温和起点。
- 特征遮挡位于自有解码器的四个输入尺度，包含局部高分辨率跳跃连接，避免完全绕过遮挡。
- 不修改 SAM-2 官方库，不新增模型参数。普通 forward 和现有权重文件格式兼容；推理/教师不启用特征遮挡。
- 当前开关只支持 modelType=mars_sam、k0Only、FocalTversky、PU=0、无 isalign。

## 损失：两个权重不是一回事

原支路损失记作 L_base。两个强视图的损失各为：

L_strong = sum(w * -log(p_k0)) / 选中的像素数

- 原始有效正标签 w=1。
- 临时正标签 w=当轮 online 权重，最高默认 0.2。
- 其他位置 w=0，不加入新的背景监督。
- 分母是未加权的选中像素数量，避免全是临时标签时将 0.2 的降权抵消。
- 使用主通道 k0 概率；不会对三个输出通道直接取 argmax 来生成二分类答案。
- 无可靠像素时跳过强视图，新增损失为零。

总损失：

L_total = L_base + lambda * (L_strong1 + L_strong2) / 2

原支路和两条强视图都除以原先的梯度累积系数。lambda 与 online 像素权重单独控制，均不是“实际总 loss 的百分比”。

## 默认训练时间表

| 轮数（从 1 开始） | 临时标签权重 | 新增一致性系数 lambda |
|---|---:|---:|
| 1–40 | 不补标 | 0，完全跳过新增训练支路 |
| 41–50 | 0.05 | 从第 41 轮的 0.0025 开始增长 |
| 51–80 | 线性增加到 0.2 | 线性增加到 0.1 |
| 81–200 | 0.2 | 0.1 |

第 60 轮 lambda=0.05。候选数量不强制增长，没有合格候选时不补标。到达一致性起始轮次后，即使没有临时候选，仍可在原始正标签上训练强视图。

--epochs 表示最终总轮数，不是再加多少轮。以上是初始实验配置，不是论文的最佳轮数结论。

## 运行：在 WSL 的 SAM code 目录

新的独立启动脚本继承原 train_wsl.sh 的数据、模型、batchsize=2、梯度累积=4、学习率等设置；额外默认 PU=0、EMA=0.98、在线补标、UniMatch、固定验证、每 5 轮预览、总共 200 轮。

从头运行：

~~~bash
bash train_unimatch_wsl.sh
~~~

相当于：

~~~bash
bash train_wsl.sh \
  --resDir train_v5_unimatch/ \
  --pu 0 \
  --emaTeacher \
  --emaDecay 0.98 \
  --epochs 200 \
  --onlineCorrection \
  --fixedVal \
  --unimatch \
  --unimatchStart 41 \
  --unimatchRampEnd 80 \
  --unimatchWeight 0.1 \
  --emaPreview \
  --previewEvery 5
~~~

原有 train_wsl.sh 没有修改。不开 --unimatch 就继续走原训练流程。新脚本可在后面追加参数覆盖默认值，例如：

~~~bash
bash train_unimatch_wsl.sh --resDir train_v5_unimatch_trial/ --epochs 80
~~~

结果目录已有模型时，不能无声重新开始；需要新目录或正常续训。

### 正常中断后续训

~~~bash
bash train_unimatch_wsl.sh \
  --resDir train_v5_unimatch/ \
  --epochs 200 \
  --emaResume train_v5_unimatch/ema_checkpoint.pth
~~~

如果首次启动调整过数据、增强、起始轮次、lambda 或遮挡开关，续训必须保持相同参数。校验会拒绝无声更改实验。优化器、教师、待累积梯度和随机数状态随 checkpoint 恢复。

### 从旧第 40 轮做探索性接入

只有此前未启用在线补标/UniMatch 的预热 checkpoint 可以显式转入新实验：

~~~bash
bash train_unimatch_wsl.sh \
  --resDir train_v5_unimatch_from40/ \
  --emaResume train_v5_ema_smoke/ema_checkpoint.pth \
  --emaStageTransition
~~~

正常续训时去掉 --emaStageTransition，改为新实验自己的 checkpoint。不要把已经在线训练完 200 轮的 checkpoint 直接当作这次实验的预热起点。

这条命令仅适合沿用旧数据划分的探索。若重划空间独立验证区，旧 checkpoint 可能见过该区域，需要从相同预训练初始化重新建立预热模型和对照组。

### 区分两种新增设计的对照

- A：原 train_wsl.sh + EMA + onlineCorrection，不加 --unimatch。
- B：新脚本加 --unimatchNoDropout，保留两个强视图，关闭互补特征遮挡。
- C：新脚本默认配置，两个强视图 + 互补特征遮挡。

三组用不同 resDir，并保持数据、初始化/预热 checkpoint、更新次数和评估口径一致。B 的例子：

~~~bash
bash train_unimatch_wsl.sh \
  --resDir train_v5_unimatch_images_only/ \
  --unimatchNoDropout
~~~

## 看哪些结果

以 train_v5_unimatch/ 为例：

- unimatch/epoch_0041.json：原支路损失、强视图损失、lambda、总损失、实际参与的 batch 和像素数量。
- unimatch/epoch_0041.png：弱视图、两个强视图、原标签、接纳的临时标签、监督权重、两个强视图预测与差异。
- unimatch/epoch_0041.npz：上述实际训练张量及该样本四个尺度的互补通道掩膜。只有实际使用了强视图且需要预览时才写图/npz。
- online/：原有的真实临时补标诊断。
- preview/：原有固定裁剪预览；其旧预览候选不等于真正接纳的训练候选。
- best_student/、best_teacher/：按原标签固定验证 loss 选出的最佳模型。
- ema_checkpoint.pth：最新完整断点。
- loss.txt：现在训练列包含一致性损失，验证列仍是原标签 FocalTversky。新旧实验应比较同口径验证/测试指标，不直接比较总训练 loss。
- numerical_failure.json：遇到非有限输入、预测、损失或梯度时的定位信息。

可靠像素/候选像素统计会重复计算训练裁剪，不代表唯一断层数量。预览里的强视图预测来自训练中的特征遮挡支路，不等于最终正常推理结果。

现有评估入口仍然可用，例如：

~~~bash
source ./activate_wsl.sh
python evaluate_cv.py \
  --model train_v5_unimatch/best_student/model.pth \
  --outDir train_v5_unimatch/cv_evaluation/student
~~~

验证和测试始终使用原始标签，不用模型补出的标签给自己评分。

## 已知实验限制

当前 earth_dataset_v3 的训练和验证图块存在空间重叠。训练功能没有自动改写旧划分；正式声称性能改善前，需要建立空间独立验证划分并重跑公平对照。现有测试区已经评估过，后续不要不断根据它调参数。

正标签一致性不保证消除错误补标，也可能扩大正类预测，需检查验证 Precision/AP、预测面积以及人工复核的补标结果。默认阈值、增强及 lambda 都只是起点。验证 loss 更低不等于已经证实漏标恢复率更高。

## 自检

单元与集成测试：

~~~bash
source ./activate_wsl.sh
python -m unittest test_unimatch_fault test_aio2_ema test_aio2_online test_aio2_preview test_training_numerics -v
~~~

真实模型单批次检查（会创建单独的测试输出，不保存正式模型）：

~~~bash
source ./activate_wsl.sh
python smoke_unimatch.py --outDir train_v5_unimatch_smoke_new/
~~~

smoke 脚本要求空输出目录。它执行真实的教师预测、原支路反向、两个强视图反向、一次优化器更新、一次 EMA 更新和更新后推理。它仅检查代码，不作为模型精度实验。

## 文件改动

- 新增 unimatch_fault.py：强视图、互补掩膜、正标签一致性、分支反向和诊断。
- 新增 train_unimatch_wsl.sh：独立启动入口。
- 新增 test_unimatch_fault.py、smoke_unimatch.py：逻辑/续训与真实模型验证。
- 扩展 config.py、trainV10.py、torchUnet.py、aio2_ema.py。
- 改动前四个文件保存在 _backup_before_unimatch_20260913_234211/。
- 旧模型、旧训练结果、原始 HDF5、PU 设计和 SAM-2 官方源码未修改。

方法来源：论文第 3.3 节，以及 https://github.com/LiheYoung/UniMatch-V2 。

## 本次验证记录

- 38 项不同测试通过：原有 28 项与新增 10 项。
- 真实 SAM-2、CUDA、batchsize=2 的原支路和双强视图反向均通过；单批检查峰值分配显存约 2.01 GiB，不代表长训练的显存上限。
- 普通 forward 与改动前源代码在真实模型上逐元素完全相同。
- 实际 WSL 入口使用 6 个训练图块、2 个验证图块完成一轮，然后从完整断点续训到第 2 轮；两轮均有 3/3 个 batch 执行强视图，教师累计更新 2 次。
- 初始真实模型没有接纳临时候选；候选降权和正向监督的行为由合成案例测试覆盖。短测试不代表补标准确率或 AP 提升。
- 未启动 200 轮正式训练。详细检查记录见 train_v5_unimatch_smoke/verification.json。
