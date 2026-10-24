# 2026-10-23 项目更新：LoRA、混合损失与有效监督区域

本次整理当前项目代码并记录已完成的诊断和待运行配置。代码在 `SAM_code_extracted/SAM code/`，下文未特别说明的路径均相对于该目录。历史 LoRA 实现已于提交 `14c2c41` 上传；本次在其基础上补充损失、监督范围和训练管理，不将既有 LoRA 实现误记为本次首次增加。

## 1. 冻结 SAM-2，训练 LoRA 和自定义分割层

- SAM-2 原始参数冻结；向 Hiera 注意力的 Q/V 投影注入 LoRA，K 投影不加入 LoRA。`loraStages=all`。
- `samAdaptation=lora, loraRank=8, loraAlpha=16`，缩放系数 `alpha/r=2`；`loraDropout=0, loraWeightDecay=0`。
- LoRA B 初始为零，保留预训练输出；自定义输入适配和分割层继续训练。
- 基础学习率：自定义层 `lr0=1e-4`，LoRA `loraLr=5e-5`，分别建立优化器参数组。
- EMA 教师维护学生模型的移动平均，包括 LoRA 与其他可训练参数；仅在有效 optimizer 更新后更新，教师不反向传播，SAM 基础权重继续冻结。
- `emaDecay=0.98`。EMA 从第一轮开始，后述 120／40 轮边界是伪标签和一致性训练的延迟，不是 EMA 的启动时间。

实现：`sam2_lora.py`、`torchUnet.py`、`aio2_ema.py`。保留 `full/frozen/lora` 三种模式；裸命令默认仍为历史模式，当前 JSON 显式选择 `lora`。

## 2. 新增稳定的混合损失

当前正式配置：`lossFunc=BCEFocalTversky, tverskyWeight=0.5, k0Only=true, pu=0`。

```text
z = logits[:, :1] - logsumexp(logits[:, 1:], dim=1)
p_fault = sigmoid(z) = softmax(logits)[:, :1]
L_supervised = weighted_BCEWithLogits(z, y) + 0.5 × FocalTversky(y, p_fault)
```

BCE 在有效权重像素上按权重和归一化；Focal Tversky 继续使用原实现的 `tv_alpha=0.3, tv_beta=0.7, gamma=0.75`，`pu=0`。这不是标准 PU 风险估计，也没有新增大幅正类加权。保留原 Focal Tversky 和单独 BCE 以便做对照。

`net(image)` 仍返回三通道概率，兼容旧推理、阈值和图库；`net(image, return_logits=True)` 用于新损失。学生、EMA 教师验证和强增强分支同步支持 logits。正类一致性可选择 `consistencyLoss=logits`，以 `softplus(-z)` 计算；它仍是正类一致性，不包含可靠负类伪标签，不等同于完整复现原始 UniMatch。

目标由距离图 `k0 < d0` 得到，`d0=0.01`。距离权重保留 `dw=0.08, alpha=0.1`，与 Focal Tversky 的 `tv_alpha` 不是同一参数。

## 3. label_known：未知像素不再直接作为背景

HDF5 每个 tile 新增同尺寸二值字段 `label_known`：`1` 表示该位置可按现有 k0 监督，`0` 表示未知。未知及无效输入位置监督权重为零，不参与监督损失和监督指标；掩膜不是神经网络输入通道。裁剪和 D4 变换与图像、标签同步，缩放掩膜使用最近邻。

V5 从 `earth_dataset_v3` 复制制作成新目录 `earth_dataset_v5`，保留原数组和划分清单；不覆盖 v3。当前掩膜依据已批准的密集沿海标注包围范围生成：EPSG:3310 下缓冲 8 km、闭运算 4 km，排除陆地及无效范围，**没有将 200 m 等深线直接当作调查边界**。它是人工接受的几何近似，不是独立核验的调查完整性证明。将来补充陆地标注时应同时更新覆盖范围和 mask。

V5 配置启用 `supervisionKey=label_known, knownCrop=true, minKnownPixels=1, knownCropStride=32`：

- 原 train/val 清单分别为 161／17 个 tile；当前本地预检查保留 80／10 个含有效监督区域的 tile，均未改写清单。
- 输入裁剪为 256×256；候选起点步长 32，含边缘起点。训练按已知像素数加权抽取裁剪，再做同步 D4；该模式不做缩放增强。
- 验证固定选每个 tile 已知像素最多的裁剪，不随机增强。原训练／验证地理重叠仍存在；指标用于监控，不能视为严格空间独立测试。
- `minKnownPixels=1` 是保留所有有监督 tile 的选择，不表示每个裁剪都有充分标注。
- 直接监督忽略未知区域，但在线正类伪标签仍可进入有效的未知区域；已知标签受保护。mask 并不禁止未知区域参与半监督训练。

V6 沿用 `earth_dataset_greece_nz_stride64_v2`，不启用 `label_known`，保留原随机裁剪／缩放／D4 及固定中心验证。当前本地为 312 个训练 tile、69 个验证 tile；未标注像素作为背景的原假设仍存在，混合损失不会自动修复漏标。

## 4. 训练稳定性、诊断和实验隔离

- 修正梯度累积：完整窗口结束才更新，尾部不足窗口按实际长度归一化；强增强使用同一除数。`accumulationVersion=window_v2`，batch=2、累积=4，完整窗口有效 batch=8。
- 无监督及无一致性信号的窗口不执行 AdamW/EMA 更新。无效像素不会再获得背景监督权重。
- `collapseGuard=true`：保存全正／全负基线、Precision/Recall/F1、正类面积、正负概率及塌缩记录；退化模型不作为通过检查的 best，诊断及普通检查点保留。防护不是训练不会塌缩的保证。
- `fixedVal=true`；预览每 5 epoch，完整检查点每 20 epoch。预览未知区灰显，指标只统计已知有效像素。
- 配置与检查点记录损失、监督方式、累积版本及数据／代码摘要；普通 resume 不允许静默混用旧训练状态。
- 新 `dataFile` 明确指定 HDF5；文件不存在即报错，不再悄悄回退到 v3。
- `run_experiment.py` 默认只检查。执行时独占创建新输出目录，拒绝已有目录；V5 退出码为零且历史末轮、EMA 检查点轮次／配置／数据身份通过核验后才启动 V6。失败、中断或源代码／数据改变会停止队列，不自动重试。

## 5. 正式配置与参数

共享配置：`experiments/common_bceft_lora_ema_unimatch.json`。两组均从原始 `sam2_hiera_small.pt` 和对应随机初始化开始，V6 **不加载 V5 权重**。

| 参数 | V5：`california_ds5.json` | V6：`greece_nz_original.json` |
|---|---|---|
| 数据 | earth_dataset_v5 | earth_dataset_greece_nz_stride64_v2 |
| epochs | 500 | 200 |
| label_known | 开启 | 关闭 |
| onlineStart | 121 | 41 |
| onlineHoldEnd | 150 | 50 |
| onlineRampEnd | 240 | 80 |
| unimatchStart / unimatchRampEnd | 121 / 240 | 41 / 80 |
| 当前预检查更新次数/epoch | 10 | 39 |
| 新输出目录（位于 runs/） | california_ds5_bceft_lora8_ema_unimatch_seed7_r01 | greece_nz_bceft_lora8_ema_unimatch_seed7_r01 |

| 共同参数组 | 数值 |
|---|---|
| 复现与模型 | seed=7, modelType=mars_sam, mn=256, samAdaptation=lora |
| LoRA | rank=8, alpha=16, dropout=0, stages=all, weightDecay=0 |
| 学习率 | lr0=1e-4, loraLr=5e-5 |
| 监督 | lossFunc=BCEFocalTversky, tverskyWeight=0.5, k0Only, pu=0, d0=0.01, dw=0.08, alpha=0.1 |
| batch 与 EMA | batchsize=2, stepPerMiniBatch=4, emaTeacher, emaDecay=0.98 |
| 伪标签 | onlineCorrection, onlineWeightStart=0.05, onlineWeightMax=0.2 |
| 候选筛选 | onlineThreshold=0.9, onlineSupport=0.5, onlineBuffer=3, onlineMinPixels=16, onlineMaxFraction=0.05 |
| 一致性 | unimatch, consistencyLoss=logits, unimatchWeight=0.1 |
| 强增强 | contrast=0.2, brightness=0.1, noise=0.03 |
| 预览 | count=6, every=5, threshold=0.7, buffer=2, minPixels=8 |
| 检查点 | checkpointEvery=20, fixedVal, collapseGuard, accumulationVersion=window_v2 |

V5 第 1–120 轮仅监督＋EMA；第 121–150 轮伪标签权重保持 0.05，此后爬坡至第 240 轮的 0.2；一致性从第 121 轮爬坡至第 240 轮的 0.1。V6 保留原 41／50／80 调度。

学习率沿用原 epoch 调度，并非全程常数：第 1 轮按 batch 热身，第 2 轮基础倍率 1，第 3–4 轮 0.5，第 5–8 轮 0.25，第 9 轮起 0.125；LoRA 与自定义层使用这一倍率。EMA／伪标签延迟的调整没有将学习率节点同步乘以 3。运行时完整有效参数写入 `resolved_config.json`。

## 6. 已完成的四组小规模对照

使用 earth_dataset_v5 的 8 个固定训练裁剪＋4 个验证裁剪，启用 label_known；关闭 EMA、伪标签、一致性和增强。各组同 seed=7、同初始化和样本顺序，各 100 次 optimizer 更新（不是 100 epoch），microbatch=2、累积=4，使用常数学习率。阈值为 0.5，仅统计已知有效像素。

| 组别 | 自定义层 / LoRA 学习率 | 最终验证 F1 | 验证预测正类比例 | 最终状态 |
|---|---|---|---|---|
| 原 Focal Tversky | 1e-4 / 5e-5 | 0.4462 | 100% | 全正预测 |
| 原 Focal Tversky，LR×0.2 | 2e-5 / 1e-5 | 0.4462 | 100% | 全正预测 |
| BCEWithLogits | 1e-4 / 5e-5 | 0.3539 | 12.55% | 未全正／全负 |
| BCE＋0.5×Focal Tversky | 1e-4 / 5e-5 | 0.5130 | 20.04% | 未全正／全负 |

验证已知区域标签正类比例为 28.71%，全正预测 F1 基线为 0.4462。混合损失在此对照中优于该基线，但小样本、空间重叠及 mask 几何假设限制结论；尚未证明在所有数据、未知区域或长程半监督训练中不会塌缩。四组均通过 SAM 原始权重不变、LoRA 参数更新及严格重载预测一致性检查。

结果依据本地 `diagnostics/v5_masked_loss_ablation_100/report.json`，本次仅发布汇总，不上传检查点和完整结果目录。复现入口为 `run_masked_loss_ablation.py`，图库入口为 `build_loss_ablation_gallery.py`。

## 7. 代码位置与运行

| 文件 | 作用 |
|---|---|
| sam2_lora.py、torchUnet.py | LoRA、参数组、模型配置和 logits 接口 |
| fault_losses.py、fault_metrics.py | 统一监督损失／mask、累积窗口、塌缩指标 |
| trainV10.py、config.py | 训练循环、新参数与约束 |
| tool.py、known_crops.py | 掩膜读取、同步变换、有效监督裁剪 |
| aio2_ema.py、aio2_online.py、aio2_preview.py | 教师 logits、验证／选模、伪标签和掩膜预览 |
| unimatch_fault.py | 稳定的正类 logits 一致性 |
| experiments/*.json、run_experiment.py、run_v5_then_v6.ps1 | 两组参数、预检查和串行执行 |
| run_greece_nz_training.py | 原 V6 入口透传损失、学习率和监督参数 |
| audit_v5_label_coverage.py、preview_v5_coastal_envelope.py、build_earth_dataset_v5.py | 覆盖审计、范围预览、新数据集制作 |
| smoke_anticollapse.py、run_masked_loss_ablation.py、build_loss_ablation_gallery.py | 对照与预测图库 |
| test_fault_losses.py、test_experiment_launcher.py、verify_fault_training.py | 数值、mask、队列和入口检查 |

`V5_different_split/sam2_lora.py` 仅同步共享工具镜像；该目录的独立训练循环未迁移到本次新流程。

准备依赖（参见 `requirements-wsl.txt` 和仓库 WSL 安装说明），在代码目录放置原始 SAM-2 权重和两个本地数据集，各数据集需要 `faults.h5`、train/val 清单及 manifest。源码库不含这些输入。数据制备辅助脚本还依赖原始 `ca_offshore.shp`、GEBCO 文件和 v3 manifest；应先运行范围预览并人工核查，再运行 builder。builder 读取预览配方，拒绝覆盖已有 v5。

```bash
cd "SAM_code_extracted/SAM code"

# 只检查：不构建正式训练，不创建实验结果目录
python run_experiment.py --sequence experiments/california_ds5.json experiments/greece_nz_original.json

# 需要正式运行时才执行这一行；本次上传未执行
python run_experiment.py --sequence experiments/california_ds5.json experiments/greece_nz_original.json --execute

# 单元检查：不启动这两组正式训练
python -m unittest test_experiment_launcher test_aio2_preview test_aio2_ema test_aio2_online test_fault_losses test_sam2_lora test_unimatch_fault -q

# 可选：重新跑四组小规模诊断（会训练，输出目录必须不存在）
python run_masked_loss_ablation.py --dataDir earth_dataset_v5 --outDir diagnostics/my_new_ablation --steps 100
```

Windows 包装器 `run_v5_then_v6.ps1` 不加参数时只检查，加 `-Execute` 才运行。它使用本机的 Ubuntu-22.04 和 `/home/ovvvin/.venvs/earthquake-sam/bin/python`；其他机器请调整环境或直接使用上方 Python 入口。串行启动器需保持运行，不是系统后台调度服务。

截至本次更新，60 项单元检查通过，两组配置的数据预检查通过，正式结果目录尚未创建。没有启动新 500／200 epoch 训练，没有覆盖原数据和旧结果。Git 仅同步源码、测试、参数配置及说明，保留远端已有的历史数据清单。
