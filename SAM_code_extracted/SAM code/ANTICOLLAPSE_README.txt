V5/V6 损失与监督掩膜改造

适用范围：主目录 trainV10.py；V5 earth_dataset_v3 和 V6 希腊/新西兰。
V5_different_split 的训练循环未改动，仅保持共享 sam2_lora.py 镜像一致。

1. 新接口
   net(image) 仍返回三通道概率；return_logits=True 返回原始 logits。
   --lossFunc BCEWithLogits 或 BCEFocalTversky（后者默认 --tverskyWeight 0.5）。
   新损失仅支持 --k0Only，--pu 必须为 0；没有实现标准 PU 风险估计。
   旧损失仍可用于对照；默认损失未改。
   --consistencyLoss logits 为可选的稳定正类一致性分支；不是负类伪标签方法。

2. 一张图上的陆地、浅海、深海
   数据 HDF5 的每个 tile 可增加同尺寸二维 uint8 数据集，例如 label_known。
   值 1：该像素的 k0 标签状态可监督（已知断层或经过核查的背景）。
   值 0：未知，不参与直接监督损失及监督评价。
   使用 --supervisionKey label_known 启用。缺少该字段或非二值掩膜会报错。
   掩膜与输入同步裁剪、缩放、D4 变换；掩膜缩放用最近邻插值。
   一旦启用，所有训练和验证 tile 都必须具备这个字段。
   不要把掩膜传成神经网络输入通道，不按水深自动判断已标注程度。
   可以共同监督浅海和陆地，未标注深海置 0；浅海/陆地尚未核查区域同样应置 0。
   已知断层缓冲带及距离图仍由数据制备决定，掩膜不会自动创建或补齐断层标签。
   无掩膜时仍使用“目录外像素作为背景”的旧假设，并在训练启动时明确打印。
   验证包含有效但完全未知的裁剪时跳过；整个验证集都未知则报错。
   完全没有监督/一致性信号的累积窗口不执行 AdamW 或 EMA 更新。

3. 检查点和诊断
   默认 accumulationVersion=window_v2：完整累积窗口更新，最后不足窗口正确归一化。
   默认 collapseGuard 开启，拒绝将阈值后全正/全负的退化模型选为 best。
   不会删掉 model.pth 或诊断日志；早期没有合格 best 是正常情况。
   near_collapse 是预警（正类比例 >99% 或 <0.1% 且标签同时有正负），不直接宣称训练失败。
   fixed_val.jsonl 保存学生/教师健康指标、常数预测基线及所用监督假设。
   BCE 基线使用 binary logit +/-20，旧概率损失使用精确 0/1。
   adaptation_history.jsonl 保存训练监督指标及验证健康指标。
   现有旧权重仍可推理。普通续训检查损失、混合权重、监督掩膜和训练实现版本；
   不允许用旧实验状态静默切换成新监督逻辑。新实验应从原始 SAM-2 开始并用新目录。

4. 小规模优化对照（在 WSL 的 SAM code 目录执行）
   Python: /home/ovvvin/.venvs/earthquake-sam/bin/python

   数据尚未补标时，只用于数值对照，不报告地质识别精度：
   python smoke_anticollapse.py --dataDir earth_dataset_v3 --outDir diagnostics/v5_loss_ablation --catalog-background-ablation --steps 100
   python smoke_anticollapse.py --dataDir earth_dataset_greece_nz_stride64_v2 --outDir diagnostics/v6_loss_ablation --catalog-background-ablation --steps 100

   有已核查的监督掩膜以后：
   python smoke_anticollapse.py --dataDir YOUR_DATA --outDir diagnostics/masked_loss_ablation --supervisionKey label_known --steps 100

   默认比较 ft、ft_low_lr、bce、mixed 四组；可用 --arms mixed 单独运行。
   固定 seed=7、d0=0.01、r=8、alpha=16，每组重置模型和随机数；无增强、无一致性。
   通常选 8 个固定训练裁剪、4 个验证裁剪；具体数量及 key 记录在 report.json。
   每步为一个 batch 的 optimizer 更新，用于小样本优化诊断，不等同于原实验 epoch。
   每组保存 steps.jsonl、model.pth、comparison.png，根目录保存 report.json。
   程序检查基础 SAM 冻结、LoRA 更新和严格重载；数值检查通过不代表不塌缩。
   请同时检查 final_health、F1、正/负类概率和图像结果；不同损失的数值不可直接横比。

5. 真实入口集成测试（复制临时数据、人工构造掩膜，仅验证程序）
   python verify_fault_training.py --outDir diagnostics/fault_cli_check
   覆盖 BCE、混合损失+EMA+强增强、续训、错误配置拒绝、普通验证、轮末梯度清空。

   earth_dataset_v5 的四组 100 次更新对照使用新入口：
   python run_masked_loss_ablation.py --dataDir earth_dataset_v5 --outDir diagnostics/v5_masked_loss_ablation_100 --steps 100
   必须使用不存在的新输出目录。本入口固定启用 label_known，关闭 EMA、伪标签、一致性和增强。
   选 8 个训练中心裁剪和 4 个验证中心裁剪；每个裁剪至少 1000 个已知像素、256 个背景像素。
   优先覆盖不同位置；有足量纯背景裁剪时包含一个。实际选择记录在 selection.json。
   每次更新使用 8 张：microbatch=2，累积 4 次，损失除以 4，然后裁剪梯度并更新。
   四组均从相同原始 SAM 和自定义层初始化开始，使用相同固定样本顺序。
   ft / bce / mixed: head LR=1e-4, LoRA LR=5e-5；ft_low_lr: 2e-5, 1e-5。
   mixed 是 BCEWithLogits + 0.5 * FocalTversky，仍使用旧版 FocalTversky 的 alpha/beta/gamma。
   每 10 次更新评估全部固定样本；第 0、20、50、100 次保存逐图预测，统一 0–1 色阶。
   index.html 为逐图对照和演化图库，metrics.png 为指标曲线，report.json 为最终结果。
   可重新渲染：python build_loss_ablation_gallery.py diagnostics/v5_masked_loss_ablation_100
   当前已批准的几何包围范围不是经独立核验的调查完整范围；掩膜外预测不计算准确率。
   旧划分存在训练/验证空间重叠，此诊断不作为独立泛化测试，也不访问 test 划分。

6. V6 启动器示例（先只打印命令）
   python run_greece_nz_training.py --trainingRecipe supervised --samAdaptation lora --lossFunc BCEFocalTversky --tverskyWeight 0.5 --lr0 1e-4 --seed 7 --epochs 5 --res-dir NEW_RUN --supervisionKey label_known
   只有数据中已提供 label_known 才能使用上述掩膜模式；加 --execute 才开始训练。
   新数据训练前仍需独立核查标签覆盖、可靠背景和空间划分。
