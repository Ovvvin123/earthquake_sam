# SAM-2 冻结与 LoRA 训练

适用于主目录的 V5/V6 训练器，以及 `V5_different_split` 独立实验。
默认仍为 `--samAdaptation full`；LoRA 必须显式启用。没有自动启动正式训练。

## 模型与配置

| 模式 | SAM-2 原权重 | Q/V LoRA | 输入适配层、融合层和卷积分割头 |
|---|---|---|---|
| `full` | 沿用原有微调 | 无 | 训练 |
| `frozen` | 冻结 | 无 | 训练 |
| `lora` | 冻结 | 训练 | 训练 |

第一版只支持 `mars_sam`、不启用 `isalign` 的冻结/LoRA 实验。
七通道输入、三通道输出与 k0 监督不变；LoRA 不改变标签与数据划分。

默认参数：`--loraRank 8 --loraAlpha 16 --loraDropout 0 --loraStages all
--loraLr 5e-5 --loraWeightDecay 0`。`loraAlpha` 与损失参数 `--alpha` 无关。
可选 `--loraStages last2` 只适配 Hiera 后两个阶段。

在 Hiera 合并的 QKV 层上，为 Q、V 各加独立低秩分支，K 不变。
A 随机初始化，B 零初始化；注入保持全局随机数状态。
冻结所有原始 SAM 参数，包括 neck 和原始解码器；自定义 `head`、
`input_sam`、多尺度融合、上采样、`tail` 继续训练。
训练不合并权重；学生 SAM 前向保留计算图。教师在注入后创建。

SAM-2 small、全部 16 个 Hiera block、r=8 的实测参数量：
总计 46,733,832；可训练 689,990，其中 LoRA 188,928、自定义层 501,062。
没有精度提升保证；少量参数不等于按同样比例减少训练时间或显存。

## 运行环境

以下命令在已有 WSL Ubuntu-22.04 环境执行。先进入对应目录并激活环境：

```bash
cd '/mnt/f/Study————————/UCLA/Earthquake-SAM_交接/SAM_code_extracted/SAM code'
source activate_wsl.sh
```

## 主目录 V5：纯监督对照

以下三次实验的数据、d0、损失、增强和预算相同，只改变适配方式。
输出目录应不存在或为空。

```bash
bash train_lora_wsl.sh --samAdaptation full   --resDir train_v5_ablation_full/   --seed 7 --epochs 200
bash train_lora_wsl.sh --samAdaptation frozen --resDir train_v5_ablation_frozen/ --seed 7 --epochs 200
bash train_lora_wsl.sh --samAdaptation lora   --resDir train_v5_ablation_lora/   --seed 7 --epochs 200
```

这个包装脚本使用 `d0=.01`、`pu=0` 和固定验证，覆盖原 V5 脚本的 PU 默认值。
需要对齐旧 d0=.02 实验时，三组都加 `--d0 .02`。不要混用标签阈值比较结果。

完整教师流程可使用：

```bash
bash train_unimatch_wsl.sh --samAdaptation lora --loraRank 8 --loraAlpha 16 \
  --loraLr 5e-5 --d0 .01 --seed 7 --resDir train_v5_unimatch_lora_r8/
```

## 独立划分 V5

此实验保留自己的数据、默认 d0=.02、3200 次更新预算和全切片验证。

```bash
cd V5_different_split
bash train_lora_wsl.sh --resDir runs/lora_r8_seed7/ --seed 7
# full / frozen 对照使用同一脚本并覆盖 --samAdaptation 与 --resDir。
bash train_unimatch_wsl.sh --samAdaptation lora --resDir runs/teacher_lora_r8_seed7/ --seed 7
```

`train_grid.py` 根据 `maxSteps` 控制预算，学习率/纠标/一致性仍按已有步数调度。
本实验只有区域内验证集，不应把它称作独立测试集。

## V6：希腊＋新西兰

回到主目录。启动器默认只打印命令，加入 `--execute` 才训练。

```bash
python run_greece_nz_training.py --trainingRecipe supervised --samAdaptation lora \
  --res-dir train_greece_nz_supervised_lora_r8 --seed 7 --d0 .01 --epochs 200

# 正式执行上述配置时，在末尾加 --execute。
# 以下为现有 EMA + 在线纠标 + UniMatch 流程的 LoRA 版本：
python run_greece_nz_training.py --trainingRecipe unimatch --samAdaptation lora \
  --res-dir train_greece_nz_unimatch_lora_r8 --seed 7 --d0 .01 --epochs 200
```

`full`、`frozen` 对照使用相同 recipe 和其他参数。V6 的正式数据检查仍保留，
加州不进入训练或选参。步长 64 的密集切片不等于独立样本数增加。

## 学习率、日志与保存

优化器参数组按 `head`、`sam`、`lora` 命名，仅包含可训练参数，启动时核验
无重复/遗漏。冻结模式只有 head 组。原始 full 模式的主干学习率曲线保留。
LoRA 使用独立基础学习率并沿用头部衰减因子；不再被原主干 1/8 比例覆盖。
非零 LoRA dropout 在学生训练时启用，教师/验证时关闭。

每次运行保存：

- `model_config.json`：完整运行参数，推理构建所需配置。
- `adaptation_report.json`：注入层、参数量、SAM 基座 SHA256。
- `model.pth`：完整学生模型，包含 SAM 原权重、LoRA 和自定义层。
- 主训练器 `adaptation_history.jsonl`：更新步数、分组学习率、梯度范数、
  预测均值/正像素比例、每轮耗时和 CUDA 峰值分配。
- grid 训练器 `history.jsonl`：相同核心诊断，另有每次更新耗时和验证指标。
- 完整检查点记录数据/基座指纹、优化器、随机数状态及训练进度。

主训练器的 `model.pth` 仍代表末轮；固定验证最优学生在 `best_student/model.pth`，
最优教师在 `best_teacher/model.pth`。grid 的 `model.pth` 是最优学生，
`teacher/model.pth` 是最优教师。所有适配方式采用同一选模口径比较。

主训练器保留历史梯度累积规则（含跨轮待更新梯度），grid 保留自己的完整组/末组
归一化规则。这次没有将两套训练器的更新语义强行统一。

## 中断恢复

必须使用同样的适配方式、rank、阶段、数据与核心训练参数。

```bash
# 主目录纯监督：沿用原命令，再加
--resume train_v5_ablation_lora/training_checkpoint.pth

# 主目录教师训练：沿用原命令，再加
--emaResume train_v5_unimatch_lora_r8/ema_checkpoint.pth

# 独立 V5：沿用原命令，再加
--resume runs/lora_r8_seed7/grid_checkpoint.pth

# V6 启动器统一使用 --resume，自动按 recipe 传递正确的恢复参数
python run_greece_nz_training.py --trainingRecipe supervised --samAdaptation lora \
  --res-dir train_greece_nz_supervised_lora_r8 --seed 7 --d0 .01 --epochs 200 \
  --resume train_greece_nz_supervised_lora_r8/training_checkpoint.pth --execute
```

`epochs/maxSteps` 是恢复后的总目标，而非再增加的轮数/步数。
旧 full 检查点缺少新字段时按历史默认值兼容；不允许把 full 权重当作 LoRA
断点直接恢复。模型结构不匹配时严格报错，不忽略缺失参数。

## 评估与导出

评估、EMA 预览和主要出图脚本从 `model_config.json` 重建 LoRA，并严格加载。
复制推理权重时应同时复制该配置文件；模型的 teacher/best 子目录可读取上一级配置。

```bash
# 主目录：此旧评估器报告的 best-F1 阈值是该评估集上搜索的描述性值。
# 若评估测试集，不应把该阈值当作预先选定的部署阈值。
python evaluate.py --resDir train_v5_ablation_lora/best_student \
  --dataDir earth_dataset_v3 --d0 .01

# 独立 V5：固定阈值 .5，并校验训练数据指纹。
cd V5_different_split
python evaluate.py --resDir runs/lora_r8_seed7 --dataDir datasets/ca_stride384 --d0 .02
```

现有 `evaluate_cv.py` 支持验证集选阈值后测试，但旧 USGS train/val 存在重叠，
必须结合其划分审计解释结果。LoRA 不会消除原数据划分的限制。
正式比较建议使用 AP、precision/recall、buffered F1、预测正像素比例，
并按地区报告；选参完成后才评估外部测试集。目录漏标仍影响指标解释。

## 验证工具

```bash
python -m unittest test_sam2_lora test_aio2_ema test_training_numerics test_unimatch_fault -v
python smoke_lora.py --dataDir earth_dataset_greece_nz_stride64_v2 \
  --outDir lora_verification/v6_new --d0 .01
python smoke_lora.py --grid --dataDir V5_different_split/datasets/ca_stride384 \
  --outDir lora_verification/v5_grid_new --d0 .02
python verify_lora_cli.py --outDir lora_verification/cli_new
```

GPU smoke 使用真实数据运行两次监督＋双强增强更新，检查初始输出一致、冻结参数
不变、LoRA/输入层更新、EMA 更新和严格重载预测一致。CLI 工具只复制少量真实切片
到临时目录，验证训练/恢复/评估入口，结束后清理临时数据和权重，保留日志与报告。
这些检查不属于正式精度实验，不能据此认定 LoRA 优于原模型。

实现中 `sam2_lora.py` 在独立 V5 内保留相同副本，单元测试检查内容一致。
修改核心实现时同步两个文件，不依赖第三方 PEFT 包，也不修改安装环境中的 SAM-2 源码。

## 本次验证结果（2026-10-10）

- 主目录 47 项回归测试通过，其中包含 9 项 LoRA 专项测试；独立 V5 的最终 7 项
  训练/数据管线回归通过，包括中断恢复与连续训练的精确一致性。
- V6 的希腊、新西兰各一个真实样本，以及独立 V5 的背景/含断层样本，分别完成
  两次 GPU 更新。原始 SAM 权重完全不变，64 个 LoRA 参数张量发生更新，
  输入适配层更新，EMA 与严格重载检查通过。
- 两组 GPU smoke 峰值分配约 1.80 GiB，包含检查时额外模型的重载，不是纯训练基准。
- 主目录冻结/LoRA 的纯监督训练及续训、LoRA 教师训练及续训、学生/教师评估，
  共 8 个 CLI 检查通过；独立 V5 的 LoRA 训练、续训和评估 3 个检查通过。
- 首次 CLI grid 测试因临时切片在独立目录外被拒绝。测试工具已修正临时目录位置，
  原目录隔离保护保留，重跑的 3 个 grid 检查通过。

本机详细报告：`lora_verification/verification_summary.json`，对应 GPU 报告在
`v6/smoke_report.json`、`v5_grid/smoke_report.json`，CLI 日志在 `cli/` 和 `cli_grid/`。
验证输出不进入 Git。正式 200 轮或 3200 次更新的精度对照尚未启动。
