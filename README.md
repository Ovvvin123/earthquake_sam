# Earthquake-SAM：冻结 SAM-2 + LoRA + 掩膜混合损失

## 当前更新：2026-10-23

本次在已有 **冻结 SAM-2＋Q/V LoRA** 的基础上，加入 **BCEWithLogits＋0.5×Focal Tversky**、`label_known` 有效监督掩膜、梯度累积修正、塌缩诊断，以及 **V5 完成后再运行 V6** 的配置化启动器。完整更新、参数、代码位置和检查结果见 [本次项目更新说明](docs/updates/2026-10-23-lora-bceft-label-known.md)。

| 配置 | V5：加州 | V6：希腊＋新西兰 |
|---|---|---|
| 数据集 | `earth_dataset_v5`，启用 `label_known` | `earth_dataset_greece_nz_stride64_v2`，沿用原标注 |
| 正式训练计划 | 500 epoch | 200 epoch，V5 完成并通过检查后启动 |
| 伪标签开始／保持结束／爬坡结束 | 121／150／240 | 41／50／80 |

共同参数：SAM 原始参数冻结，LoRA `r=8, alpha=16, dropout=0`，`seed=7, d0=0.01, pu=0`；基础学习率为自定义层 `1e-4`、LoRA `5e-5`；EMA `decay=0.98`；batch `2`、梯度累积 `4`。EMA 从训练开始随有效 optimizer 更新，不是第 121／41 轮才创建。完整参数以 [`experiments/`](SAM_code_extracted/SAM%20code/experiments/) 的 JSON 为准。

**当前状态：**四组各 100 次更新的监督对照已完成，BCE＋0.5×Focal Tversky 在该小样本检查中未出现最终全正预测；60 项测试通过，正式配置完成数据预检查。新的 500／200 epoch 训练尚未启动，不能将小样本结果视为正式训练或独立泛化结论。

在 `SAM_code_extracted/SAM code/` 中运行 `python run_experiment.py --sequence experiments/california_ds5.json experiments/greece_nz_original.json` **只检查，不训练**；明确加 `--execute` 才串行训练。数据、权重、旧训练结果不随本次代码上传。

本分支同步 Windows 本地交接目录中的源码、配置、数据划分清单和说明文档，包含 WSL 环境、EMA teacher、在线临时补标、预览与训练数值修复相关代码。基于仓库原有提交保留历史；`main` 分支的 Ubuntu 适配版本仍可单独查看。

## 上传版本标注

| 上传版本 | Git 标签 | 训练方式 |
|---|---|---|
| 上一次上传（提交 `a8d8938`） | [sam2-trainable-2026-10-10](https://github.com/Ovvvin123/earthquake_sam/tree/sam2-trainable-2026-10-10) | **SAM-2 参与训练**：SAM-2 参数参与原有微调，自定义分割层同时训练。 |
| 历史上传（2026-10-11） | [sam2-frozen-lora-2026-10-11](https://github.com/Ovvvin123/earthquake_sam/tree/sam2-frozen-lora-2026-10-11) | **SAM-2 冻结 + LoRA**：LoRA 模式冻结 SAM-2 原始参数，训练 Q/V LoRA 和自定义分割层。 |
| 当前更新（2026-10-23） | 当前分支；[更新说明](docs/updates/2026-10-23-lora-bceft-label-known.md) | **LoRA＋掩膜混合损失**：新增有效监督、塌缩诊断及 V5→V6 独立配置。 |

本次版本仍保留 `full`、`frozen`、`lora` 三种模式，命令行默认保持 `full` 以兼容旧命令。启用本次标注的训练方式，请使用 `bash train_lora_wsl.sh` 或显式传入 `--samAdaptation lora`；仅下载此版本不会自动改变旧命令的训练模式。完整命令见下方 LoRA 说明。

## 从这里开始

- [WSL 环境与运行说明](WSL环境与运行说明.md)：环境安装、激活、验证和训练方法。
- [项目交接说明](项目交接说明_给学生.md)：研究背景、数据与历史实验说明。
- [EMA 使用说明](SAM_code_extracted/SAM%20code/AIO2_EMA_使用说明.md)。
- [在线临时补标](SAM_code_extracted/SAM%20code/AIO2_在线临时补标.md)。
- [预览与诊断](SAM_code_extracted/SAM%20code/AIO2_预览与诊断.md)。
- [SAM-2 冻结与 LoRA 训练](SAM_code_extracted/SAM%20code/LoRA_训练与验证.md)：V5/V6 适配模式、对照训练、断点恢复和验证命令。
- [UniMatch 断层适配说明](SAM_code_extracted/SAM%20code/UniMatch_断层适配_使用说明.md)：双强视图、互补特征遮挡及正标签一致性训练。
- [独立网格划分实验](SAM_code_extracted/SAM%20code/V5_different_split/运行说明.txt)：不重叠切片、随机训练/验证划分及按更新次数调度的训练器。
- [全球多区域运行步骤](v6_全球多区域_运行步骤.md)。

核心代码和 WSL 脚本位于 `SAM_code_extracted/SAM code/`。本分支保留本地交接版的文件布局和源码，不包含只存在于 Ubuntu main 分支的根目录训练包装脚本；运行时以本分支 WSL 说明为准。

## Git 同步范围

2026-10-10 源码同步新增 UniMatch 适配、希腊/新西兰数据准备与训练、独立网格实验，以及预测导出、断层计数、标签审计和绘图脚本。希腊/新西兰流程入口为 `SAM_code_extracted/SAM code/prepare_greece_nz.py` 和 `run_greece_nz_training.py`；绘图辅助代码位于 `input_channels_figures/`。实验控制脚本及部分分析脚本依赖本地已有数据或训练结果，克隆源码后需按脚本说明准备输入。

本次仅同步源码及说明；新增实验的 HDF5、训练结果、模型权重和本地生成的数据清单需自行准备或重新生成。原有根目录 `earth_dataset/` 的历史清单保留在 Git 中，不代表本机仍有对应数据。

Git 保存源码、测试、依赖清单、说明文档、数据划分清单及已有的小型参考资料。模型权重（`.pt`、`.pth`）、数据集（`.h5`）、地形数据（`.nc`）、训练产物、日志、缓存及本地备份不上传，仍保留在本地。

克隆代码后需要自行放置数据与权重，至少包括：

- `SAM_code_extracted/SAM code/sam2_hiera_small.pt`
- `SAM_code_extracted/SAM code/earth_dataset_v3/faults.h5`
- 评估历史 v5 时需要 `SAM_code_extracted/SAM code/train_v5_local/model.pth`

数据重建还需要 GEBCO 原始地形。说明文档中的本机绝对路径需按自己的环境调整。上方历史版本的实验结论不由本次同步重新验证；2026-10-23 更新的测试和小规模对照范围见更新说明。
