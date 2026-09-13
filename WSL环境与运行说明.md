# Earthquake-SAM：WSL2 使用说明

安装位置：Ubuntu-22.04，用户 ovvvin。

- 当前项目：Windows F 盘 `F:\Study————————\UCLA\Earthquake-SAM_交接`，在 WSL 中对应 `/mnt/f/Study————————/UCLA/Earthquake-SAM_交接`
- 当前代码：`/mnt/f/Study————————/UCLA/Earthquake-SAM_交接/SAM_code_extracted/SAM code`
- Python 环境：`/home/ovvvin/.venvs/earthquake-sam`
- Codex 和 WSL 现在直接使用 F 盘的同一份代码、数据与输出，无需同步。原 WSL 副本及其中的旧结果保留为历史记录；其 activate_wsl.sh 已改为跳转到 F 盘。

## 每次使用

在 Windows PowerShell 中进入 WSL：

```powershell
wsl -d Ubuntu-22.04
```

在 Ubuntu 中激活项目：

```bash
source "/mnt/f/Study————————/UCLA/Earthquake-SAM_交接/SAM_code_extracted/SAM code/activate_wsl.sh"
```

此命令会进入代码目录、激活环境，并配置无桌面绘图和 CPU 线程数。不需要另装 Conda。

## 验证与评估

```bash
python smoke_test_wsl.py
bash evaluate_wsl.sh
bash evaluate_wsl.sh --keyFile key_test_unseen.lst
python -m unittest test_aio2_ema -v
```

冒烟测试使用真实 v3 数据执行 CUDA 前向、反向和一次优化器更新，不保存训练权重。新评估指标与图件写入 `wsl_validation/v5/`，交接的 `train_v5_local` 原始指标保留。

## 训练

```bash
bash train_wsl.sh
```

默认：v3 数据、v5 参数、batch size 2、梯度累积 4、160 轮，结果在 `train_v5_wsl/`。适配本机 RTX 4060 Laptop 的 8 GB 显存；实际显存需求会随参数、其他 GPU 程序变化。

先做一轮验证：

```bash
bash train_wsl.sh --epochs 1 --resDir wsl_validation/train_one_epoch/
```

EMA 实验可追加 `--emaTeacher --emaDecay 0.99`，并用新的结果目录。不要在不打算覆盖时复用已有训练目录。

## 兼容处理

- SAM2 固定到 Meta 官方提交 `2b90b9f5ceec907a1c18123530e92e794ad901a4`；下载源码逐文件验证 Git blob SHA-1。
- PyTorch 2.7.1 + CUDA 12.6、torchvision 0.22.1、Python 3.10、NumPy 2.2.6。
- `torchUnet.py` 的小模型配置路径更新为 `configs/sam2/sam2_hiera_s.yaml`，对应现有 SAM2 小模型权重。
- 新增 `smoke_test_wsl.py`，指向实际包含的 `earth_dataset_v3/faults.h5`，使用 v5 的 FocalTversky/k0 训练目标。
- 新训练命令省略旧说明中当前解析器没有定义的 `--model`，保留实际支持的 `--modelType`。
- GPU 计算使用 PyTorch 自带 CUDA 运行库和 Windows 显卡驱动。未安装 Linux 显卡驱动或完整 CUDA Toolkit。
- SAM2 的可选连通域 CUDA 扩展未编译；本项目的图像编码器训练/评估路径不调用该扩展。若以后使用 SAM2 的连通域后处理，需要单独补齐扩展。
- 数据构建依赖包括 GeoPandas、netCDF4、pyshp、SciPy；另装 ipykernel 供编辑器选择该 Python 环境。

## 数据范围

当前直接使用 F 盘交接包中的模型、v3 数据、加州 GEBCO 与断层源数据。v6 全球方案需要额外的全球 GEBCO 和 GEM 数据，当前 v5 环境验证不需要它们；全球数据构建和多日训练需作为后续实验安排。

## 重建环境

系统需先具备 `python3-venv python3-dev build-essential git curl rsync libgomp1`，然后在代码目录运行：

```bash
bash setup_wsl.sh
```

Python 科学计算依赖本次使用清华 PyPI 镜像，GPU 版 PyTorch 使用官方 cu126 源。

官方参考：[SAM2 安装要求](https://github.com/facebookresearch/sam2)、[PyTorch 版本与 CUDA 配对](https://pytorch.org/get-started/previous-versions/)。

## 本机验证结果

已完成本机验证（2026-09-09）：

- `pip check`：通过，无依赖冲突。
- 真实数据 CUDA 冒烟测试：通过。batch size 2，输入 2×7×256×256，输出 2×3×256×256；有限梯度和参数更新均已验证；峰值已分配显存约 1582 MiB。
- 主干权重、v5 权重、HDF5 数据集：WSL 副本与 Windows 源文件 SHA-256 完全一致。
- 训练/验证/测试 key 清单可读取、互不重复，所有 key 都存在于 HDF5。
- 原始加州断层 shapefile 和 GEBCO netCDF 均可读取。
- 现有 EMA 单元测试：7 项全部通过，包含 CUDA 检查。
- v5 全部 38 块测试集评估：AP 0.12044545，buffered-F1 0.37460837，最佳阈值 0.49。交接基准 AP 0.12044246、buffered-F1 0.37460368，两者非常接近。
- 使用 `train_wsl.sh --epochs 1 --resDir wsl_validation/train_one_epoch/` 完成一轮训练；80 个训练 batch，验证损失 0.70728396，正常退出并保存约 178 MiB 的 model.pth。此结果只验证训练流程，不代表完整训练后的模型质量。
- 本次没有启动默认 160 轮长训练。

依赖精确版本已保存在代码目录的 `requirements-wsl-lock.txt`，重装脚本优先使用该文件。
完整日志和 JSON 在 WSL 代码目录 `wsl_validation/`；小型验证产物另复制到 Windows 交接目录的 `wsl-setup/validation/`。

## F 盘运行方式（2026-09-09）

代码、数据、相对路径输出均位于 F 盘。Python 环境仍位于 WSL 的 /home/ovvvin/.venvs/earthquake-sam。原来的 ~/projects/Earthquake-SAM/SAM_code_extracted/SAM code/activate_wsl.sh 命令仍然可用，现在会转入 F 盘。使用 pwd 可以确认路径以 /mnt/f/ 开头。

从 Windows 文件系统读写大量小文件可能比 WSL 内部文件系统慢；本次优先满足 Codex 编辑与运行共用同一份代码。训练已启动后 Python 通常不会自动加载后续编辑，修改代码后需重新启动进程才能使用新代码。

F 盘验证：已确认当前目录及 torchUnet.__file__ 指向 /mnt/f/ 下的项目；真实数据 CUDA 前向、反向及参数更新通过。结果见代码目录 wsl_validation/f_drive_smoke.json。
