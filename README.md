# Earthquake-SAM（Ubuntu 本地版）

本项目使用 SAM-2 和地形/断层数据进行断层分割。代码最初来自 macOS 和 CARC 环境，当前默认入口已整理为 Ubuntu + NVIDIA CUDA，同时保留 CPU 和 Apple MPS 兼容。

## 1. 当前可用资源

- 训练数据：`SAM_code_extracted/SAM code/earth_dataset_v3/faults.h5`
- 空间无泄漏训练/验证/测试清单：同目录下的 `key_train.lst`、`key_val.lst`、`key_test.lst`
- SAM-2 权重：`SAM_code_extracted/SAM code/sam2_hiera_small.pt`
- 历史训练结果：`SAM_code_extracted/SAM code/train_v5_local/`

`earth_dataset_v4` 只有切分清单，没有 HDF5 数据，当前不要作为默认数据集。

## 2. 环境

已有环境可直接检查：

```bash
conda activate sam_env
which python
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

重新创建或补全环境：

```bash
bash setup_ubuntu.sh
conda activate sam_env
```

如果需要指定特殊 CUDA 版本，请先按 PyTorch 官网命令安装 `torch/torchvision`，再运行：

```bash
python -m pip install -r requirements.txt
```

## 3. 首次验证

从项目根目录运行：

```bash
conda activate sam_env
python smoke_test.py
```

正常情况下 Ubuntu NVIDIA 机器会显示 `device = cuda:0`，最后显示：

```text
=== SMOKE TEST PASSED ===
```

不要使用 `/usr/bin/python3`，它会绕过 Conda 环境。

## 4. 训练

推荐入口：

```bash
conda activate sam_env
bash run_train.sh
```

默认结果写入 `outputs/train_local/`。显存较小时可以降低 batch size：

```bash
SAM_BATCH_SIZE=1 bash run_train.sh
```

只做短流程检查：

```bash
SAM_EPOCHS=1 SAM_BATCH_SIZE=1 bash run_train.sh
```

也可以直接传入参数：

```bash
python train.py \
  --job train \
  --modelType mars_sam \
  --dataset "SAM_code_extracted/SAM code/earth_dataset_v3/key_train.lst" \
  --resDir outputs/my_run \
  --batchsize 2 \
  --epochs 160
```

## 5. 评估

评估新训练结果：

```bash
python evaluate.py \
  --resDir outputs/train_local \
  --dataDir "SAM_code_extracted/SAM code/earth_dataset_v3"
```

评估历史模型：

```bash
python evaluate.py \
  --resDir "SAM_code_extracted/SAM code/train_v5_local" \
  --dataDir "SAM_code_extracted/SAM code/earth_dataset_v3"
```

## 6. 重新构建数据集

现有 `earth_dataset_v3` 已经可以训练，通常不需要重建。需要重建时：

```bash
python prepare_earth_dataset.py --out "SAM_code_extracted/SAM code/earth_dataset_local"
```

默认读取本项目已有的 GEBCO netCDF 和 `Qfaults_GIS/SHP/ca_offshore.shp`，输出使用统一文件名 `faults.h5`。

## 7. 路径覆盖

本 README 中的推荐训练、评估和数据构建流程不再依赖 `/Users/...`、`/Volumes/...`、`/H5/...` 等原作者路径。移动大文件后可设置：

```bash
export SAM_DATA_DIR=/path/to/dataset
export SAM_DATA_H5=/path/to/dataset/faults.h5
export SAM2_CHECKPOINT=/path/to/sam2_hiera_small.pt
export SAM_OUTPUT_DIR=/path/to/outputs
export GEBCO_FILE=/path/to/gebco.nc
export QFAULTS_FILE=/path/to/ca_offshore.shp
```

## 8. 代码布局

```text
train.py                       训练入口
evaluate.py                    评估入口
smoke_test.py                  完整冒烟测试入口
prepare_earth_dataset.py       数据构建入口
setup_ubuntu.sh                Ubuntu 环境配置
run_train.sh                   默认训练命令
SAM_code_extracted/SAM code/   原始核心代码（当前唯一实现位置）
```

`tool.py`、`torchUnet.py` 和 `config.py` 是项目本地模块，不是需要从 pip 安装的库。

`trainV10.py` 和 `tool.py` 底部仍保留了原作者的 Mars/全球数据分析分支，其中部分功能需要原作者未交付的外部 HDF5 数据。它们不在当前 Ubuntu 训练/评估流程中执行，因此保留作为历史研究代码。

## 9. Git/GitHub

`.gitignore` 已排除 HDF5、NetCDF、模型权重、缓存和训练输出。首次提交前仍应使用：

```bash
git status
git add -n .
```

确认没有数据集和模型权重后再正式提交。

大文件的预期位置和 SHA-256 校验值见 `ARTIFACTS.md`。
