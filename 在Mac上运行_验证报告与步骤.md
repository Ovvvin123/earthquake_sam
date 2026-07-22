# 在 Mac mini (M4, 24GB) 上运行 SAM 海底断层训练 — 验证报告 + 步骤

日期：2026-05-28

## 摘要

按你的要求，在你花时间长跑之前，我用**真实代码**对整条链路做了一次自检，**修掉了两个会导致中途失败的硬伤**，并把所有改动落到代码里。最容易"跑半天才发现挂"的数据环节已经实测通过；唯一无法在我这边实测的是 SAM 模型前向（因为这个环境没法装 Meta 的 `sam2` 包、内存也不够），所以我给了你一个**1 分钟的冒烟测试**，在 Mac 上先跑它，过了再开长跑。

---

## 一、已实测通过的部分（用真实 `Generator_new`，80 次随机抽样）

| 检查项 | 结果 |
|---|---|
| 数据能被真实 `Generator_new` 读取 | ✅ |
| 输入张量形状 7×256×256 / 标签 3×256×256 | ✅ 全部正确（匹配模型 `in_Channel=7, out_Channel=3`）|
| 训练时随机裁剪崩溃 | ✅ 已修复（80 次抽样 0 崩溃，修复前约 50% 崩溃）|
| `mole`(地形) 通道是否有效 | ✅ 已修复（std 0.78，修复前 ≈4e-6 等于全空）|
| 裁剪后仍包含断层的比例 | ✅ 52/80，平均断层像素 6%（不会再坍缩）|
| 验证模式 (training=False) | ✅ 正常 |

## 二、修复的两个硬伤（已写进代码）

1. **切片尺寸太小会崩溃**
   `Generator_new` 会从每张存图里**随机裁一个 256 窗口**，所以存图必须**大于 256**。原来切成正好 256 → 训练时约一半概率 `randint(0,0)` 直接报错。
   **修复**：数据准备改成存 **384×384** 切片（`prepare_earth_dataset.py` 里 `TILE=384`），训练再裁 256。

2. **火星归一化把地形通道清零**
   `tool.py` 里 `mole` 做了 `/1e8`（火星 MOLA 的尺度），套到米级海底地形上会被压成 ≈0，等于废掉一个主输入通道。
   **修复**：`tool.py` 改成**按每块标准差归一化**（`mole = (mole-mean)/std`），尺度无关，火星地球都适用。

## 三、另外顺手改好的（让它能在 Mac 上跑）

- `trainV10.py`：数据路径 `trainFile`/`dataFile` → 指向 `earth_dataset/ca_offshore_faults.h5`。
- `trainV10.py`：设备从写死的 `cuda` → **自动选 CUDA / Apple MPS / CPU**，Mac 上会自动用 MPS。

## 四、我这边无法实测、需你在 Mac 上确认的一步

**SAM 模型前向**：需要 Meta 的 `sam2` 包 + 较大内存，这个沙盒装不了。
→ 已为你写好 `smoke_test_mac.py`：它会真正地"建模型 → 喂一批真实数据 → 算损失 → 反向传播一步"。**过了就说明长跑没问题。**

---

## 五、在 Mac 上的操作步骤

### 1. 装依赖（建议用 conda/venv）
```bash
# PyTorch（带 MPS 的 Apple Silicon 版，官网默认 pip 即可）
pip install torch torchvision
pip install h5py scipy matplotlib pyshp pillow
# mpi4py 需要先装 MPI 库
brew install open-mpi && pip install mpi4py
# Meta 的 SAM-2
pip install "git+https://github.com/facebookresearch/sam2.git"
```

### 2. 放好文件
把 `earth_dataset/`（含 `ca_offshore_faults.h5` 和 `key.lst`）放到 `SAM code/` 目录下；
确认 `sam2_hiera_small.pt` 在 `SAM code/` 里（已有）。

### 3. 先跑冒烟测试（<1–2 分钟）
```bash
cd "SAM code"
PYTORCH_ENABLE_MPS_FALLBACK=1 python smoke_test_mac.py
```
看到 **`SMOKE TEST PASSED`** 再继续；如果报错，把报错发我，多半是依赖没装齐。

### 4. 正式训练（二分类、推荐参数）
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python trainV10.py \
  --job train --model mars_sam --modelType mars_sam \
  --dataset earth_dataset/key.lst \
  --resDir train_ca_offshore_v1/ \
  --mn 256 --batchsize 2 --stepPerMiniBatch 4 \
  --lr0 0.0001 --d0 0.02 --dw 0.08 --alpha 0.05 --dropout 0
```
- `--batchsize 2`：M4 上稳妥（别用原来的 24）。
- `--d0 0.02`：断层判定阈值（比火星的 0.05 小，更合理）。
- 训练产物在 `train_ca_offshore_v1/`：`model.pth`、`loss.txt`、`plot/`。

### 5. 怎么判断这次"成功了"（避免重蹈 11 月覆辙）
- **看 `loss.txt`**：应当**持续下降**，而不是像上次第 2 轮就卡在 0.0433。
- **看 `plot/` 里的预测图**：预测应当有**沿地形的纹理结构**，而不是一整片纯色。
- 我在 CPU 上用替身模型已经验证过：同样的数据 + 同样的不平衡处理，loss 会稳步下降、预测有结构（见 `poc_loss_curve.png`、`poc_predictions.png`）。

## 六、对 M4 的预期
- 内存：24GB 足够（小批量 + 256 分辨率）。
- 速度：MPS 比 4090 慢，但这个加州数据集很小，跑出第一版大概**数小时**量级。
- 若嫌慢：可先把 `prepare_earth_dataset.py` 的 `CROP`/`--mn` 降到 128 做快速试验。

## 七、相关文件
- `smoke_test_mac.py` — **先跑这个**
- `prepare_earth_dataset.py` — 数据准备（已改 384 切片）
- `SAM_code_extracted/SAM code/tool.py` — 已修复归一化
- `SAM_code_extracted/SAM code/trainV10.py` — 已修复路径 + 设备自适应
- `earth_dataset/` — 训练数据
- `poc_loss_curve.png` / `poc_predictions.png` — CPU 替身模型的"会学、不坍缩"证据
- `问题归纳与数据准备说明.md` — 更全面的问题归纳
