# 海底断层检测 (SAM-2 微调) — 方法学与论文级协议

作者管线衔接：Ruobing Guo, Yuan Jiang（火星版）→ 地球海底版
日期：2026-05-28

本文件描述一个**完整、可发表**的流程：把火星断层检测的 SAM-2 微调框架迁移到地球海底，
用 GEBCO 海底地形 + USGS 加州外海断层目录，做断层分割与类型识别。所有数据/划分/评估均按
发表标准构建并已在真实代码上自检通过；**训练跑分需在你的 Mac/GPU 上完成**（沙盒无 GPU、无法装 `sam2`）。

---

## 1. 任务定义（不简化）

- **输入**：GEBCO 2026 海底地形（15 角秒，约 450 m/格）及其衍生量。
- **输出（3 个标注通道，多标签，距离场，>0.35°→999）**：
  - `k0` = 到**任意断层**的距离（presence，主任务）
  - `k1` = 到**含倾滑分量断层**的距离（slip_sense 含 N 或 R；倾滑/斜滑）
  - `k2` = 到**其他地貌强线性特征**的距离（远离断层的强地形边缘——海山、海脊、峡谷等，作为**难负样本/判别**通道）
- **为什么不套火星的"正/逆"两分类**：加州外海断层 754 条含走滑分量、仅 315 条含倾滑分量、纯正断层仅 1 条。
  强行用"正断层/逆断层"在外海几乎无样本。故按**实际 slip_sense** 设计为 `k0`(全部)+`k1`(倾滑) 多标签，
  并把"其他地貌"显式建模为 `k2`，对应代码里已有的 `--isOther` 判别机制。

## 2. 输入表征（针对"海底无颜色/纹理"做了强化）

火星版报告指出 SAM 在"颜色变化不明显、与陨石坑等地貌混淆"时表现差。海底地形是单通道高程，
更缺颜色纹理，因此输入通道刻意强化**线性构造**：

- `CTX`：海底地形稳健拉伸到 0–255（灰度）。
- `image`(mole)：原始米级水深；加载时**按每块标准差归一化**（已修复火星的 `/1e8` 错误）。
- `TW`：**三方位角晕渲**（azimuth 315°/75°/195°，相差 120°），让任意走向的断层在某一光照下凸显。

（见 `dataset_channels_panel.png`）

## 3. 数据集构建（`prepare_earth_dataset.py`）

- 存图 **384×384**，训练时随机裁 **256**（火星代码就是"大图随机裁块"，存图必须>裁剪尺寸——这是一个实测到的崩溃点，已修）。
- 步长 64 px、重叠采样；正样本=含≥40 断层像素；负样本**优先难负样本**（高地形崎岖度但无断层）。
- `k0/k1/k2` 由断层线/强边缘经 `distance_transform` 得到距离场，>0.35°→999（与火星一致）。
- 每块写入经纬度、split、断层像素数、崎岖度等元数据；全局 `manifest.json` 记录 provenance。
- **本次产出**：227 块（train 89 / val 53 / test 85；正 138 / 负 89）。

## 4. 无泄漏的空间划分（发表硬要求）

相邻切片高度重叠，随机划分会让 val/test "见过"训练内容 → 指标虚高。本流程按**连续纬度块**划分，
并**丢弃跨边界切片**作为隔离带：

- **test**：纬度 < 35°N
- **train**：35°–38°N
- **val**：> 38°N

实测三个 split 的 key **零重叠**（见 `dataset_spatial_split.png`，35°/38° 处有空白隔离带）。
`trainV10.py` 已打补丁：训练时自动读取 `key_train.lst` / `key_val.lst`（而非原来的随机 90/10）。

## 5. 模型与训练

- 模型：`conv_sam`（冻结 SAM-2 hiera_small 主干，训练上层卷积/解码；`--modelType mars_sam`）。检查点用目录里的 `sam2_hiera_small.pt`。
- 设备：已改为自动 CUDA / **Apple MPS** / CPU。
- 损失：加权 Dice，近断层像素按 `exp(-d²/dw²)` 加权（缓解类别不平衡）；开 `--isOther` 时用 `k2` 增强判别。
- **推荐命令（Mac）**：
  ```bash
  cd "SAM code"
  PYTORCH_ENABLE_MPS_FALLBACK=1 python trainV10.py \
    --job train --model mars_sam --modelType mars_sam --isOther \
    --dataset earth_dataset/key.lst --resDir train_ca_offshore_v1/ \
    --mn 256 --batchsize 2 --stepPerMiniBatch 4 \
    --lr0 1e-4 --d0 0.02 --dw 0.08 --alpha 0.05 --dropout 0
  ```
- **先跑 `smoke_test_mac.py`**（<1–2 分钟）确认模型前向/反传通过，再开长跑。

## 6. 评估（`evaluate.py`，在空间 test 集上）

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python evaluate.py --resDir train_ca_offshore_v1/
```
- 确定性**中心裁剪**遍历整个 test 集（模型从未见过的纬度带）。
- 指标（对 k0、k1 各算）：
  - **AP**（PR 曲线下面积，免阈值）
  - 最优 F1 工作点的 precision / recall / F1 / **IoU**
  - **缓冲检测指标**（预测落在真值 ±`tol_px` 内即算命中）——细线状目标的公平评分方式
- 产出 `test_metrics.json` + `test_qualitative.png`。

## 7. 建议补充的实验（让论文更扎实）

1. **消融**：单方位 vs 三方位晕渲；有/无 `k2`(`--isOther`)；冻结主干 vs 解冻末层；阈值 d0(0.02/0.05)。
2. **基线对比**：同数据训练一个普通 U-Net（代码里有 `UNet`，`--modelType` 留空即用），对比 SAM-2 微调。
3. **判别力**：专门统计模型在 `k2`（其他地貌）区域的误报率，回应火星版"断层 vs 其他地貌混淆"的问题。
4. **跨区域泛化**：若后续接入其他海域（或全美 `Qfaults_US_Database.shp`），用本框架做 leave-one-region-out。

## 8. 诚实的局限（应写进论文）

- **标注覆盖**：外海断层仅沿岸窄带、最北约 40°N；深海无标注。"无标注≠无断层"，会产生假阴性、压低 recall；
  深海外推结果不可当真。
- **类型样本不均**：倾滑/纯正/纯逆样本很少，`k1` 指标方差大；走滑为主。
- **数据量**：单幅 GEBCO + 单一目录，样本量有限（靠随机裁剪+翻转/旋转增广扩充）。扩区域会显著增强可发表性。
- **`k2` 为启发式**：用地形梯度阈值近似"其他地貌"，非人工标注。

## 9. 文件清单

数据与脚本
- `prepare_earth_dataset.py` — 发表级数据构建（多方位晕渲 + 断层分型 + k2 难负样本 + 空间划分 + provenance）
- `earth_dataset/ca_offshore_faults.h5`、`key.lst`、`key_train/val/test.lst`、`manifest.json`
- `smoke_test_mac.py` — **长跑前先跑**：真实 SAM 前向+反传一步自检
- `evaluate.py` — 空间 test 集评估（AP / F1 / IoU / 缓冲指标）

已打补丁的代码（在 `SAM_code_extracted/SAM code/`）
- `tool.py` — 修复 mole 归一化（`/1e8`→按 std）
- `trainV10.py` — 数据路径、设备自适应（CUDA/MPS/CPU）、空间划分读取

图件
- `dataset_channels_panel.png` — 输入/标注各通道
- `dataset_spatial_split.png` — 无泄漏空间划分
- `faults_on_bathymetry.png` / `fault_catalog_sidebyside.png` — 断层目录与地形
- `poc_loss_curve.png` / `poc_predictions.png` — CPU 替身实验（"会学、不坍缩"的证据）

参考前置文档
- `问题归纳与数据准备说明.md`、`在Mac上运行_验证报告与步骤.md`
