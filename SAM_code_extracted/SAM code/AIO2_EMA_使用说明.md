# AIO2 第一步：EMA 教师

本次仅增加 EMA 教师，不启用 ACT、在线补标或一致性损失，不修改 PU 参数、原运行文档或损失函数。默认不开启 EMA，旧命令继续按原流程运行。

在现有训练命令末尾添加：

```text
--emaTeacher --emaDecay 0.99
```

PU 仍由你在启动命令中自行指定，例如 `--pu 0`。建议为 EMA 实验自行指定新的 `--resDir`。0.99 是可调的起始值，按实际 optimizer 更新计数，不是经过本项目验证的最优值。

## 行为

- 教师在训练开始时完整复制学生；教师仅 eval/no_grad，不参与优化器。
- 每次实际 `opt.step()` 后更新一次 EMA；原有梯度累积时机保持原样。
- 可训练参数和浮点 buffers 做 EMA；冻结参数、整数 buffers 直接复制。
- 去重处理 conv_sam 中重复注册的模块，避免同一参数多次 EMA。
- 验证时教师与学生使用同一批输入和原始标签，教师的 FocalTversky 验证损失使用 pu=0。
- 当前验证集仍沿用原脚本的随机增强，只用于本阶段观察；尚不是 ACT 所需的确定性训练监测集。
- 教师整份模型增加显存占用，验证也增加一次前向。学生训练目标保持原样。

## 输出

在你指定的结果目录中：

- `model.pth`：原有学生权重，格式保持不变。
- `teacher/model.pth`：教师权重，可直接供现有 evaluate.py 读取。
- `ema_loss.txt`：每行依次为 epoch（从 0 开始）、累计 EMA 更新次数、教师验证 loss。
- `ema_config.json`：本次运行参数。
- `ema_checkpoint.pth`：最近完成一轮的学生、教师、EMA 系数/计数、优化器、未提交的累积梯度和随机数状态，用于继续训练。这一步尚未保存 ACT 回退所需的一系列历史检查点。

教师评估使用原 evaluate.py，把原来的 `--resDir YOUR_RUN/` 改为 `--resDir YOUR_RUN/teacher/`，其余数据和模型参数保持一致。评估输出也写入 teacher 子目录。断层类型 k1 若使用 k0Only 没有训练，不应据此解读类型表现。

## 继续训练

在原训练命令中保留相同训练参数，并追加：

```text
--emaTeacher --emaDecay 0.99 --emaResume YOUR_RUN/ema_checkpoint.pth
```

`--epochs` 表示总目标轮数，例如完成 20 轮后设置为 40，将继续第 21–40 轮。恢复时允许更改 epochs、resDir 和 emaResume；其余运行参数必须一致。使用保存的 EMA 检查点，不能把历史学生 model.pth 当作完整续训检查点。恢复发生在整轮边界；跨硬件不保证逐位复现。

## 验证

在 SAM code 目录运行：

```text
python -m unittest test_aio2_ema -v
```

测试无需 sam2：涵盖教师隔离、别名去重、buffer 策略、实际 trainV10.py 训练分支接入、EMA 关闭/开启时学生结果一致、包含待累积梯度和 RNG 的续训等价性，以及 CUDA 上的预测和检查点恢复（无 CUDA 时跳过）。

训练分支测试使用小模型和合成数据，不能替代完整 SAM-2 模型与真实数据的运行验证。
