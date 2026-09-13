# Earthquake-SAM：WSL 交接版

本分支同步 Windows 本地交接目录中的源码、配置、数据划分清单和说明文档，包含 WSL 环境、EMA teacher、在线临时补标、预览与训练数值修复相关代码。基于仓库原有提交保留历史；`main` 分支的 Ubuntu 适配版本仍可单独查看。

## 从这里开始

- [WSL 环境与运行说明](WSL环境与运行说明.md)：环境安装、激活、验证和训练方法。
- [项目交接说明](项目交接说明_给学生.md)：研究背景、数据与历史实验说明。
- [EMA 使用说明](SAM_code_extracted/SAM%20code/AIO2_EMA_使用说明.md)。
- [在线临时补标](SAM_code_extracted/SAM%20code/AIO2_在线临时补标.md)。
- [预览与诊断](SAM_code_extracted/SAM%20code/AIO2_预览与诊断.md)。

核心代码和 WSL 脚本位于 `SAM_code_extracted/SAM code/`。本分支保留本地交接版的文件布局和源码，不包含只存在于 Ubuntu main 分支的根目录训练包装脚本；运行时以本分支 WSL 说明为准。

## Git 同步范围

Git 保存源码、测试、依赖清单、说明文档、数据划分清单及已有的小型参考资料。模型权重（`.pt`、`.pth`）、数据集（`.h5`）、地形数据（`.nc`）、训练产物、日志、缓存及本地备份不上传，仍保留在本地。

克隆代码后需要自行放置数据与权重，至少包括：

- `SAM_code_extracted/SAM code/sam2_hiera_small.pt`
- `SAM_code_extracted/SAM code/earth_dataset_v3/faults.h5`
- 评估历史 v5 时需要 `SAM_code_extracted/SAM code/train_v5_local/model.pth`

数据重建还需要 GEBCO 原始地形。说明文档中的本机绝对路径需按自己的环境调整。本次同步只检查 Git 文件范围和上传结果，没有重新运行训练或验证历史实验指标。
