V9 云端 Linux 部署（当前数据是30米 Earth_dataset_v7，1024×1024）

本目录全部为新增文件；原SAM code、V9核心代码和本地8GB配置未修改。
训练算法沿用已经验证的V9：BF16 + FlashAttention + poolFirst + decoder重计算，
LoRA rank8、batch1/累积8、Teacher250轮启用/350轮最大权重、总计500轮。

1. 上传内容
把整个V9代码目录上传到云端Linux，保留其目录结构，包括：
  experiments/california_v9_30m_1024_cloud.json
  experiments/common_bceft_lora_ema_unimatch.json
  cloud/下全部文件（含vendor源码包和校验清单）
  sam2_hiera_small.pt
另外上传现有数据目录中的：
  faults.h5、key_train.lst、key_val.lst、key_test.lst、manifest.json
代码副本不含HDF5数据。无需上传原始DEM、GeoTIFF大图或逐图图库。
不要上传或直接搬用Windows/WSL的虚拟环境目录。

2. 选择存储目录
必须先确认平台实际的持久盘挂载位置、停机/释放实例后是否保留。
下例的 /persistent 是占位示例，不是对任何平台的挂载点保证：
  cd /persistent/earthquake-sam-v9
  export V9_DATA_DIR=/persistent/data/Earth_dataset_v7
  export V9_RUNS_DIR=/persistent/runs
  export V9_CHECKS_DIR=/persistent/checks
  export V9_RUN_NAME=v9_cloud_r01
  export V9_BACKUP_DIR=/persistent/backups
所有变量支持路径中的空格，shell中请加引号。也可以不设置变量：默认使用项目下
data/Earth_dataset_v7、runs和cloud_checks，额外backupDir默认关闭。
V9_BACKUP_DIR设为非空后，备份位于其下的V9_RUN_NAME子目录。
跨目录复制不是跨机器备份，同一云盘上的两个目录无法防止整块盘被释放。
不要使用 /mnt/f 来指代本地F盘；云端不存在这种自动映射。

3. 准备环境
要求Linux x86_64、Python3.10及可用的NVIDIA GPU驱动。
在项目根目录执行：
  export V9_BOOTSTRAP_PYTHON=python3.10
  export V9_VENV=/persistent/venvs/earthquake-v9
  bash cloud/setup.sh --dry-run
  bash cloud/setup.sh
setup只创建新的venv，不使用sudo、不修改已有环境、不启动训练。
目标venv若已存在则拒绝覆盖；失败后选择一个新目录重新安装。
如果python3.10或venv组件缺失，需按所选云镜像安装/选择Python3.10环境。
安装需要联网下载PyTorch CUDA12.6轮子和锁定的Python依赖。
已验证的torch/torchvision分别为2.7.1+cu126和0.22.1+cu126。
GPU必须实际支持BF16和当前PyTorch的Flash内核，预检会验证。
其他架构或需要更新CUDA轮子的GPU应另行准备/验证环境，不自动升级核心依赖。

SAM2从vendor/sam2-runtime.tar.gz安装；它来自已验证本地源码，
source commit 2b90b9f5ceec907a1c18123530e92e794ad901a4，包含许可证及38个
运行时Python/YAML文件的哈希。安装前校验压缩包；预检核对安装后的源码。
SAM2_BUILD_CUDA=0跳过其连通分量后处理扩展；当前训练只使用SAM图像编码器，
伪标签连通分量使用现有SciPy逻辑。这不会关闭PyTorch自带的CUDA/FlashAttention。
这里不安装制作DEM数据所需的完整GIS环境。

4. 只预检（不会训练）
  bash cloud/preflight.sh
如果使用一个已经准备好的兼容环境，可设置：
  export V9_PYTHON=/path/to/venv/bin/python
V9_PYTHON优先于V9_VENV；setup.sh使用V9_BOOTSTRAP_PYTHON。
预检核对环境、SAM2源码、模型/数据SHA-256、三个split列表、目录可写性，
执行真实Flash内核和一次1024推理，随后检查全部训练/验证样本。
只会创建检查报告和输出父目录，不会创建正式run、反向传播或更新参数。
报告在V9_CHECKS_DIR中按UTC时间和随机后缀分开保存，可重复预检。
资产清单固定对应当前V7数据，换数据时应重新准备配置与清单，不应绕过校验。
预检的推理通过不等于云端完整反向/长训稳定性保证；实际开训前建议另做受限显存检查。

5. 正式训练入口（本次没有执行）
只有显式执行下面命令才会启动500轮训练：
  bash cloud/train.sh --execute
该命令会重新完成预检，任何检查失败都会停止，不会自动启动训练。
直接运行train.sh而不加参数会退出，且不会调用训练程序。
训练在前台运行；远程使用时可在平台提供的持久终端或tmux会话中执行。
console.log、模型、完整检查点和训练预览均写入V9_RUNS_DIR/V9_RUN_NAME。
目录已存在会拒绝覆盖，请用新的V9_RUN_NAME。该入口只创建新实验，不自动续训。
断点续训需单独使用完整ema_checkpoint.pth；数据路径变化会触发现有严格校验，
不要把model.pth当作完整断点，不要删掉校验来强行恢复。

6. 备份与下载
backupDir关闭时，每轮保存、最佳模型及每20轮阶段检查点仍然正常写入run目录。
启用V9_BACKUP_DIR后沿用每20轮和249/350/500轮的完整检查点复制、校验、发布机制。
以后从云端下载到Windows，请用平台下载或SFTP，优先选择已完成的epoch备份目录，
对照backup_manifest.json核验；不要拷贝正在写入的检查点。
自动回传Windows/对象存储未包含在这些脚本中。
实例释放前，应确认持久盘保留策略并导出需要的权重、配置和结果。

7. 代码入口
_common.sh：统一项目位置、Python和无界面环境。
manage.py：生成独立的resolved_cloud.json，不修改任何源配置。
requirements.txt / environment_baseline.json：已验证环境的传递依赖锁定。
assets_manifest.json：当前数据、split列表、manifest和基础权重哈希。
verify_vendor.py：SAM2打包校验。
build_assets.py：维护者生成清单的工具，云端正常安装/训练无需运行。
test_cloud.py：包装脚本、路径隔离和不启动训练等测试。

官方安装依据：
https://pytorch.org/get-started/previous-versions/
https://github.com/facebookresearch/sam2/blob/main/INSTALL.md
