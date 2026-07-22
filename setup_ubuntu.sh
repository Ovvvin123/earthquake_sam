#!/usr/bin/env bash
# Create/update the local Ubuntu environment. PyTorch is installed separately
# because its wheel must match the local NVIDIA/CUDA setup.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${SAM_ENV_NAME:-sam_env}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found. Install Miniconda/Anaconda first."
  exit 1
fi

eval "$(conda shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" python=3.10
fi
conda activate "$ENV_NAME"

python -m pip install --upgrade pip wheel
if ! python -c 'import torch, torchvision' >/dev/null 2>&1; then
  echo "Installing PyTorch. For a custom CUDA wheel, cancel and use the command from pytorch.org."
  python -m pip install torch==2.5.1 torchvision==0.20.1
fi
python -m pip install -r "$PROJECT_ROOT/requirements.txt"

python - <<'PY'
import torch
print('python/torch environment ready')
print('torch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
PY

echo "Run next: conda activate $ENV_NAME && python $PROJECT_ROOT/smoke_test.py"
