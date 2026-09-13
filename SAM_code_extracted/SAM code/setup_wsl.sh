#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 -m venv "$HOME/.venvs/earthquake-sam"
source "$HOME/.venvs/earthquake-sam/bin/activate"
python -m pip install --upgrade pip wheel setuptools
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
REQUIREMENTS="$SCRIPT_DIR/requirements-wsl.txt"
if [[ -f "$SCRIPT_DIR/requirements-wsl-lock.txt" ]]; then
  REQUIREMENTS="$SCRIPT_DIR/requirements-wsl-lock.txt"
fi
python -m pip install -r "$REQUIREMENTS" --index-url https://pypi.tuna.tsinghua.edu.cn/simple
python3 "$SCRIPT_DIR/../../wsl-setup/fetch_sam2.py"
# This project does not call the optional connected-components CUDA extension.
SAM2_BUILD_CUDA=0 python -m pip install --no-build-isolation --no-deps "$HOME/src/sam2-runtime"
python -m pip check
