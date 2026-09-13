#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/activate_wsl.sh"
mkdir -p wsl_validation/v5
if [[ ! -e wsl_validation/v5/model.pth ]]; then
  ln -s ../../train_v5_local/model.pth wsl_validation/v5/model.pth
fi
python evaluate.py --resDir wsl_validation/v5/ --dataDir earth_dataset_v3 --d0 0.02 "$@"
