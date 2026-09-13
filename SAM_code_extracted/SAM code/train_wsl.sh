#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/activate_wsl.sh"
python trainV10.py --job train --modelType mars_sam \
  --dataset earth_dataset_v3/key_train.lst --resDir train_v5_wsl/ \
  --mn 256 --batchsize 2 --stepPerMiniBatch 4 \
  --lr0 1e-4 --d0 0.02 --dw 0.08 --alpha 0.1 --dropout 0 \
  --lossFunc FocalTversky --k0Only --pu 0.4 "$@"
