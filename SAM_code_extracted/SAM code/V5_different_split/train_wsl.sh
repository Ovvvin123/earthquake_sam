#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/activate_wsl.sh"
python trainV10.py --job train --modelType mars_sam \
  --dataset datasets/ca_stride384/key_train.lst --resDir runs/supervised/ \
  --mn 256 --batchsize 2 --stepPerMiniBatch 4 \
  --lr0 1e-4 --d0 0.02 --dw 0.08 --alpha 0.1 --dropout 0 \
  --lossFunc FocalTversky --k0Only --pu 0 --maxSteps 3200 --evalEverySteps 100 "$@"
