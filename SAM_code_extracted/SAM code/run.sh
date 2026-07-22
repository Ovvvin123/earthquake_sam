#!/usr/bin/env bash
# Ubuntu/CUDA default training recipe.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${SAM_PYTHON:-python}"
BATCH_SIZE="${SAM_BATCH_SIZE:-2}"
EPOCHS="${SAM_EPOCHS:-160}"

exec "$PYTHON_BIN" "$PROJECT_ROOT/train.py" \
  --job train \
  --modelType mars_sam \
  --isOther \
  --dataset "$SCRIPT_DIR/earth_dataset_v3/key_train.lst" \
  --resDir "$PROJECT_ROOT/outputs/train_local" \
  --mn 256 \
  --batchsize "$BATCH_SIZE" \
  --stepPerMiniBatch 4 \
  --epochs "$EPOCHS" \
  --lr0 1e-4 \
  --d0 0.02 \
  --dw 0.08 \
  --alpha 0.05 \
  --dropout 0 \
  --lossFunc FocalTversky
