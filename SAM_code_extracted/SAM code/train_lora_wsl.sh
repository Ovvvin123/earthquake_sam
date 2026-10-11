#!/usr/bin/env bash
set -euo pipefail
# Supervised LoRA starting point; append teacher flags or override adaptation for ablations.
bash "$(dirname "$0")/train_wsl.sh" \
  --resDir train_v5_lora_r8/ --samAdaptation lora \
  --loraRank 8 --loraAlpha 16 --loraDropout 0 --loraLr 5e-5 \
  --d0 .01 --pu 0 --fixedVal "$@"
