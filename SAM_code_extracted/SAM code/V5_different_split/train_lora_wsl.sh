#!/usr/bin/env bash
set -euo pipefail
bash "$(dirname "$0")/train_wsl.sh" \
  --resDir runs/lora_r8/ --samAdaptation lora \
  --loraRank 8 --loraAlpha 16 --loraDropout 0 --loraLr 5e-5 "$@"
