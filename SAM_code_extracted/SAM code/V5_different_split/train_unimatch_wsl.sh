#!/usr/bin/env bash
set -euo pipefail
bash "$(dirname "$0")/train_wsl.sh" \
  --resDir runs/teacher/ --emaTeacher --emaDecay 0.98 \
  --onlineCorrection --fixedVal --unimatch --emaPreview \
  --onlineStartStep 800 --onlineHoldStep 1000 --onlineRampStep 1600 \
  --unimatchStartStep 800 --unimatchRampStep 1600 "$@"
