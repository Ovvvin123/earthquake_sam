#!/usr/bin/env bash
set -euo pipefail
# Separate defaults/output; train_wsl.sh and the PU implementation are unchanged.
bash "$(dirname "$0")/train_wsl.sh" \
  --resDir train_v5_unimatch/ \
  --pu 0 --emaTeacher --emaDecay 0.98 --epochs 200 \
  --onlineCorrection --fixedVal --unimatch \
  --emaPreview --previewEvery 5 "$@"
