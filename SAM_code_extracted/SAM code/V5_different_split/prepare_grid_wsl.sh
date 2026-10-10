#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/activate_wsl.sh"
if (( $# )); then
  echo "For custom paths use the three Python commands separately." >&2
  exit 2
fi
python prepare_dataset_gem.py
python make_random_split.py
python audit_grid_dataset.py
