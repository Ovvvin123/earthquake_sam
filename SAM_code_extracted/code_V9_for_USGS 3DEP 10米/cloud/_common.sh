#!/usr/bin/env bash
# Source this file; it never installs packages or starts a trainer.
set -euo pipefail
V9_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
V9_PYTHON="${V9_PYTHON:-${V9_VENV:-$V9_ROOT/.venv-cloud}/bin/python}"
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
cd -- "$V9_ROOT"

v9_require_python() {
  if [[ ! -x "$V9_PYTHON" ]]; then
    printf 'Python not found: %s\nRun cloud/setup.sh or set V9_PYTHON to an existing compatible interpreter.\n' "$V9_PYTHON" >&2
    exit 2
  fi
}
