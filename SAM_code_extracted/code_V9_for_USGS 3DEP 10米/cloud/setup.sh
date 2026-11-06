#!/usr/bin/env bash
set -euo pipefail
V9_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
V9_BOOTSTRAP_PYTHON="${V9_BOOTSTRAP_PYTHON:-python3.10}"
V9_VENV="${V9_VENV:-$V9_ROOT/.venv-cloud}"
V9_DRY_RUN=0
case "${1:-}" in
  '') ;;
  --dry-run) V9_DRY_RUN=1 ;;
  -h|--help) printf 'Usage: bash cloud/setup.sh [--dry-run]\nVariables: V9_BOOTSTRAP_PYTHON, V9_VENV (must be a NEW directory).\n'; exit 0 ;;
  *) printf 'Unexpected argument: %s\n' "$1" >&2; exit 2 ;;
esac
[[ $# -le 1 ]] || { printf 'Too many arguments\n' >&2; exit 2; }
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { printf 'This environment lock targets Linux x86_64.\n' >&2; exit 2; }
"$V9_BOOTSTRAP_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3,10), "Python 3.10 is required for this validated environment"'
[[ ! -e "$V9_VENV" ]] || { printf 'Refusing to modify existing environment: %s\nChoose a new V9_VENV.\n' "$V9_VENV" >&2; exit 2; }
"$V9_BOOTSTRAP_PYTHON" "$V9_ROOT/cloud/verify_vendor.py"

v9_run() {
  printf '%q ' "$@"; printf '\n'
  if [[ "$V9_DRY_RUN" == 0 ]]; then "$@"; fi
}
v9_run "$V9_BOOTSTRAP_PYTHON" -m venv "$V9_VENV"
v9_run "$V9_VENV/bin/python" -m pip install 'setuptools>=61' wheel
v9_run "$V9_VENV/bin/python" -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
v9_run "$V9_VENV/bin/python" -m pip install -r "$V9_ROOT/cloud/requirements.txt"
v9_run env SAM2_BUILD_CUDA=0 "$V9_VENV/bin/python" -m pip install --no-build-isolation --no-deps "$V9_ROOT/cloud/vendor/sam2-runtime.tar.gz"
v9_run "$V9_VENV/bin/python" -m pip check
printf 'Environment preparation only. No training was started.\nNext: bash cloud/preflight.sh\n'
