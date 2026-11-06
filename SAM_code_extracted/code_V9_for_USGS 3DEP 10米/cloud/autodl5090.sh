#!/usr/bin/env bash
set -euo pipefail
V9_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export V9_ENV_PROFILE=autodl5090
export V9_VENV="${V9_VENV:-$V9_ROOT/.venv-autodl5090}"
export V9_PYTHON="$V9_VENV/bin/python"
export MPLBACKEND=Agg
export PYTHONNOUSERSITE=1
export PIP_REQUIRE_VIRTUALENV=true
mode="${1:-}"
[[ $# -eq 1 ]] || { printf 'Usage: bash cloud/autodl5090.sh setup|check\n' >&2; exit 2; }
case "$mode" in
  setup)
    [[ ! -e "$V9_VENV" ]] || { printf 'Refusing to overwrite existing venv: %s\n' "$V9_VENV" >&2; exit 2; }
    base_python="${V9_BOOTSTRAP_PYTHON:-python}"
    "$base_python" "$V9_ROOT/cloud/environment_autodl5090.py" base
    "$base_python" "$V9_ROOT/cloud/verify_vendor.py"
    "$base_python" -m venv --system-site-packages "$V9_VENV"
    "$V9_PYTHON" "$V9_ROOT/cloud/environment_autodl5090.py" constraints > "$V9_VENV/gpu-constraints.txt"
    "$V9_PYTHON" -m pip install -c "$V9_VENV/gpu-constraints.txt" 'setuptools>=61' wheel
    "$V9_PYTHON" -m pip install -c "$V9_VENV/gpu-constraints.txt" -r "$V9_ROOT/cloud/requirements-autodl5090.txt"
    SAM2_BUILD_CUDA=0 "$V9_PYTHON" -m pip install --force-reinstall --no-build-isolation --no-deps "$V9_ROOT/cloud/vendor/sam2-runtime.tar.gz"
    "$V9_PYTHON" "$V9_ROOT/cloud/environment_autodl5090.py" record
    printf 'Setup complete. No training started. Next: bash cloud/autodl5090.sh check\n'
    ;;
  check)
    exec bash "$V9_ROOT/cloud/preflight.sh"
    ;;
  *) printf 'Usage: bash cloud/autodl5090.sh setup|check (this wrapper cannot start training)\n' >&2; exit 2 ;;
esac
