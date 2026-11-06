#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 0 ]]; then
  printf 'Usage: bash cloud/preflight.sh (no arguments; cannot start training)\n' >&2
  exit 2
fi
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
v9_require_python
exec "$V9_PYTHON" "$V9_ROOT/cloud/manage.py" check
