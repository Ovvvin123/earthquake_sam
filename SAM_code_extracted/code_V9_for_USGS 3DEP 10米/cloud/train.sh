#!/usr/bin/env bash
set -euo pipefail
# Deliberately require the user's explicit launch flag even when run by mistake.
if [[ $# -ne 1 || "${1:-}" != --execute ]]; then
  printf 'No training started. To explicitly launch: bash cloud/train.sh --execute\nPreflight only: bash cloud/preflight.sh\n' >&2
  exit 2
fi
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
v9_require_python
exec "$V9_PYTHON" "$V9_ROOT/cloud/manage.py" train --execute
