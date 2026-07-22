#!/bin/bash
# ==================================================================
# setup_mac.sh — one-time environment setup on the M4 Mac mini.
# Run it once from inside the "SAM code" folder:
#     bash setup_mac.sh
# ==================================================================
set -e
cd "$(dirname "$0")"
echo "==> Working in: $(pwd)"

# 1) Xcode command-line tools (provides git + a compiler, needed by sam2)
if ! xcode-select -p >/dev/null 2>&1; then
  echo "==> Installing Xcode command-line tools (a popup will appear; click Install, then re-run this script)"
  xcode-select --install || true
  echo "    >>> Finish the popup install, then run:  bash setup_mac.sh   again."
  exit 0
fi

# 2) Isolated Python environment (keeps your system clean)
echo "==> Creating Python virtual environment (.venv)"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel

# 3) Core packages (Apple-Silicon / MPS wheels)
echo "==> Installing PyTorch + scientific packages"
pip install torch torchvision
pip install numpy h5py scipy matplotlib pyshp pillow hydra-core iopath timm numba

# 4) Meta's Segment-Anything-2
echo "==> Installing SAM-2 (from GitHub)"
pip install "git+https://github.com/facebookresearch/sam2.git"

echo ""
echo "============================================================"
echo " Setup complete."
echo " Next, every time you open a new Terminal, first run:"
echo "     cd \"$(pwd)\""
echo "     source .venv/bin/activate"
echo " Then run the smoke test:"
echo "     PYTORCH_ENABLE_MPS_FALLBACK=1 python smoke_test_mac.py"
echo "============================================================"
