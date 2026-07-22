#!/bin/bash
# ==================================================================
# setup_carc.sh — one-time environment setup on CARC (USC Discovery).
# Run on a LOGIN node:   bash setup_carc.sh
# ==================================================================
set -e
module purge
module load conda/25.11.0 gcc    # CARC: conda + a compiler (sam2 builds C++)
eval "$(conda shell.bash hook)"
# create an isolated env
conda create -y -n sam_env python=3.11
source activate sam_env
pip install --upgrade pip wheel
# PyTorch with CUDA (CARC GPUs). cu121 wheels work on most CARC GPU nodes.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy h5py scipy matplotlib pyshp pillow geopandas netCDF4 timm numba hydra-core iopath
pip install "git+https://github.com/facebookresearch/sam2.git"
echo "==> done. Activate later with:  module load conda gcc && source activate sam_env"
