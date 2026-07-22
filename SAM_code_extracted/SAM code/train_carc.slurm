#!/bin/bash
#SBATCH --job-name=sam_fault
#SBATCH --partition=gpu
#SBATCH --account=xianruic_1452
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# ===== environment =====
module purge
module load conda/25.11.0 gcc
eval "$(conda shell.bash hook)"
conda activate sam_env

cd "$SLURM_SUBMIT_DIR"
echo "Job $SLURM_JOB_ID on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ===== training (the "stronger recipe") =====
# trainV10.py is already patched: 160 epochs, auto CUDA/MPS/CPU, spatial split,
# num_workers=0. On an A100 you can raise --batchsize well above the Mac's 2.
python trainV10.py \
  --job train --model mars_sam --modelType mars_sam --isOther \
  --dataset $DATASET/key.lst --resDir $RESDIR \
  --mn 256 --batchsize 8 --stepPerMiniBatch 2 \
  --lr0 1e-4 --d0 0.03 --dw 0.14 --alpha 0.02 --dropout 0

echo "training done; evaluating on the held-out test split"
python evaluate.py --resDir $RESDIR --dataDir $DATASET --d0 0.02
