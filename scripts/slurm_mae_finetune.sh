#!/bin/bash
#SBATCH -J mae-finetune
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o logs/mae_finetune_%j.out
#SBATCH -e logs/mae_finetune_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

# module avail
# module load <本机 python/conda 模块> <本机 cuda 模块>
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate dafatt

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

ENCODER=/scratch/$USER/mae/pretrain/checkpoints/encoder.pt
CUBES=/scratch/$USER/crism_labeled
LABELS=/scratch/$USER/crism_labels
OUT=/scratch/$USER/mae/finetune
PY=python

$PY scripts/run_mae.py finetune \
  --encoder "$ENCODER" \
  --data "$CUBES" \
  --label "$LABELS" \
  --output "$OUT" \
  --num-classes 23 \
  --max-per-class 200 \
  --epochs 40 \
  --batch-size 32 \
  --device cuda:0 \
  --freeze-encoder
