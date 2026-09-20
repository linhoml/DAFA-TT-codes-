#!/bin/bash
# =============================================================================
# 超算互联网 (SCNet) — MAE 微调（预训练出 encoder.pt 之后再交这个）
#
# 队列 / --gres / module load 必须和预训练脚本一致。
# =============================================================================
#SBATCH -J mae-finetune
#SBATCH -p kshcnormal          # ← 与预训练同一队列
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1           # ← NVIDIA: gpu:1；DCU: dcu:1
#SBATCH -t 24:00:00
#SBATCH -o logs/mae_finetune_%j.out
#SBATCH -e logs/mae_finetune_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

echo "======== 微调开始 $(date) job=${SLURM_JOB_ID:-local} ========"

CONDA_SH="${HOME}/miniconda3/etc/profile.d/conda.sh"
# shellcheck source=/dev/null
source "$CONDA_SH"
conda activate dafatt

module purge
# module load nvidia/cuda/11.4
# module load compiler/devtoolset/7.3.1 mpi/hpcx/2.11.0/gcc-7.3.1
# module load compiler/dtk/23.10

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

ENCODER=/public/home/${USER}/mae/pretrain/checkpoints/encoder.pt
CUBES=/public/home/${USER}/crism_labeled
LABELS=/public/home/${USER}/crism_labels
OUT=/public/home/${USER}/mae/finetune

if [[ ! -f "$ENCODER" ]]; then
  echo "找不到预训练编码器: $ENCODER" >&2
  echo "等预训练跑完，或改 ENCODER 指向 encoder.pt / encoder_epN.pt" >&2
  exit 1
fi
if [[ ! -e "$CUBES" || ! -e "$LABELS" ]]; then
  echo "立方体或标签路径不存在:" >&2
  echo "  CUBES=$CUBES" >&2
  echo "  LABELS=$LABELS" >&2
  exit 1
fi
mkdir -p "$OUT"

python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"

python scripts/run_mae.py finetune \
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

echo "======== 微调结束 $(date) ========"
echo "最佳模型应在: $OUT 下的 model_best.pth"
