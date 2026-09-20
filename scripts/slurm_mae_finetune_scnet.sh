#!/bin/bash
# =============================================================================
# 超算互联网 (SCNet) — MAE 微调
#
# 当前按 linhonglei 的路径填好：
#   编码器目录  /public/home/linhonglei/ChaoSuan
#   立方体      /public/home/linhonglei/train_data
#   标签        /public/home/linhonglei/label_dat_remove_none_global
#
# 还要改的只有队列：#SBATCH -p、#SBATCH --gres、以及下面的 module load
# （与你预训练成功时用的那几行保持一致）。
#
#   cd 到代码根目录
#   mkdir -p logs
#   sbatch scripts/slurm_mae_finetune_scnet.sh
#   squeue -u "$USER"
#   tail -f logs/mae_finetune_<作业号>.out
# =============================================================================
#SBATCH -J mae-finetune
#SBATCH -p kshcnormal          # ← 改成 whichpartition 里有卡的队列
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1           # ← NVIDIA: gpu:1；国产 DCU: dcu:1
#SBATCH -t 24:00:00
#SBATCH -o logs/mae_finetune_%j.out
#SBATCH -e logs/mae_finetune_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

echo "======== 微调开始 $(date) job=${SLURM_JOB_ID:-local} node=${SLURMD_NODENAME:-?} ========"

CONDA_SH="${HOME}/miniconda3/etc/profile.d/conda.sh"
if [[ ! -f "$CONDA_SH" ]]; then
  echo "找不到 $CONDA_SH" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$CONDA_SH"
conda activate dafatt

module purge
# NVIDIA:
# module load nvidia/cuda/11.4
# 国产 DCU:
# module load compiler/devtoolset/7.3.1 mpi/hpcx/2.11.0/gcc-7.3.1
# module load compiler/dtk/23.10

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

ENCODER_DIR=/public/home/linhonglei/ChaoSuan
CUBES=/public/home/linhonglei/train_data
LABELS=/public/home/linhonglei/label_dat_remove_none_global
OUT=/public/home/linhonglei/ChaoSuan/finetune

if [[ -f "$ENCODER_DIR/encoder.pt" ]]; then
  ENCODER="$ENCODER_DIR/encoder.pt"
elif [[ -f "$ENCODER_DIR/checkpoints/encoder.pt" ]]; then
  ENCODER="$ENCODER_DIR/checkpoints/encoder.pt"
else
  ENCODER="$(find "$ENCODER_DIR" -name 'encoder.pt' -type f 2>/dev/null | head -n 1 || true)"
fi
if [[ -z "${ENCODER:-}" || ! -f "$ENCODER" ]]; then
  echo "在 $ENCODER_DIR 下找不到 encoder.pt" >&2
  echo "该目录现有文件：" >&2
  ls -lh "$ENCODER_DIR" >&2 || true
  exit 1
fi
if [[ ! -e "$CUBES" || ! -e "$LABELS" ]]; then
  echo "立方体或标签路径不存在:" >&2
  echo "  CUBES=$CUBES" >&2
  echo "  LABELS=$LABELS" >&2
  exit 1
fi
mkdir -p "$OUT"

echo "ENCODER=$ENCODER  ($(du -h "$ENCODER" | cut -f1))"
echo "CUBES=$CUBES"
echo "LABELS=$LABELS"
echo "OUT=$OUT"
ls -ld "$CUBES" "$LABELS"
python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve() / 'app'))
from identification.io import format_listing_report
print(format_listing_report(r'''$CUBES''', kind='立方体'))
print(format_listing_report(r'''$LABELS''', kind='标签'))
"

echo "======== 设备检查 ========"
nvidia-smi || true
command -v rocm-smi >/dev/null && rocm-smi || true
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available());
print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU-ONLY')"

# --max-per-class 200：每类最多 200 个像元（少样本，一轮通常几分钟到几十分钟）
# 要用全部标注：把 200 改成 0（像元一多，一轮可能数小时）
# --freeze-encoder 只训分类头；这里默认整网微调。只要线性探针就加上这一行。
python scripts/run_mae.py finetune \
  --encoder "$ENCODER" \
  --data "$CUBES" \
  --label "$LABELS" \
  --output "$OUT" \
  --num-classes 23 \
  --max-per-class 200 \
  --epochs 40 \
  --batch-size 32 \
  --device cuda:0

echo "======== 微调结束 $(date) ========"
echo "最佳模型: $OUT/model_best.pth"
