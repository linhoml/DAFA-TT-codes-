#!/bin/bash
#SBATCH -J mae-pretrain
#SBATCH -p gpu                 # 改成超算上的 GPU 分区名
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH -o logs/mae_pretrain_%j.out
#SBATCH -e logs/mae_pretrain_%j.err

# 在登录节点: mkdir -p logs && sbatch scripts/slurm_mae_pretrain.sh
# 改下面四行路径后提交。不要在计算节点上开 spectral_app.py 图形界面。

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

# ---- 按超算环境修改（不要照抄 anaconda / cuda，多数机器没有这两个名字）----
# module avail                    # 或: module spider python ; module spider cuda
# module load <本机列出的 python/conda 模块> <本机列出的 cuda 模块>
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate dafatt

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

DATA=/scratch/$USER/crism_unlabeled          # 无标签 CRISM 文件夹（本地盘/Lustre，不要网盘）
OUT=/scratch/$USER/mae/pretrain
PY=python                                    # 或 conda 环境里的 python

nvidia-smi || true
$PY -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"

$PY scripts/run_mae.py pretrain \
  --data "$DATA" \
  --output "$OUT" \
  --epochs 30 \
  --samples-per-epoch 8192 \
  --batch-size 64 \
  --num-readers 8 \
  --preprocess crop \
  --device cuda:0 \
  --resume
