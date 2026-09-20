#!/bin/bash
#SBATCH -J mae-pretrain
#SBATCH -p kshcnormal          # 先运行 whichpartition，改成你有权限的队列
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1           # 国产 DCU 队列改成: --gres=dcu:1
#SBATCH -t 48:00:00
#SBATCH -o logs/mae_pretrain_%j.out
#SBATCH -e logs/mae_pretrain_%j.err

# 超算互联网 (SCNet) 示例。不要 module load anaconda cuda（这两个名字不存在）。
# 官方说明:
#   https://www.scnet.cn/help/docs/mainsite/ai/appendix/environment-conda/
#   https://www.scnet.cn/help/docs/mainsite/hpc/cmd/install-software/

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

# 1) 自己家目录里的 miniconda（登录节点先装好，见下方注释）
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate dafatt

# 2) 加速卡运行库：二选一，以 module avail 为准
module purge
# NVIDIA 中心（昆山/山东/雄安等有 RTX/A100 队列）:
# module load nvidia/cuda/11.4
# 国产 DCU 中心（队列名常带 hd，如 kshdtest）:
# module load compiler/devtoolset/7.3.1 mpi/hpcx/2.11.0/gcc-7.3.1
# module load compiler/dtk/23.10

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

DATA=/public/home/${USER}/crism_unlabeled
OUT=/public/home/${USER}/mae/pretrain

python -c "import torch; print('cuda', torch.cuda.is_available())"
python scripts/run_mae.py pretrain \
  --data "$DATA" \
  --output "$OUT" \
  --epochs 30 \
  --samples-per-epoch 8192 \
  --batch-size 64 \
  --num-readers 8 \
  --preprocess crop \
  --device cuda:0 \
  --resume
