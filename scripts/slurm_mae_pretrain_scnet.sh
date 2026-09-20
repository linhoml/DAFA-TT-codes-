#!/bin/bash
# =============================================================================
# 超算互联网 (SCNet) — MAE 自监督预训练（第三步：提交作业）
#
# 登录节点上先做完 1–2 步（家目录 Miniconda + conda 环境 dafatt）。
# 本脚本只在计算节点上跑，不要在登录节点 python 开 GPU，也不要开图形界面。
#
# ---------- 提交前在登录节点敲这几条 ----------
#   cd ~/DAFA-TT-codes-          # 改成你的代码目录
#   mkdir -p logs
#   whichpartition               # 看你有权限的队列名和加速卡类型
#   # 用文本编辑器改本文件 4 处：
#   #   1) #SBATCH -p 后面的队列名
#   #   2) #SBATCH --gres=  gpu:1 或 dcu:1
#   #   3) 下面 module load 那一两行（取消注释你中心有的）
#   #   4) DATA / OUT 路径
#   sbatch scripts/slurm_mae_pretrain_scnet.sh
#   squeue -u "$USER"
#   tail -f logs/mae_pretrain_<作业号>.out
# =============================================================================
#SBATCH -J mae-pretrain
#SBATCH -p kshcnormal          # ← 改成 whichpartition 里你有权限的队列
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1           # ← NVIDIA 用 gpu:1；国产 DCU 改成 dcu:1
#SBATCH -t 48:00:00
#SBATCH -o logs/mae_pretrain_%j.out
#SBATCH -e logs/mae_pretrain_%j.err

# 官方说明:
#   https://www.scnet.cn/help/docs/mainsite/ai/appendix/environment-conda/
#   https://www.scnet.cn/help/docs/mainsite/hpc/cmd/install-software/

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

echo "======== 作业开始 $(date) job=${SLURM_JOB_ID:-local} node=${SLURMD_NODENAME:-?} ========"
echo "提交目录: $PWD"
echo "分区: ${SLURM_JOB_PARTITION:-?}  卡: ${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-?}}"

# 1) 自己家目录里的 miniconda（登录节点先装好）
#    若你装到了别的路径，只改这一行。
CONDA_SH="${HOME}/miniconda3/etc/profile.d/conda.sh"
if [[ ! -f "$CONDA_SH" ]]; then
  echo "找不到 $CONDA_SH 。登录节点先装 Miniconda，或改 CONDA_SH。" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$CONDA_SH"
conda activate dafatt
echo "python: $(command -v python)"
python -c "import sys; print('python', sys.version.split()[0])"

# 2) 加速卡运行库：二选一，以登录节点 module avail 为准。
#    不要 module load anaconda / cuda（SCNet 没有这两个名字）。
module purge
# NVIDIA 中心（昆山/山东/雄安等有 RTX/A100 队列）:
# module load nvidia/cuda/11.4
# 国产 DCU 中心（队列名常带 hd，如 kshdtest）:
# module load compiler/devtoolset/7.3.1 mpi/hpcx/2.11.0/gcc-7.3.1
# module load compiler/dtk/23.10

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# 3) 数据必须在计算节点能直接读的盘上（家目录 /public 或中心的 work/scratch）。
#    不要指向 Windows 网盘 / RaiDrive。文件夹里放 ENVI .hdr+.img 或 .npy。
DATA=/public/home/${USER}/crism_unlabeled
OUT=/public/home/${USER}/mae/pretrain

if [[ ! -e "$DATA" ]]; then
  echo "数据路径不存在: $DATA" >&2
  echo "请先把无标签 CRISM 立方体拷到计算节点能读的目录，并改脚本里的 DATA。" >&2
  exit 1
fi
mkdir -p "$OUT"

echo "DATA=$DATA"
echo "OUT=$OUT"
ls -ld "$DATA" || true
python -c "
from pathlib import Path
p = Path(r'''$DATA''')
files = [x for x in p.rglob('*') if x.is_file()] if p.is_dir() else [p]
print(f'数据文件 {len(files)} 个（含 hdr/img/npy 等）')
for x in files[:8]:
    print(' ', x, x.stat().st_size, 'bytes')
if not files:
    raise SystemExit('数据目录是空的')
"

echo "======== 设备检查 ========"
nvidia-smi || true
command -v rocm-smi >/dev/null && rocm-smi || true
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available());
print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU-ONLY')"

# 若上面打印 cuda False / CPU-ONLY：先改 --gres 和 module load，不要硬跑 48 小时。
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

echo "======== 作业结束 $(date) ========"
echo "编码器应在: $OUT/checkpoints/encoder.pt"
