#!/bin/bash
#SBATCH --job-name=diffpir-step1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# 用法：
#   source env.sh
#   for t in sisr deblur inpaint; do
#     sbatch -A $DIFFPIR_ACCOUNT -p $DIFFPIR_PARTITION --gres=$DIFFPIR_GRES 20-step1.sbatch $t
#   done
# 三个任务分三次投：挂一个不影响另外两个，日志也好读。

set -eux
TASK="${1:-sisr}"
source "$SLURM_SUBMIT_DIR/env.sh"
module purge
module load "$DIFFPIR_MODULE"
cd "$SLURM_SUBMIT_DIR"

if [ -f "configs/${TASK}.yaml" ]; then
  srun python3 main_ddpir.py --opt "configs/${TASK}.yaml"
else
  case "$TASK" in
    sisr)    srun python3 main_ddpir_sisr.py ;;
    deblur)  srun python3 main_ddpir_deblur.py ;;
    inpaint) srun python3 main_ddpir_inpainting.py ;;
    *) echo "未知任务: $TASK"; exit 1 ;;
  esac
fi
echo "DONE $TASK"
