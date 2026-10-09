#!/bin/bash
#SBATCH --job-name=diffpir-demosaic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00   # gputest 上限就是 15 min；换 gpumedium 时在命令行加 --time= 覆盖
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Step 2 — DiffPIR + Bayer mask。
#
# 先跑烟雾测试（5 张 demo_test、FFHQ 模型、sigma=0、20 NFE），只看掩码对不对：
#   source env.sh
#   sbatch -A $DIFFPIR_ACCOUNT -p $DIFFPIR_TEST_PARTITION --gres=$DIFFPIR_GRES 30-step2-demosaic.sbatch
#
# 验收：PSNR 30+ 才算掩码方向对。掉到 10-20 说明 R/B 位置或 CFA 相位错了，先别往下做。
#
# 过了再改 main_ddpir_demosaic.py 的 testset_name / model_name / noise_level_img，
# 换 -p $DIFFPIR_PARTITION 正式跑。

set -eux
source "$SLURM_SUBMIT_DIR/env.sh"
module purge
module load "$DIFFPIR_MODULE"
cd "$SLURM_SUBMIT_DIR"

srun python3 main_ddpir_demosaic.py
echo "DONE demosaic"
