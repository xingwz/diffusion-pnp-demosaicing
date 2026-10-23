#!/bin/bash
#SBATCH --job-name=drunet-dc-ablation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# 消融：sigma=0 下数据一致性投影 x[mask]=y[mask] 做几次。
#
# 为什么要这组：DRUNet PnP 在 sigma=0 上比 DiffPIR 高 7.42 dB（Kodak，40 vs 100 NFE）。
# 面试官必问「这是先验带来的，还是你多做了 40 次数据投影带来的」。这一组给出答案。
#
#   loop  = 每次迭代都投影（DPIR 自己的 demosaick 脚本）
#   final = 只在最后投影一次（DiffPIR 的 "recover conditional part"）
#   none  = 完全不投影
#
# sigma=0 时投影写回的是干净真值，三档都合法，差值就是投影本身的贡献。
#
#   source env.sh
#   sbatch -A $DIFFPIR_ACCOUNT -p $DIFFPIR_TEST_PARTITION --gres=$DIFFPIR_GRES 33-step2-ablation.sbatch

set -eux
source "$SLURM_SUBMIT_DIR/env.sh"
module purge
module load "$DIFFPIR_MODULE"
cd "$SLURM_SUBMIT_DIR"

export DP_NFE=40
export DP_LAMBDA=0.23
export DP_X8=0
export DP_SIGMA=0

for ds in kodak24_c256 mcm18_c256; do
  for dc in loop final none; do
    export DP_TESTSET="$ds" DP_DC="$dc"
    t0=$SECONDS
    srun python3 main_ddpir_demosaic_drunet.py
    echo "WALL $ds dc=$dc $((SECONDS - t0)) s"
  done
done

echo "DONE dc-ablation"
