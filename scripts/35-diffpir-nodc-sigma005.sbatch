#!/bin/bash
#SBATCH --job-name=diffpir-nodc-s005
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# 方案 B，DiffPIR 这一半：sigma=0.05，关掉 "recover conditional part"，重扫 lambda。
# 前置：python3 31-patch-env.py && python3 33-patch-dc.py
#
# ⚠️ 用 gpumedium，不要用 gputest：100 NFE × 4 组 + 四次 2.1G 权重加载，15 分钟不够。
#   source env.sh
#   sbatch -A $DIFFPIR_ACCOUNT -p $DIFFPIR_PARTITION --gres=$DIFFPIR_GRES 35-diffpir-nodc-sigma005.sbatch
#
# ⚠️ lambda=3 是在**带投影**的设定下扫出来的。投影一去掉，data step 的权重就真正起作用了，
#    lambda* 很可能变。所以这里重扫，不能直接沿用 3。

set -eux
source "$SLURM_SUBMIT_DIR/env.sh"
module purge
module load "$DIFFPIR_MODULE"
cd "$SLURM_SUBMIT_DIR"

export DP_MODEL=256x256_diffusion_uncond
export DP_NFE=100
export DP_SIGMA=0.05
export DP_ZETA=1.0
export DP_DC=none

run () {   # $1 = testset, $2 = lambda
  export DP_TESTSET="$1" DP_LAMBDA="$2"
  t0=$SECONDS
  srun python3 main_ddpir_demosaic.py
  echo "WALL $1 lambda=$2 $((SECONDS - t0)) s"
}

run kodak24_c256 1
run kodak24_c256 3
run kodak24_c256 6
run mcm18_c256   3

echo "DONE diffpir-nodc-s005"
