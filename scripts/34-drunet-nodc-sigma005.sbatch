#!/bin/bash
#SBATCH --job-name=drunet-nodc-s005
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# 方案 B，DRUNet 这一半：sigma=0.05，关掉数据一致性投影（DP_DC=none），顺便扫 lambda。
#
# 10-16 作废的那次跑法（DP_DC=loop）把带噪测量每轮灌回 1/3 的输出，
# 给 PSNR 设了一个与先验无关的天花板（kodak 30.94 / mcm 31.35 dB）。
# sigma>0 时正确做法是不投影：闭式解里的 rho 已经在处理噪声了。
#
# lambda=0.23 是 DPIR 在无噪设定下的值，sigma>0 从没扫过，所以这里扫五个点。
#
# lambda 列表可以覆盖：
#   sbatch ... --export=ALL,DP_LAMBDAS="0.01 0.02 0.05 0.1"
#
# ⚠️ rho_k = lambda*sigma_y^2/sigma_k^2，lambda 越小 rho 越小，而 rho->0 时闭式解
#    在实测点上退化成 x=y —— 又把带噪测量钉回去了（= dc=loop 的 28.32 dB）。
#    所以曲线在 lambda<0.1 某处必然掉头，往下扫是为了找拐点，不是越小越好。
#
#   source env.sh
#   sbatch -A $DIFFPIR_ACCOUNT -p $DIFFPIR_TEST_PARTITION --gres=$DIFFPIR_GRES 34-drunet-nodc-sigma005.sbatch

set -eux
source "$SLURM_SUBMIT_DIR/env.sh"
module purge
module load "$DIFFPIR_MODULE"
cd "$SLURM_SUBMIT_DIR"

export DP_NFE=40
export DP_X8=0
export DP_SIGMA=0.05
export DP_DC=none

for ds in kodak24_c256 mcm18_c256; do
  for lam in ${DP_LAMBDAS:-0.1 0.23 0.5 1 2}; do
    export DP_TESTSET="$ds" DP_LAMBDA="$lam"
    t0=$SECONDS
    srun python3 main_ddpir_demosaic_drunet.py
    echo "WALL $ds lambda=$lam $((SECONDS - t0)) s"
  done
done

echo "DONE drunet-nodc-s005"
