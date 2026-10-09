#!/bin/bash
#SBATCH --job-name=diffpir-collect
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# 10-09 块 2 的「挂上去不盯」作业：一个 job 里跑完 README 还缺的三组数字。
# 前置：python3 31-patch-env.py（把配置项改成读环境变量）
#
#   sbatch -A $DIFFPIR_ACCOUNT -p $DIFFPIR_TEST_PARTITION --gres=$DIFFPIR_GRES 31-step2-collect.sbatch
#
# 1) mcm18_c256 @ 定稿工作点（λ=3 / ζ=1.0 / 100 NFE / σ=0.05）—— 第二个数据集，Step 2 目前只有 Kodak24
# 2) kodak24_c256 @ 50 NFE  ┐ 补 quality-vs-NFE 曲线：σ=0.05 这档现在只有 100 NFE
# 3) kodak24_c256 @ 20 NFE  ┘ （σ=0 那档的 20/100 不同口径，不能混用）
#
# ⚠️ λ=3 是在 100 NFE 上扫出来的。50 / 20 NFE 这两点是「固定工作点下降 NFE」，
#    不是各自的最优 λ —— README 里要这么写，别说成「NFE 降到 20 仍然最优」。

set -eux
source "$SLURM_SUBMIT_DIR/env.sh"
module purge
module load "$DIFFPIR_MODULE"
cd "$SLURM_SUBMIT_DIR"

# §4.1 / 计划 §8 待回填的版本号，顺手打进日志
python3 -c "import torch; print('VERSIONS torch', torch.__version__, '| cuda', torch.version.cuda, '|', torch.cuda.get_device_name(0))"

export DP_MODEL=256x256_diffusion_uncond
export DP_SIGMA=0.05
export DP_LAMBDA=3
export DP_ZETA=1.0

run () {   # $1 = testset, $2 = NFE
  export DP_TESTSET="$1" DP_NFE="$2"
  t0=$SECONDS
  srun python3 main_ddpir_demosaic.py
  echo "WALL $1 NFE$2 $((SECONDS - t0)) s"
}

run mcm18_c256   100
run kodak24_c256 50
run kodak24_c256 20

echo "DONE collect"
