#!/bin/bash
#SBATCH --job-name=drunet-pnp-demosaic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00   # gputest 上限 15 min；40 NFE×42 张在 GH200 上远远够
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Step 2 第 2 行 —— DRUNet PnP-HQS 基线（diffusion-project-plan.md §3.3）。
# 和 DiffPIR 那次 run 共用 Bayer mask、data 子问题、测试集、PSNR/LPIPS 口径，
# 唯一的变量是先验：扩散模型 → DRUNet，DDIM → DPIR 的 HQS 调度。
#
#   source env.sh
#   sbatch -A $DIFFPIR_ACCOUNT -p $DIFFPIR_TEST_PARTITION --gres=$DIFFPIR_GRES 32-step2-drunet.sbatch
#
# 先跑 SMOKE_ONLY=1 看 demo_test 一组（见下），数字合理再跑全部四组。
#
# 验收：σ=0 的 Kodak24 应该在 **36-39 dB** 量级（DPIR 论文 Kodak 去马赛克 ~39-40 dB，
# 我们这里是 256×256 裁剪 + border=0，低一点正常）。掉到 20 出头说明 CFA 相位或
# 值域错了 —— 先别往下跑，把日志贴出来。

set -eux
source "$SLURM_SUBMIT_DIR/env.sh"
module purge
module load "$DIFFPIR_MODULE"
cd "$SLURM_SUBMIT_DIR"

python3 -c "import torch; print('VERSIONS torch', torch.__version__, '| cuda', torch.version.cuda, '|', torch.cuda.get_device_name(0))"

# DPIR 去马赛克的默认工作点，不扫参
export DP_NFE=40
export DP_LAMBDA=0.23
export DP_X8=0          # DiffPIR 不做几何自集成，基线也不做

run () {   # $1 = testset, $2 = sigma
  export DP_TESTSET="$1" DP_SIGMA="$2"
  t0=$SECONDS
  srun python3 main_ddpir_demosaic_drunet.py
  echo "WALL $1 sigma$2 NFE$DP_NFE $((SECONDS - t0)) s"
}

if [ "${SMOKE_ONLY:-0}" = "1" ]; then
  run demo_test 0
else
  run kodak24_c256 0
  run kodak24_c256 0.05
  run mcm18_c256   0
  run mcm18_c256   0.05
fi

echo "DONE drunet-pnp"
