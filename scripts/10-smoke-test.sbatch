#!/bin/bash
#SBATCH --job-name=diffpir-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# 用法：
#   source env.sh && mkdir -p logs
#   sbatch -A $DIFFPIR_ACCOUNT -p $DIFFPIR_TEST_PARTITION --gres=$DIFFPIR_GRES \
#          10-smoke-test.sbatch deblur_in
#
# 目的：只证明「权重能加载 + GPU 能用 + demo_test 五张图能出结果」。不看指标。
# demo_test 只有 5 张，100 NFE 在 GH200 上是分钟级，15 分钟够。

set -eux
CFG="${1:-deblur_in}"
source "$SLURM_SUBMIT_DIR/env.sh"
module purge
module load "$DIFFPIR_MODULE"
cd "$SLURM_SUBMIT_DIR"

nvidia-smi
python3 -c "import torch,platform;print(platform.machine(),torch.__version__,torch.cuda.is_available(),torch.cuda.get_device_name(0))"

# 权重完整性检查：download.sh 不检查文件已存在就重下，会把完整的 2.1G 截断。
# 不检查的话，作业会跑到一半才报 pickle 解析错误，白排一次队。
CKPT="model_zoo/$(grep '^model_name:' "configs/${CFG}.yaml" | awk '{print $2}').pt"
SZ=$(stat -c %s "$CKPT" 2>/dev/null || echo 0)
echo "checkpoint: $CKPT  ($((SZ/1024/1024)) MB)"
if [ "$SZ" -lt 300000000 ]; then
  echo "❌ 权重不完整（<300MB），多半是 download.sh 还在下。等 dl.log 跑完再提交。"
  exit 1
fi

srun python3 main_ddpir.py --opt "configs/${CFG}.yaml"
echo "SMOKE OK"
