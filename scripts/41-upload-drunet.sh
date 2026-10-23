#!/bin/bash
# 本机跑（不是登录节点）：bash 41-upload-drunet.sh
# DRUNet PnP-HQS 基线需要的东西。幂等，可重复跑。
set -eu

HOST="${ROIHU_HOST:-roihu-gpu}"
DIFFPIR_DIR="/projappl/project_2009811/wenzhu/DiffPIR"
LOCAL="$(cd "$(dirname "$0")/.." && pwd)"          # drafts/
DPIR="${DPIR_LOCAL:-$HOME/Desktop/DPIR}"           # 本机 DPIR 仓库

echo "===== 1. 脚本 ====="
scp "$LOCAL/main_ddpir_demosaic_drunet.py"            "$HOST:$DIFFPIR_DIR/"
for f in 32-step2-drunet.sbatch 33-step2-ablation.sbatch 33-patch-dc.py \
         34-drunet-nodc-sigma005.sbatch 35-diffpir-nodc-sigma005.sbatch; do
  scp "$LOCAL/diffpir-roihu/$f" "$HOST:$DIFFPIR_DIR/"
done

echo; echo "===== 2. DPIR 的三个依赖文件 ====="
ssh "$HOST" "mkdir -p $DIFFPIR_DIR/models"
scp "$DPIR/models/network_unet.py"   "$HOST:$DIFFPIR_DIR/models/"
scp "$DPIR/models/basicblock.py"     "$HOST:$DIFFPIR_DIR/models/"
scp "$DPIR/utils/utils_mosaic.py"    "$HOST:$DIFFPIR_DIR/utils/"

echo; echo "===== 3. DRUNet 权重（125M，已存在就跳过）====="
if ssh "$HOST" "[ -s $DIFFPIR_DIR/model_zoo/drunet_color.pth ]"; then
  echo "已存在，跳过"
else
  scp "$DPIR/model_zoo/drunet_color.pth" "$HOST:$DIFFPIR_DIR/model_zoo/"
fi

echo; echo "===== 4. 验收 ====="
ssh "$HOST" "cd $DIFFPIR_DIR && ls -lh model_zoo/drunet_color.pth models/network_unet.py models/basicblock.py utils/utils_mosaic.py main_ddpir_demosaic_drunet.py"

echo; echo "----- md5 对比（10-16 踩过：手动同步的旧版本跑出四组一模一样的结果）-----"
LOCAL_MD5=$(md5 -q "$LOCAL/main_ddpir_demosaic_drunet.py" 2>/dev/null || md5sum "$LOCAL/main_ddpir_demosaic_drunet.py" | cut -d" " -f1)
REMOTE_MD5=$(ssh "$HOST" "md5sum $DIFFPIR_DIR/main_ddpir_demosaic_drunet.py | cut -d' ' -f1")
echo "本机   : $LOCAL_MD5"
echo "Roihu  : $REMOTE_MD5"
if [ "$LOCAL_MD5" = "$REMOTE_MD5" ]; then
  echo "✅ 上传完成，md5 一致"
else
  echo "❌ md5 不一致 —— 别往下跑，先查 scp 是不是失败了"; exit 1
fi
