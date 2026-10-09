#!/bin/bash
# 本机跑（不是登录节点）：bash 40-upload.sh
# 把 Step 2 需要的代码和数据传到 Roihu。幂等，可重复跑。
set -eu

HOST="${ROIHU_HOST:-roihu-gpu}"
DIFFPIR_DIR="/projappl/project_2009811/wenzhu/DiffPIR"
DATA_DIR="/scratch/project_2009811/wenzhu/diffpir-data"
LOCAL="$(cd "$(dirname "$0")/.." && pwd)"          # drafts/

echo "===== 0. 先备份 Roihu 上会被覆盖的那个文件 ====="
ssh "$HOST" "cd $DIFFPIR_DIR && [ -f utils/utils_inpaint.py.bak ] || cp utils/utils_inpaint.py utils/utils_inpaint.py.bak; ls -l utils/utils_inpaint.py*"

echo; echo "===== 1. 代码 ====="
scp "$LOCAL/main_ddpir_demosaic.py"              "$HOST:$DIFFPIR_DIR/"
scp "$LOCAL/utils_inpaint.py"                    "$HOST:$DIFFPIR_DIR/utils/utils_inpaint.py"
scp "$LOCAL/make_crops.py"                       "$HOST:$DIFFPIR_DIR/"
scp "$LOCAL/diffpir-roihu/30-step2-demosaic.sbatch" "$HOST:$DIFFPIR_DIR/"

echo; echo "===== 2. 数据（42 张 256x256 裁剪 + 清单）====="
ssh "$HOST" "mkdir -p $DATA_DIR"
scp -r "$LOCAL/diffpir-data/kodak24_c256" "$HOST:$DATA_DIR/"
scp -r "$LOCAL/diffpir-data/mcm18_c256"   "$HOST:$DATA_DIR/"

echo; echo "===== 3. 软链接挂进 testsets/ ====="
ssh "$HOST" "cd $DIFFPIR_DIR/testsets && ln -sfn $DATA_DIR/kodak24_c256 kodak24_c256 && ln -sfn $DATA_DIR/mcm18_c256 mcm18_c256 && ls -l | grep c256"

echo; echo "===== 4. 验收 ====="
ssh "$HOST" "cd $DIFFPIR_DIR && \
  echo -n 'bayer_mask 存在: '; grep -c 'def bayer_mask' utils/utils_inpaint.py; \
  echo -n '切片写法(应为 i::2): '; grep -o 'm\[i::\?[0-9.]*,' utils/utils_inpaint.py; \
  echo -n 'kodak 张数: '; ls testsets/kodak24_c256/*.png | wc -l; \
  echo -n 'mcm   张数: '; ls testsets/mcm18_c256/*.png | wc -l"
echo; echo "✅ 上传完成"
