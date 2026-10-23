#!/bin/bash
# 本机跑：bash scripts/42-download-results.sh [图名，默认 kodim19] [数据集] [sigma]
#
# 把并排图需要的四张输出图从 Roihu 拉回本机，供 61-make-sidebyside.py 用。
# 只读远端，幂等，可重复跑。
#
# 前置（在 Roihu 上各跑一次，图才存在）：
#   1) DiffPIR 与 DRUNet 的 run 本身 save_E=True，结果已经在 results/ 下了，不用重跑
#   2) 经典插值那一路以前不存图，要补一次（CPU，两分钟）：
#        python3 50-classical-baseline.py --testset kodak24_c256 --sigma 0.05 \
#                --save-dir results/classical_kodak24_c256_sigma0.05
#
# 四个 results 子目录名可以用环境变量覆盖（名字里带 lambda/NFE，换工作点就会变）：
#   DP_DIR_DIFFPIR / DP_DIR_DRUNET / DP_DIR_CLASSICAL
set -eu

HOST="${ROIHU_HOST:-roihu-gpu}"
DIFFPIR_DIR="/projappl/project_2009811/wenzhu/DiffPIR"
STEM="${1:-kodim19}"
TESTSET="${2:-kodak24_c256}"
SIGMA="${3:-0.05}"
LOCAL="$(cd "$(dirname "$0")/.." && pwd)/results-for-figure/${TESTSET}_sigma${SIGMA}"

DIR_CLASSICAL="${DP_DIR_CLASSICAL:-results/classical_${TESTSET}_sigma${SIGMA}}"
# 下面两个是 result_name 的通配：testset_任务_..._sigma..._NFE..._lambda...
DIR_DRUNET="${DP_DIR_DRUNET:-results/${TESTSET}_demosaic_*drunet*sigma${SIGMA}*dcnone}"
DIR_DIFFPIR="${DP_DIR_DIFFPIR:-results/${TESTSET}_demosaic_*256x256_diffusion_uncond_sigma${SIGMA}*NFE100*lambda3.0*}"

mkdir -p "$LOCAL"

echo "===== 远端有哪些候选目录 ====="
ssh "$HOST" "cd $DIFFPIR_DIR && ls -d $DIR_CLASSICAL $DIR_DRUNET $DIR_DIFFPIR 2>/dev/null" || {
  echo "❌ 一个都没匹配上。先在 Roihu 上 ls results/ 看实际目录名，再用 DP_DIR_* 覆盖。"; exit 1; }

echo; echo "===== 拉 $STEM 的四张图 ====="
# 经典插值 + CFA 输入（50-classical-baseline.py --save-dir 产出）
scp "$HOST:$DIFFPIR_DIR/$DIR_CLASSICAL/${STEM}_classical.png" "$LOCAL/" || echo "  ! 没有 classical 图：先跑 50-classical-baseline.py --save-dir"
scp "$HOST:$DIFFPIR_DIR/$DIR_CLASSICAL/${STEM}_cfa.png"       "$LOCAL/" || echo "  ! 没有 cfa 图：同上"
# 两个方法的输出图（文件名是 <stem>_<model_name>.png）
scp "$HOST:$DIFFPIR_DIR/$DIR_DRUNET/${STEM}_*.png"  "$LOCAL/" || echo "  ! 没有 DRUNet 图"
scp "$HOST:$DIFFPIR_DIR/$DIR_DIFFPIR/${STEM}_*.png" "$LOCAL/" || echo "  ! 没有 DiffPIR 图"

echo; echo "===== 本机 $LOCAL ====="
ls -l "$LOCAL"

cat <<EOF

下一步（本机，只要 numpy / matplotlib / Pillow）：

  python3 scripts/61-make-sidebyside.py \\
      --gt-dir ../diffpir-data/$TESTSET --image $STEM.png \\
      --cfa $LOCAL --classical $LOCAL \\
      --drunet  $LOCAL/${STEM}_drunet_color.png \\
      --diffpir $LOCAL/${STEM}_256x256_diffusion_uncond.png \\
      --zoom 96,112,64 --title 'Kodak24 crop, sigma_n = 12.75/255' \\
      --out figures/sidebyside-${TESTSET}-sigma${SIGMA}.png
EOF
