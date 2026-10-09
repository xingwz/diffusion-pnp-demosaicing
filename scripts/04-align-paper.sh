#!/bin/bash
# ❌❌ 2026-10-07 作废，不要再跑这个脚本。
#
# 两个原因：
# 1. 它写 `lambda_: 8.0`，但 sisr 的 lambda_ 是**倍率**不是 λ 本身：
#    main_ddpir.py:557 `lambdas = [config.lambda_*i for i in range(2,13)]`
#    → lambda_=8.0 扫的是 λ=16,24,…,96，全部超出论文的 8，结果反而更差。
#    要复现论文 λ=8，必须保持 `lambda_: 1.0` 再从日志取 λ=8 那组。
# 2. 它已被 05-full-repro.sh 取代——那个脚本写法正确，而且跑的是 ffhq_val_100 全 100 张。
#
# 留档只为说明「为什么当初写错了」，是面试材料的一部分。
# 登录节点跑：bash 04-align-paper.sh
# 把 sisr / inpaint 的 λ、ζ 对齐论文 Table 3，生成 *_paper.yaml 并提交。
# 论文 Table 3（σy=0.05 FFHQ）：SR×4 → λ=8.0 ζ=0.2 ；（σy=0）Inpaint(random) → λ=7.0 ζ=1.0
echo "❌ 本脚本已作废，见文件头注释。请用 05-full-repro.sh。"; exit 1

set -u
source "$(dirname "$0")/env.sh"; cd "$DIFFPIR_DIR"

echo "===== 改之前，先看现在写的是什么 ====="
for y in configs/sisr.yaml configs/inpaint.yaml; do
  echo "--- $y ---"; grep -nE "^(lambda|lambda_|zeta|iter_num|noise_level_img|sf|mask_type)" "$y"
done

echo; echo "===== 生成对齐论文的 config ====="
# sisr：lambda_ 在仓库里是个列表（日志里扫了 2..12），直接写成单值既消掉扫描也对上论文
sed -E -e 's/^lambda_?:.*/lambda_: 8.0/' -e 's/^zeta:.*/zeta: 0.2/' configs/sisr.yaml > configs/sisr_paper.yaml
sed -E -e 's/^lambda_?:.*/lambda_: 7.0/' -e 's/^zeta:.*/zeta: 1.0/' configs/inpaint.yaml > configs/inpaint_paper.yaml
for y in configs/sisr_paper.yaml configs/inpaint_paper.yaml; do
  echo "--- $y ---"; grep -nE "^(lambda|lambda_|zeta)" "$y"
done

echo; echo "===== 提交 ====="
mkdir -p logs
for t in sisr_paper inpaint_paper; do
  sbatch -A "$DIFFPIR_ACCOUNT" -p "$DIFFPIR_PARTITION" --gres="$DIFFPIR_GRES" 20-step1.sbatch "$t"
done
squeue --me
cat <<'MSG'

⚠️ 维护窗口 10-06 08:00 → 10-07 22:00。现在提交还能跑完（每个几分钟）。
   万一排队没赶上，作业会保留到维护结束后自动执行，不用重投。
   跑完看：grep "Average" logs/diffpir-step1-*.out
MSG
