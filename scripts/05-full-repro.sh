#!/bin/bash
# 登录节点跑：bash 05-full-repro.sh
# 用 ffhq_val_100（论文的 100 张 hold-out）+ 论文 Table 3 超参，完整复现三个任务。
set -u
source "$(dirname "$0")/env.sh"; cd "$DIFFPIR_DIR"

echo "===== 1. 检查数据 ====="
N=$(ls testsets/ffhq_val_100/*.png 2>/dev/null | wc -l)
echo "testsets/ffhq_val_100: $N 张 png"
[ "$N" -eq 100 ] || { echo "❌ 不是 100 张，先把数据 scp 上来"; exit 1; }

echo; echo "===== 2. 生成 config（testset → ffhq_val_100，超参对齐论文 Table 3）====="
# deblur: λ=7 ζ=0.3（原 config 已对）；核随机化已在 03-patch.sh 修掉
sed -E 's/^testset_name:.*/testset_name: ffhq_val_100/' configs/deblur.yaml > configs/deblur_full.yaml
# sisr: ζ=0.2；lambda_ 保持 1.0（它是倍率，扫描 lambda_×[2..12]，含论文的 8）
sed -E -e 's/^testset_name:.*/testset_name: ffhq_val_100/' -e 's/^zeta:.*/zeta: 0.2/' \
       -e 's/^lambda_?:.*/lambda_: 1.0/' configs/sisr.yaml > configs/sisr_full.yaml
# inpaint: λ=7 ζ=1.0（σ=0，λ 实测无影响，但对齐论文便于交代）
sed -E -e 's/^testset_name:.*/testset_name: ffhq_val_100/' -e 's/^zeta:.*/zeta: 1.0/' \
       -e 's/^lambda_?:.*/lambda_: 7.0/' configs/inpaint.yaml > configs/inpaint_full.yaml
for y in configs/*_full.yaml; do
  echo "--- $y ---"; grep -nE "^(testset_name|lambda_?|zeta|iter_num|noise_level_img|sf|model_name):" "$y"
done

echo; echo "===== 3. 提交 ====="
mkdir -p logs
for t in deblur_full inpaint_full sisr_full; do
  sbatch -A "$DIFFPIR_ACCOUNT" -p "$DIFFPIR_PARTITION" --gres="$DIFFPIR_GRES" 20-step1.sbatch "$t"
done
squeue --me
cat <<'MSG'

预计：deblur ~100 s · inpaint ~30 s · sisr ~13 min（11 组 λ 扫描）
⚠️ 集群 10-06 08:00 停机，现在提交来得及。
看结果：
  grep -hE "Average (PSNR|LPIPS) of \(ffhq_val_100\)" logs/diffpir-step1-*.out
  SR 要取 lambda:8.0 那组：grep -A1 "zeta:0.2, lambda:8.0" logs/diffpir-step1-*.out
MSG
