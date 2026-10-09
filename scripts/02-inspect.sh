#!/bin/bash
# 登录节点跑：bash 02-inspect.sh   （不占 GPU，几秒钟）
# 目的：在投作业前搞清楚 ①默认用哪个权重 ②默认用哪个测试集 ③有没有那三个已知坑
source "$(dirname "$0")/env.sh"; cd "$DIFFPIR_DIR"

echo "===== A. 测试集 ====="
ls testsets/ 2>/dev/null || echo "  没有 testsets/ 目录"
for d in testsets/*/; do printf "  %-28s %s 个文件\n" "$d" "$(ls "$d" 2>/dev/null | wc -l)"; done

echo; echo "===== B. configs ====="
ls configs/ 2>/dev/null || echo "  没有 configs/ 目录 → 走单任务脚本，配置写在 .py 顶部"

echo; echo "===== C. 各脚本的默认 model_name / testset_name / iter_num ====="
for f in main_ddpir_sisr.py main_ddpir_deblur.py main_ddpir_inpainting.py; do
  [ -f "$f" ] || continue
  echo "--- $f ---"
  grep -n -E "model_name|testset_name|iter_num|noise_level_img|sf *=|calc_LPIPS|sub_1_analytic" "$f" | head -20
done
[ -d configs ] && for y in configs/*.yaml; do echo "--- $y ---"; grep -n -E "model_name|testset_name|iter_num|noise_level_img|sf:|calc_LPIPS|sub_1_analytic" "$y"; done

echo; echo "===== D. 三个已知坑的扫描 ====="
echo "-- torch.load 出现在哪（torch 2.6+ weights_only 默认 True）--"
grep -rn "torch\.load" --include=*.py . | grep -v "\.ipynb" | head -20
echo "-- np.float / np.int / np.bool / np.object 旧别名（numpy 2.5 已删）--"
grep -rn -E "np\.(float|int|bool|object|complex)[^0-9a-zA-Z_]" --include=*.py . | head -20
echo "-- mpi4py（已确认装了，这里只是确认它在哪被 import）--"
grep -rn "mpi4py" --include=*.py . | head -5

echo; echo "===== E. 权重 ====="
ls -lh model_zoo/*.pt 2>/dev/null
