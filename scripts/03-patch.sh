#!/bin/bash
# 登录节点跑：bash 03-patch.sh   （可重复运行，不会改两次）
# 1) 修 torch 2.13 / numpy 2.5 带来的两个不兼容
# 2) 看 FFHQ 权重下来没有；没有就生成用 ImageNet 权重的 config
set -u
source "$(dirname "$0")/env.sh"; cd "$DIFFPIR_DIR"

echo "===== 1. torch.load：补 weights_only=False ====="
# torch 2.6+ 把默认改成 True，加载 guided-diffusion 的 .pt 会抛 UnpicklingError。
# 这两个 checkpoint 是 OpenAI 官方权重、你自己下的，可信，显式关掉即可。
for f in main_ddpir.py main_ddpir_sisr.py main_ddpir_deblur.py main_ddpir_inpainting.py; do
  [ -f "$f" ] || continue
  if grep -q 'weights_only' "$f"; then
    echo "  已改过，跳过: $f"
  else
    cp -n "$f" "$f.bak"
    sed -i 's/torch\.load(args\.model_path, map_location="cpu")/torch.load(args.model_path, map_location="cpu", weights_only=False)/' "$f"
    grep -q 'weights_only' "$f" && echo "  ✅ $f" || echo "  ❌ $f 没匹配上，手工改"
  fi
done

echo; echo "===== 2. numpy 2.5：np.int 已被删除 ====="
f=guided_diffusion/resample.py
if grep -q 'dtype=np\.int)' "$f" 2>/dev/null; then
  cp -n "$f" "$f.bak"
  sed -i 's/dtype=np\.int)/dtype=int)/' "$f"
  echo "  ✅ $f:132  np.int → int"
else
  echo "  已改过或不存在，跳过"
fi

echo; echo "===== 3. FFHQ 权重到了吗 ====="
ls -lh model_zoo/*.pt 2>/dev/null
HAVE_FFHQ=0
[ -f model_zoo/diffusion_ffhq_10m.pt ] && HAVE_FFHQ=1
# 常见情况：下下来叫 ffhq_10m.pt，要改名才能和 config 的 model_name 对齐
if [ $HAVE_FFHQ -eq 0 ] && [ -f model_zoo/ffhq_10m.pt ]; then
  mv model_zoo/ffhq_10m.pt model_zoo/diffusion_ffhq_10m.pt
  echo "  ✅ 已把 ffhq_10m.pt 改名为 diffusion_ffhq_10m.pt"; HAVE_FFHQ=1
fi
[ -f dl.log ] && { echo "  --- dl.log 末尾 ---"; tail -5 dl.log; }

echo; echo "===== 4. 生成 ImageNet 权重版 config ====="
# 三个默认 config 都写死 model_name: diffusion_ffhq_10m。
# 你手上一定有 256x256_diffusion_uncond，所以无论 FFHQ 下没下来，先用它跑通流程。
for t in deblur sisr inpaint; do
  src="configs/${t}.yaml"; dst="configs/${t}_in.yaml"
  [ -f "$src" ] || continue
  sed 's/^model_name:.*/model_name: 256x256_diffusion_uncond/' "$src" > "$dst"
  echo "  ✅ $dst  (model_name → 256x256_diffusion_uncond，其余参数原样)"
done

echo; echo "===== 下一步 ====="
cat <<MSG
烟雾测试（demo_test 只有 5 张图，gputest 的 15 分钟绰绰有余）：

  source env.sh && mkdir -p logs
  sbatch -A \$DIFFPIR_ACCOUNT -p \$DIFFPIR_TEST_PARTITION --gres=\$DIFFPIR_GRES \\
         10-smoke-test.sbatch deblur_in

MSG
[ $HAVE_FFHQ -eq 1 ] && echo "FFHQ 权重在 → Step 1 两份权重都能跑，论文 FFHQ / ImageNet 两张表都能对。" \
                     || echo "FFHQ 权重还没到 → 先用 *_in.yaml 跑通；下来了再补 FFHQ 那组，不阻塞。"
