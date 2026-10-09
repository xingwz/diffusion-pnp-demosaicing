#!/bin/bash
# 在【登录节点】跑一次：bash 01-setup-login.sh
# 为什么不放进 sbatch：①装包要外网 ②每次作业重装纯属浪费
set -eu
source "$(dirname "$0")/env.sh"

module purge
module load "$DIFFPIR_MODULE"
mkdir -p "$PYTHONUSERBASE"

echo "===== 架构与 torch ====="
python3 - <<'PY'
import platform, torch, torchvision
print("machine  :", platform.machine(), "  ← 预期 aarch64（Grace ARM）")
print("torch    :", torch.__version__, "| cuda", torch.version.cuda)
print("tv       :", torchvision.__version__)
print("arch_list:", torch.cuda.get_arch_list(), "  ← 必须含 sm_90，否则 GH200 跑不了")
PY

echo; echo "===== 装缺的小包 ====="
# ⚠️ 两个 flag 都是必须的：
#  --no-deps        lpips 的依赖里有 torch/torchvision。不加它，pip 会往 PYTHONUSERBASE
#                   里再塞一份 PyPI 版 torch，盖掉 module 里那份对好 CUDA+sm_90 的，
#                   症状是 torch.cuda.is_available() 突然变 False。
#  --only-binary    Roihu 是 aarch64。没有现成 wheel 的包，pip 默认会去编译源码——
#                   要么卡二十分钟要么报一堆看不懂的编译错。加上它就立刻失败，好排查。
PIPQ="--user --only-binary=:all:"
pip install $PIPQ --no-deps 'lpips==0.1.4'   # 版本钉住：换版本 LPIPS 数值与论文表格不可比
pip install $PIPQ hdf5storage blobfile opencv-python-headless
# scipy/numpy/matplotlib/tqdm 一般 module 自带；缺哪个单独补，不要整条重装
# 若某个包报 "No matching distribution ... only-binary"，说明它没有 aarch64 wheel，
# 回来找我换方案（多半能用 conda-forge 的 aarch64 包或直接绕开）

echo; echo "===== 验证 ====="
python3 - <<'PY'
import importlib
for m in ["torch","torchvision","numpy","scipy","cv2","lpips","hdf5storage","blobfile","tqdm","matplotlib"]:
    try: importlib.import_module(m); print(f"  OK   {m}")
    except Exception as e: print(f"  FAIL {m}: {type(e).__name__}: {e}")
# guided_diffusion 的 dist_util 可能 import mpi4py。DiffPIR 的 requirements 把它注释掉了，
# 但万一报错：CSC 上有 mpi4py module，或把 dist_util 里那几行 import 注释掉（纯推理用不到分布式）
try:
    import mpi4py; print("  OK   mpi4py（有，不用管）")
except Exception: print("  --   mpi4py 没有。只有报错时才需要处理，见脚本注释")
PY

echo; echo "===== checkpoint（必须在登录节点下，计算节点没外网）====="
cd "$DIFFPIR_DIR"; ls -lh model_zoo/ 2>/dev/null || echo "  (空)"
cat <<'MSG'
还没下就跑: bash download.sh
下完确认两个文件：
  model_zoo/256x256_diffusion_uncond.pt   ← ImageNet 通用图像，Step 1 的 SR/deblur 用这个
  model_zoo/diffusion_ffhq_10m.pt         ← FFHQ 人脸；下下来若叫 ffhq_10m.pt 要改名，
                                             否则报找不到权重（要和 config 里的 model_name 对齐）
MSG
