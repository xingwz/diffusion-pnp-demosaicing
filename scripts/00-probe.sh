#!/bin/bash
# 在 Roihu 【登录节点】直接跑：bash 00-probe.sh
# 输出照抄进 drafts/diffusion-project-plan.md 第 8 节「环境记录」。
set -u

echo "===== 1. 我能用哪个 account ====="
sacctmgr -n show assoc user=$USER format=account%30,partition%20

echo; echo "===== 2. 哪些 partition 带 GPU / 时限多久 ====="
sinfo -o "%20P %10G %6D %10l %14a %t" | sort -u

echo; echo "===== 3. PyTorch / CUDA module 的真名 ====="
module -t avail 2>&1 | grep -i -E "^(pytorch|torch|cuda|python|anaconda|miniconda)" | sort -u

echo; echo "===== 4. 旧脚本里的东西在不在（CSC 专有，Roihu 大概率没有）====="
for p in /projappl /scratch; do
  [ -d "$p" ] && echo "存在: $p" || echo "不存在: $p   ← 旧脚本的 PYTHONUSERBASE 路径在这里失效"
done

echo; echo "===== 5. 代码和权重 ====="
D="${DIFFPIR_DIR:-$HOME/DiffPIR}"
[ -d "$D" ] && { echo "仓库: $D"; ls "$D"/main_ddpir*.py 2>/dev/null; echo "--- configs ---"; ls "$D"/configs 2>/dev/null; echo "--- model_zoo ---"; ls -lh "$D"/model_zoo 2>/dev/null; } || echo "找不到 $D，改 env.sh 里的 DIFFPIR_DIR"
