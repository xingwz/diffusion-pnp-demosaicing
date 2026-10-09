# DiffPIR on Roihu — 所有脚本共用的变量。改这一个文件就够。
# 2026-10-05 已按 00-probe.sh 的实际输出填满，不用再改。

export DIFFPIR_ACCOUNT="project_2009811"      # 另一个可选：project_2020049

# gputest   : gh200×1, 15:00,        3 idle  → 烟雾测试
# gpumedium : gh200,   1-12:00:00,  48 idle  → Step 1 正式跑
# gpularge 是给多节点的，这个项目单卡就够，别占
export DIFFPIR_PARTITION="gpumedium"
export DIFFPIR_TEST_PARTITION="gputest"
export DIFFPIR_GRES="gpu:gh200:1"             # ⚠️ 不是 v100

export DIFFPIR_DIR="/projappl/project_2009811/wenzhu/DiffPIR"

# pip --user 的安装目标。放在自己那层，和代码并排，重装时整个删掉即可。
export PYTHONUSERBASE="/projappl/project_2009811/wenzhu/diffpir-env"

# ⚠️ module 真名是 python-pytorch，不是 pytorch —— 这是刚才 Lmod 报 unknown 的原因。
# 可选 2.10 / 2.13。钉死版本，不要用默认，否则哪天默认变了结果就不可复现。
export DIFFPIR_MODULE="python-pytorch/2.13"
