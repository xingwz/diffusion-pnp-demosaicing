#!/usr/bin/env python3
"""给 main_ddpir_demosaic.py 的「recover conditional part」加一个 DP_DC 开关。

为什么（2026-10-16）：那一行 `x[mask] = y[mask]` 在 sigma=0 时写回的是干净真值，合法；
sigma>0 时写回的是**带噪测量**，等于把 1/3 的输出值设成「真值 + N(0,sigma)」。
实测这给 PSNR 设了一个与先验无关的天花板：

    kodak24_c256 sigma=0.05 -> 30.9384 dB
    mcm18_c256   sigma=0.05 -> 31.3503 dB

而已记录的 DiffPIR 值是 30.6704 / 31.2000 —— 离天花板只有 0.15–0.27 dB，
也就是说那两个数字 94% 的平方误差就是自己写回去的噪声，量的不是先验。

用法（在 Roihu 的 DiffPIR 目录下）：
    python3 33-patch-dc.py

幂等：已经打过补丁就直接退出。原文件备份为 main_ddpir_demosaic.py.pre-dc。

    DP_DC=final   默认，保持原行为（sigma=0 的历史结果照旧可复现）
    DP_DC=none    关掉投影，sigma>0 时用这个
"""
import os
import re
import shutil
import sys

P = "main_ddpir_demosaic.py"

src = open(P, encoding="utf-8").read()
if "DP_DC" in src:
    print("already patched — 不动")
    sys.exit(0)

SUBS = [
    # 1) 投影加条件
    (r"(?m)^(\s*)if generate_mode in \['repaint','DiffPIR'\]:\n(\s*)x\[mask\.to\(torch\.bool\)\] = y\[mask\.to\(torch\.bool\)\]$",
     r"\1if generate_mode in ['repaint','DiffPIR'] and os.environ.get('DP_DC', 'final') != 'none':\n\2x[mask.to(torch.bool)] = y[mask.to(torch.bool)]"),
    # 2) result_name 带上 dc，免得不同设定互相覆盖
    (r"(?m)^(\s*)result_name(\s*)= f'\{testset_name\}(.*)$",
     r"\1result_name\2= f'{testset_name}\3\n\1result_name\2= result_name + '_dc' + os.environ.get('DP_DC', 'final')"),
]

out = src
for pat, rep in SUBS:
    out, n = re.subn(pat, rep, out)
    if n != 1:
        sys.exit("ABORT: %r 匹配了 %d 次（应为 1 次），文件没有被改动" % (pat[:50], n))

if not os.path.exists(P + ".pre-dc"):
    shutil.copy2(P, P + ".pre-dc")
open(P, "w", encoding="utf-8").write(out)

print("patched. 改掉的行：")
for line in out.splitlines():
    if "DP_DC" in line:
        print("   " + line.strip())
