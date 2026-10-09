#!/usr/bin/env python3
"""把 main_ddpir_demosaic.py 的六个配置项改成读环境变量，这样一个 sbatch 里能连跑多组配置，
不用再手改文件（10-08 的 λ/ζ 扫描是手改的，容易把上一组的值留在文件里）。

用法（在 Roihu 的 DiffPIR 目录下）：
    python3 31-patch-env.py

幂等：已经打过补丁就直接退出。原文件备份为 main_ddpir_demosaic.py.pre-env。

环境变量（都有默认值，不设就是原来的烟雾测试配置）：
    DP_SIGMA   noise_level_img，[0,1] 刻度（12.75/255 → 0.05）
    DP_MODEL   diffusion_ffhq_10m | 256x256_diffusion_uncond
    DP_TESTSET demo_test | kodak24_c256 | mcm18_c256
    DP_NFE     iter_num（skip 会自动跟着重算）
    DP_LAMBDA  lambda_
    DP_ZETA    zeta
"""
import os
import re
import shutil
import sys

P = "main_ddpir_demosaic.py"

src = open(P, encoding="utf-8").read()
if "DP_SIGMA" in src:
    print("already patched — 不动")
    sys.exit(0)

SUBS = [
    (r"(?m)^(\s*)noise_level_img(\s*)=.*$",
     r"\1noise_level_img\2= float(os.environ.get('DP_SIGMA', '0'))"),
    (r"(?m)^(\s*)model_name(\s*)=.*$",
     r"\1model_name\2= os.environ.get('DP_MODEL', 'diffusion_ffhq_10m')"),
    (r"(?m)^(\s*)testset_name(\s*)=.*$",
     r"\1testset_name\2= os.environ.get('DP_TESTSET', 'demo_test')"),
    (r"(?m)^(\s*)iter_num(\s*)=.*$",
     r"\1iter_num\2= int(os.environ.get('DP_NFE', '20'))"),
    (r"(?m)^(\s*)lambda_(\s*)=.*$",
     r"\1lambda_\2= float(os.environ.get('DP_LAMBDA', '1'))"),
    (r"(?m)^(\s*)zeta(\s*)=.*$",
     r"\1zeta\2= float(os.environ.get('DP_ZETA', '1.0'))"),
    # 扫描循环收成单点：扫描靠多次提交，不靠文件里的 range
    (r"(?m)^(\s*)lambdas\s*=.*$", r"\1lambdas = [lambda_]"),
    (r"(?m)^(\s*)for zeta_i in .*$", r"\1for zeta_i in [zeta]:"),
]

out = src
for pat, rep in SUBS:
    out, n = re.subn(pat, rep, out)
    if n != 1:
        sys.exit("ABORT: %r 匹配了 %d 次（应为 1 次），文件没有被改动" % (pat, n))

shutil.copy2(P, P + ".pre-env") if not os.path.exists(P + ".pre-env") else None
open(P, "w", encoding="utf-8").write(out)

print("patched. 改掉的 8 行：")
for line in out.splitlines():
    if "DP_" in line or re.match(r"^\s*(lambdas\s*=|for zeta_i in)", line):
        print("   " + line.strip())
