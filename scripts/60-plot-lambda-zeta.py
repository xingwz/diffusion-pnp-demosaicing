#!/usr/bin/env python3
"""画 README §3.2.1 的 λ–ζ 图。数据来自 data/lambda_zeta_sweep.csv（36 组实跑结果）。

    python3 scripts/60-plot-lambda-zeta.py

输出 figures/lambda-zeta-sweep.png。
配色用的是单色蓝序数档（ζ 是有序参数，不是并列类别）：浅 → 深 对应 ζ 0.3 → 1.0。
"""
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SURFACE = "#fcfcfb"
INK = "#1a1a19"
INK_MUTED = "#6b6b68"
GRID = "#e4e3e0"
# 序数（有序）单色蓝，step 250 / 450 / 650
COLORS = {0.3: "#86b6ef", 0.7: "#2a78d6", 1.0: "#104281"}

rows = defaultdict(list)
with open(os.path.join(HERE, "data", "lambda_zeta_sweep.csv")) as f:
    for r in csv.DictReader(f):
        rows[float(r["zeta"])].append((int(r["lambda"]), float(r["psnr"]), float(r["lpips"])))
for z in rows:
    rows[z].sort()

plt.rcParams.update({
    "font.size": 10, "font.family": "DejaVu Sans",
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
})

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), dpi=200)

panels = [
    (axes[0], 2, "PSNR (dB) — higher is better", max, (16, -4)),
    (axes[1], 3, "LPIPS (VGG) — lower is better", min, (16, 10)),
]

for ax, idx, ylabel, pick, star_off in panels:
    for z in sorted(rows):
        lam = [p[0] for p in rows[z]]
        val = [p[idx - 1] for p in rows[z]]
        ax.plot(lam, val, color=COLORS[z], linewidth=2, marker="o", markersize=5.5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3, clip_on=False)
        # 直接标注曲线身份（不只靠图例）；右图末端 1.0 与 0.3 太近，上下错开
        dy = 0
        if idx == 3:
            dy = {1.0: 7, 0.3: -7}.get(z, 0)
        ax.annotate(f"ζ = {z}", xy=(lam[-1], val[-1]), xytext=(5, dy),
                    textcoords="offset points", va="center", ha="left",
                    color=COLORS[z], fontsize=9, fontweight="bold")
        # 只标最优点，不是每点都写数
        best = pick(rows[z], key=lambda p: p[idx - 1])
        if z == 1.0:
            ax.annotate(f"λ* = {best[0]}  ({best[idx - 1]:.4f})",
                        xy=(best[0], best[idx - 1]), xytext=star_off,
                        textcoords="offset points", color=INK, fontsize=9,
                        arrowprops=dict(arrowstyle="-", color=INK_MUTED, linewidth=1))

    ax.set_xlabel("data-fidelity weight  λ")
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(1, 13))
    ax.set_xlim(0.6, 13.6)
    ax.margins(y=0.16)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)

# 论文自己的 λ 区间，作为参照带
for ax in axes:
    ax.axvspan(7, 9, color="#efefec", zorder=1)
    ax.annotate("λ the paper uses\nfor its own tasks", xy=(8, 1.0), xycoords=("data", "axes fraction"),
                xytext=(0, -2), textcoords="offset points", ha="center", va="top",
                color=INK_MUTED, fontsize=8)

fig.suptitle("Bayer demosaicing + denoising: the data-fidelity optimum moves with ζ",
             x=0.009, ha="left", fontsize=12.5, fontweight="bold")
fig.text(0.009, 0.885,
         "Kodak24, 24 × 256×256 crops · σₙ = 12.75/255 · 100 NFE · 36 runs",
         ha="left", fontsize=9, color=INK_MUTED)
fig.tight_layout(rect=(0, 0, 1, 0.88))

out = os.path.join(HERE, "figures", "lambda-zeta-sweep.png")
fig.savefig(out, facecolor=SURFACE)
print("wrote", out)
