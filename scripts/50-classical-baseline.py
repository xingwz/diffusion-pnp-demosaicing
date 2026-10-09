#!/usr/bin/env python3
"""经典 CFA 插值基线——和 DiffPIR 走同一条退化路径、同一套指标。

在 DiffPIR 仓库根目录下跑：
    module load python-pytorch/2.13
    python3 50-classical-baseline.py --testset kodak24_c256 --sigma 0
    python3 50-classical-baseline.py --testset mcm18_c256   --sigma 0
    python3 50-classical-baseline.py --testset kodak24_c256 --sigma 0.05   # 无去噪，预期很差

退化、指标都直接复用仓库自己的代码，保证和 main_ddpir_demosaic.py 逐步一致：
  * 同一个 bayer_mask()
  * 同一段加噪（[-1,1] 域、std = 2σ、seed 0、加完再乘 mask）
  * util.calculate_psnr（RGB）与 lpips.LPIPS(net='vgg')，和主脚本同一套

⚠️ OpenCV 的 Bayer 命名和传感器命名是错位的，所以不靠记忆：先在第一张图上把
   四种 pattern code × 两种插值全试一遍打表，再用最好的那个跑全集，并把选中的 code 打出来。
   只有几何上正确的那个能到 30+ dB，其余会掉到 20 dB 上下并伴随颜色互换。
"""
import argparse
import glob
import os

import cv2
import numpy as np
import torch

from utils import utils_image as util
from utils.utils_inpaint import bayer_mask

CODES = {
    "bilinear/BG": cv2.COLOR_BayerBG2BGR, "bilinear/GB": cv2.COLOR_BayerGB2BGR,
    "bilinear/RG": cv2.COLOR_BayerRG2BGR, "bilinear/GR": cv2.COLOR_BayerGR2BGR,
    "EA/BG": cv2.COLOR_BayerBG2BGR_EA, "EA/GB": cv2.COLOR_BayerGB2BGR_EA,
    "EA/RG": cv2.COLOR_BayerRG2BGR_EA, "EA/GR": cv2.COLOR_BayerGR2BGR_EA,
}


def degrade(img_H, sigma):
    """与 main_ddpir_demosaic.py 的 img_L 构造逐行一致。"""
    mask = bayer_mask(*img_H.shape[:2])
    img_L = img_H * mask / 255.0
    np.random.seed(seed=0)
    img_L = img_L * 2 - 1
    img_L += np.random.normal(0, sigma * 2, img_L.shape)
    img_L = img_L / 2 + 0.5
    img_L = img_L * mask
    return img_L


def demosaic(img_L, code):
    bayer = np.clip(img_L.sum(axis=2) * 255.0, 0, 255).astype(np.uint8)  # 每像素只有一个通道非零
    bgr = cv2.cvtColor(bayer, code)
    return bgr[:, :, ::-1].copy()  # BGR -> RGB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", default="kodak24_c256")
    ap.add_argument("--sigma", type=float, default=0.0)
    ap.add_argument("--testsets", default="testsets")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.testsets, args.testset, "*.png")))
    if not paths:
        raise SystemExit("no images under %s/%s" % (args.testsets, args.testset))

    # --- 第一张图上定 pattern code ---
    img_H0 = util.imread_uint(paths[0], n_channels=3)
    L0 = degrade(img_H0, args.sigma)
    print("pattern check on %s:" % os.path.basename(paths[0]))
    scored = []
    for name, code in CODES.items():
        psnr = util.calculate_psnr(demosaic(L0, code), img_H0, border=0)
        scored.append((psnr, name, code))
        print("    %-12s PSNR %7.4f" % (name, psnr))
    scored.sort(reverse=True)
    best_psnr, best_name, best_code = scored[0]
    print("  -> using %s (%.4f dB on image 1)" % (best_name, best_psnr))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import lpips
    loss_fn_vgg = lpips.LPIPS(net="vgg").to(device)

    psnrs, lps = [], []
    for p in paths:
        img_H = util.imread_uint(p, n_channels=3)
        img_E = demosaic(degrade(img_H, args.sigma), best_code)
        psnrs.append(util.calculate_psnr(img_E, img_H, border=0))

        t = lambda a: torch.from_numpy(np.transpose(a, (2, 0, 1)))[None].to(device) / 255.0 * 2 - 1
        lps.append(float(loss_fn_vgg(t(img_E), t(img_H)).detach().cpu().numpy().ravel()[0]))

    print("CLASSICAL %s | %s | sigma %.3f | n=%d | PSNR %.4f | LPIPS %.4f"
          % (best_name, args.testset, args.sigma, len(paths),
             sum(psnrs) / len(psnrs), sum(lps) / len(lps)))


if __name__ == "__main__":
    main()
