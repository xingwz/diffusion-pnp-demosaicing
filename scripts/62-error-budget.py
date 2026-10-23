#!/usr/bin/env python3
"""Split a run's squared error by the Bayer mask: sampled third vs reconstructed two thirds.

The numbers in README §3.4.4 come from this script. It answers two questions the
average PSNR cannot: how much of the measurement noise a method actually removed
at the pixels it was given, and where the remaining error lives.

    python3 scripts/62-error-budget.py --gt-dir testsets/kodak24_c256 \
        --cfa results/classical_kodak24_c256_sigma0.05 \
        "DiffPIR=results/kodak24_c256_..._lambda3.0_dcnone=_256x256_diffusion_uncond.png" \
        "DRUNet=results/kodak24_c256_..._lambda0.003_x8False_dcnone=_drunet_color.png"

Each positional argument is label=directory=suffix. --cfa points at the directory
holding <stem>_cfa.png (50-classical-baseline.py --save-dir writes them) and adds
a reference row: the noise that actually entered, after clipping and quantisation.

numpy + Pillow only. Reads the saved uint8 PNGs, so the per-image PSNR it prints
matches what the runs logged.
"""
import argparse
import glob
import os

import numpy as np


def bayer_mask(h, w, pattern="RGGB"):
    """Same mask as utils/utils_inpaint.bayer_mask."""
    m = np.zeros((h, w, 3), np.float32)
    idx = {"R": 0, "G": 1, "B": 2}
    p = [[pattern[0], pattern[1]], [pattern[2], pattern[3]]]
    for i in range(2):
        for j in range(2):
            m[i::2, j::2, idx[p[i][j]]] = 1.0
    return m


def load(path):
    from PIL import Image
    return np.array(Image.open(path).convert("RGB"), dtype=np.float64)


def psnr_from_mse(mse):
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def budget(gt_paths, directory, suffix):
    se_m = se_u = n_m = n_u = 0.0
    psnrs = []
    for g in gt_paths:
        stem = os.path.splitext(os.path.basename(g))[0]
        gt = load(g)
        est = load(os.path.join(directory, stem + suffix))
        m = bayer_mask(*gt.shape[:2]).astype(bool)
        sq = (gt - est) ** 2
        se_m += sq[m].sum(); n_m += m.sum()
        se_u += sq[~m].sum(); n_u += (~m).sum()
        psnrs.append(psnr_from_mse(sq.mean()))
    return se_m / n_m, se_u / n_u, se_m / (se_m + se_u), float(np.mean(psnrs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--cfa", default=None, help="directory with <stem>_cfa.png, for the input-noise row")
    ap.add_argument("runs", nargs="+", metavar="label=dir=suffix")
    args = ap.parse_args()

    gt_paths = sorted(glob.glob(os.path.join(args.gt_dir, "*.png")))
    if not gt_paths:
        raise SystemExit("no ground-truth PNGs under " + args.gt_dir)
    print("%d images from %s\n" % (len(gt_paths), args.gt_dir))

    head = "%-22s %9s %16s %16s %10s" % ("", "avg PSNR", "sampled 1/3", "other 2/3", "SE on 1/3")
    print(head)
    print("-" * len(head))

    noise_mse = None
    if args.cfa:
        mse_m, _, _, _ = budget(gt_paths, args.cfa, "_cfa.png")
        noise_mse = mse_m
        print("%-22s %9s %7.2f %7.2f dB %16s %10s"
              % ("measurement y", "-", mse_m, psnr_from_mse(mse_m), "-", "-"))

    for spec in args.runs:
        label, directory, suffix = spec.rsplit("=", 2)   # rsplit: labels may contain "="
        mse_m, mse_u, share, avg_psnr = budget(gt_paths, directory, suffix)
        print("%-22s %9.4f %7.2f %7.2f dB %7.2f %7.2f dB %9.1f%%"
              % (label, avg_psnr, mse_m, psnr_from_mse(mse_m), mse_u, psnr_from_mse(mse_u), 100 * share))
        if noise_mse:
            print("%-22s %s removed %.2f dB (%.0f%%) of the noise energy on the sampled pixels"
                  % ("", " " * 9, 10 * np.log10(noise_mse / mse_m), 100 * (1 - mse_m / noise_mse)))


if __name__ == "__main__":
    main()
