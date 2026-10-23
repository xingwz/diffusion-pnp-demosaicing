#!/usr/bin/env python3
"""Assemble the side-by-side visual comparison for §3 of the README.

One call = one figure = one image. Top row is the full 256x256 crop per method,
bottom row is a zoom of the same box, nearest-neighbour upscaled so the pixels
stay honest. PSNR (RGB, border=0) is printed over every panel that has a
ground-truth reference.

    python3 scripts/61-make-sidebyside.py \
        --gt-dir testsets/kodak24_c256 --image kodim19.png \
        --cfa       results/classical_kodak24_c256_sigma0.05 \
        --classical results/classical_kodak24_c256_sigma0.05 \
        --drunet    results/kodak24_c256_demosaic_DiffPIR_bayer_RGGB_drunet_color_sigma0.05_NFE40_lambda0.003_x8False_dcnone \
        --diffpir   results/kodak24_c256_demosaic_DiffPIR_bayer_RGGB_256x256_diffusion_uncond_sigma0.05_NFE100_eta0.0_zeta1.0_lambda3.0 \
        --zoom 96,112,64 --title 'Kodak24 crop, sigma_n = 12.75/255' \
        --out figures/sidebyside-kodak24-sigma005.png

Every --cfa/--classical/--drunet/--diffpir argument is either a file or a
directory; a directory is searched for a PNG whose name starts with the image
stem (that is how both main_ddpir_demosaic*.py and 50-classical-baseline.py
--save-dir name their output). Panels left out are simply dropped, so this also
works before all the runs are in.

Needs numpy, matplotlib and Pillow only -- no torch, no cv2. The CFA and
classical panels come from 50-classical-baseline.py --save-dir, i.e. from the
same code path that produced the numbers in the README, instead of being
re-derived here.
"""
import argparse
import glob
import os

import numpy as np

PANELS = [
    ("gt", "Ground truth", False),
    ("cfa", "CFA input (mosaicked + noise)", False),
    ("classical", "Classical CFA interpolation", True),
    ("drunet", "DRUNet PnP-HQS", True),
    ("diffpir", "DiffPIR (diffusion prior)", True),
]


def load(path):
    from PIL import Image
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def resolve(arg, stem, suffix=None):
    """A file stays as it is; a directory is searched for <stem>*.png."""
    if arg is None:
        return None
    if os.path.isfile(arg):
        return arg
    pattern = os.path.join(arg, stem + ("*" + suffix if suffix else "*") + ".png")
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise SystemExit("no match for %s" % pattern)
    if len(hits) > 1:
        print("  ! %d matches for %s, taking %s" % (len(hits), pattern, os.path.basename(hits[0])))
    return hits[0]


def psnr(a, b):
    """Identical to utils/utils_image.calculate_psnr for uint8 RGB, border=0."""
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--image", required=True, help="file name inside --gt-dir, e.g. kodim19.png")
    ap.add_argument("--cfa")
    ap.add_argument("--classical")
    ap.add_argument("--drunet")
    ap.add_argument("--diffpir")
    ap.add_argument("--zoom", default=None, help="x,y,size of the inset box (default: centre, 64 px)")
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dpi", type=int, default=160)
    args = ap.parse_args()

    stem = os.path.splitext(os.path.basename(args.image))[0]
    sources = {
        "gt": os.path.join(args.gt_dir, args.image),
        # 50-classical-baseline.py --save-dir writes <stem>_cfa.png and <stem>_classical.png
        "cfa": resolve(args.cfa, stem, "_cfa"),
        "classical": resolve(args.classical, stem, "_classical"),
        "drunet": resolve(args.drunet, stem),
        "diffpir": resolve(args.diffpir, stem),
    }

    imgs, labels, scored = [], [], []
    gt = load(sources["gt"])
    h, w = gt.shape[:2]
    for key, label, has_ref in PANELS:
        if sources.get(key) is None:
            print("  - %-10s skipped" % key)
            continue
        im = load(sources[key])
        if im.shape[:2] != (h, w):
            raise SystemExit("%s is %s, ground truth is %s" % (sources[key], im.shape[:2], (h, w)))
        print("  + %-10s %s" % (key, os.path.relpath(sources[key])))
        imgs.append(im)
        labels.append("%s\n%.2f dB" % (label, psnr(im, gt)) if has_ref else label)
        scored.append(key)

    if args.zoom:
        zx, zy, zs = (int(v) for v in args.zoom.split(","))
    else:
        zs = 64
        zx, zy = (w - zs) // 2, (h - zs) // 2
    if zx < 0 or zy < 0 or zx + zs > w or zy + zs > h:
        raise SystemExit("zoom box %s falls outside the %dx%d image" % ((zx, zy, zs), w, h))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    n = len(imgs)
    fig, axes = plt.subplots(2, n, figsize=(2.1 * n, 4.6), dpi=args.dpi,
                             gridspec_kw={"height_ratios": [1, 1], "hspace": 0.04, "wspace": 0.03})
    axes = np.atleast_2d(axes)
    for col, (im, label) in enumerate(zip(imgs, labels)):
        top, bottom = axes[0, col], axes[1, col]
        top.imshow(im, interpolation="nearest")
        top.add_patch(Rectangle((zx, zy), zs, zs, fill=False, color="#e8413a", lw=1.2))
        top.set_title(label, fontsize=8, pad=4)
        bottom.imshow(im[zy:zy + zs, zx:zx + zs], interpolation="nearest")
        for ax in (top, bottom):
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ax.spines.values():
                side.set_linewidth(0.4)
        for side in bottom.spines.values():
            side.set_color("#e8413a")
            side.set_linewidth(1.2)
    axes[0, 0].set_ylabel("full crop", fontsize=8)
    axes[1, 0].set_ylabel("%d$\\times$%d zoom" % (zs, zs), fontsize=8)
    if args.title:
        fig.suptitle(args.title, fontsize=10, y=0.975)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", facecolor="white")
    print("wrote %s  (%d panels: %s)" % (args.out, n, ", ".join(scored)))


if __name__ == "__main__":
    main()
