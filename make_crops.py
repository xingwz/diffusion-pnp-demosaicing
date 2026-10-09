#!/usr/bin/env python3
"""Build 256x256 center crops of Kodak (set24) and McMaster (set18) for
DiffPIR-based demosaicing experiments.

Why center crops: the pretrained guided-diffusion priors are fixed at 256x256,
so Kodak (768x512) and McM (500x500) cannot be fed directly.

Why even crop origins: the Bayer phase is defined modulo 2. An odd x0/y0 would
silently rotate RGGB into GRBG and every method would be evaluated on a
different CFA than the one bayer_mask() assumes. Origins are forced even.

Every method in the comparison table MUST read these same files.
Writes crops.csv as the manifest (source file -> crop box), keep it in the repo.

Usage:
    python make_crops.py --src <dir with originals> --out <dir> [--size 256]
"""
import argparse, csv, os
from PIL import Image

def center_box(w, h, s):
    x0 = (w - s) // 2
    y0 = (h - s) // 2
    x0 -= x0 % 2            # keep the CFA phase
    y0 -= y0 % 2
    return x0, y0, x0 + s, y0 + s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--size', type=int, default=256)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    names = sorted(n for n in os.listdir(a.src)
                   if n.lower().endswith(('.png', '.tif', '.tiff', '.bmp', '.jpg')))
    rows = []
    for n in names:
        im = Image.open(os.path.join(a.src, n)).convert('RGB')
        w, h = im.size
        if w < a.size or h < a.size:
            print(f'  skip {n}: {w}x{h} smaller than {a.size}')
            continue
        box = center_box(w, h, a.size)
        out_name = os.path.splitext(n)[0] + '.png'
        im.crop(box).save(os.path.join(a.out, out_name))
        rows.append([out_name, n, w, h, *box])

    with open(os.path.join(a.out, 'crops.csv'), 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['crop', 'source', 'src_w', 'src_h', 'x0', 'y0', 'x1', 'y1'])
        wr.writerows(rows)
    print(f'{len(rows)} crops -> {a.out}')

if __name__ == '__main__':
    main()
