# Diffusion Priors for Plug-and-Play Image Restoration — Reproduction + a Bayer Demosaicing Extension

A reproduction of **DiffPIR** (Zhu et al., *Denoising Diffusion Models for Plug-and-Play Image Restoration*, CVPR Workshops / NTIRE 2023) on super-resolution, Gaussian deblurring and inpainting, plus one task the paper does not cover: **Bayer CFA demosaicing under additive noise**, formulated as channel-wise masked restoration, with its data-fidelity weight swept and its optimum explained.

This is a **reproduction and extension, not a new method.** No claim of state of the art. The diffusion priors are pretrained and used at inference time only; nothing here is trained from scratch.

**Why this project:** my PhD work is plug-and-play restoration — a denoiser used as an implicit prior inside an iterative solver (CVPR 2021, EUSIPCO 2025). DiffPIR keeps that skeleton and swaps the discriminative prior for a generative one, so it is the shortest honest path from what I already do to diffusion-based restoration.

---

## 1. What is in here

| | |
|---|---|
| **Step 1 — reproduction** | DiffPIR on SR ×4, Gaussian deblurring, random inpainting; paper's 100-image FFHQ validation split; PSNR / LPIPS vs published values → [§2](#2-step-1--reproduction-results) |
| **Step 2 — extension** | Bayer CFA demosaicing + denoising via a channel-wise mask in the same closed-form data step; λ / ζ swept (36 runs), optimum explained, quality-vs-NFE curve → [§3](#3-step-2--bayer-demosaicing-extension) |
| **Reproduction notes** | One implementation/paper discrepancy worth 2.7 dB, and two configuration traps → [§2.3](#23-one-discrepancy-between-the-released-code-and-the-paper), [§2.4](#24-two-configuration-traps) |
| **How to run** | Environment, checkpoints, commands, data preparation → [§4](#4-how-to-run) |

**This repository is a patch set, not a fork.** It carries only what is new or modified; clone [upstream DiffPIR](https://github.com/yuanzhi-zhu/DiffPIR) and drop these in.

| Path | |
|---|---|
| `main_ddpir_demosaic.py` | the demosaicing entry point — a copy of `main_ddpir_inpainting.py` differing in its mask and import |
| `utils_inpaint.py` | → `utils/utils_inpaint.py`, adds `bayer_mask()` |
| `make_crops.py` | builds the 256×256 even-origin crops for Kodak24 / McMaster |
| `data/*/crops.csv` | the exact crop manifests (source file + box) for all 42 crops |
| `scripts/` | the SLURM/cluster scripts the runs were produced with, plus `README-cluster.md` |

---

## 2. Step 1 — reproduction results

### 2.1 Headline numbers

Evaluated on **`ffhq_val_100`** — the 100-image 256×256 FFHQ hold-out split used in the paper — with the paper's own Table 3 hyperparameters (λ, ζ). Prior: `diffusion_ffhq_10m`. Metrics: PSNR (dB) and LPIPS (AlexNet).

| Task | NFE | σ<sub>n</sub> | λ / ζ | **Mine** PSNR / LPIPS | **Paper** PSNR / LPIPS | ΔPSNR | ΔLPIPS |
|---|---|---|---|---|---|---|---|
| Deblur (Gaussian, 61×61, std 3.0) | 100 | 0.05 | 7.0 / 0.3 | **27.26** / **0.2242** | 27.36 / 0.236 | **−0.10** | −0.012 |
| SR ×4 (bicubic) † | 100 | 0.05 | 8.0 / 0.2 | **26.24** / **0.2552** | 26.64 / 0.260 | **−0.40** | −0.005 |
| Inpaint (random mask) | 20 | 0.0 | 7.0 / 1.0 | **34.06** / **0.1118** | 34.03 / 0.116 | **+0.03** | −0.004 |

**PSNR is within 0.4 dB of the published value on all three tasks; LPIPS is at or below the published value on all three.**

Paper references: Table 1 (σ<sub>n</sub> = 0.05, 100 NFE) for deblurring and SR, Table 2 (σ<sub>n</sub> = 0) for inpainting, Table 3 for λ / ζ.

> † The SR row is a run at exactly the paper's ζ = 0.2 (26.2397 / 0.2552), confirmed against the full-reproduction log. On the 5-image subset ζ turned out to be insensitive for this task (ζ = 0.25 → 25.24 / 0.2645 vs ζ = 0.2 → 25.11 / 0.2653), which is why the sweep logs and the final run agree.

Approximate single-image cost on one GH200 (100 NFE, 256×256): ~5 s deblur, ~4 s SR, ~1.4 s inpaint at 20 NFE.

### 2.2 How the remaining gap was accounted for

The interesting part of this reproduction was not the final table, it was ruling out the alternative explanations for an earlier mismatch.

An intermediate stage evaluated on the 5 images shipped in the repo's `demo_test/`:

| Task | Mine (5 images) | Paper (100 images) | ΔPSNR |
|---|---|---|---|
| Deblur (Gaussian) | 25.91 / 0.2444 | 27.36 / 0.236 | −1.45 |
| SR ×4 | 25.11 / 0.2653 | 26.64 / 0.260 | −1.53 |
| Inpaint (random) | 32.70 / 0.1212 | 34.03 / 0.116 | −1.33 |

ΔPSNR spans only **0.2 dB across the three tasks**, even though their degradation operators and closed-form data sub-problems share nothing (Gaussian convolution / bicubic downsampling / binary mask). An implementation error would not be that uniform. A **uniform offset points at the evaluation set**, not at the code — so the hypothesis was: the gap is `demo_test`'s 5 images vs the paper's 100-image hold-out split.

Switching to `ffhq_val_100` removed the offset entirely (table in §2.1). Hypothesis → designed control → competing explanations excluded.

Residuals (LPIPS consistently ~0.005 lower, SR 0.4 dB lower) are attributed to operator-level numerical differences between torch 2.13 and the paper's 1.13, and to random seeding. Not pursued further.

### 2.3 One discrepancy between the released code and the paper

`main_ddpir.py:61` in the released implementation:

```python
kernel_std_i = self.config.kernel_std * np.abs(np.random.rand() * 2 + 1)
```

The base `kernel_std = 3.0` matches the paper, but it is multiplied by a random factor in [1, 3) — so the effective Gaussian blur std is drawn per image from [3.0, 9.0), mean 6.0, i.e. **twice the paper's fixed value**. Section 4.1 of the paper specifies *"a blur kernel of size 61×61 and a standard deviation of 3.0"*.

Fix — drop the random factor:

```python
kernel_std_i = self.config.kernel_std
```

Effect on Gaussian deblurring: PSNR **23.22 → 25.91** (+2.69 dB) on `demo_test`, LPIPS 0.3015 → 0.2444; and 27.26 on the full 100-image split. Without this fix, running the full split still lands around 23 dB with no obvious place to look.

This is a common situation in reproduction work and not a criticism of the authors — released research code drifts from the manuscript. Recording it because locating it is the part that took the work.

### 2.4 Two configuration traps

1. **`lambda_` in `configs/sisr.yaml` is a multiplier, not λ.** The script sweeps `lambda_ × [2, 3, …, 12]`. So `lambda_: 1.0` sweeps λ = 2…12 (which *includes* the paper's 8), while `lambda_: 8.0` sweeps λ = 16…96 — all past the paper's value, and the results get worse (λ = 16 gives 24.57 / 0.2743). To reproduce λ = 8, keep `lambda_: 1.0` and read the `lambda:8.0` group out of the log.
2. **The last line of the SR log is not the result.** `Average PSNR 25.2017 / LPIPS 0.2772` at the end of a sweep is the mean over all 11 λ groups. Take the λ = 8 group.

### 2.5 Two experiment settings to check first when numbers do not match

- The paper evaluates **100 hold-out validation images** per dataset, not the 5 in `demo_test`. A single image deviating several dB from the mean is normal.
- The Gaussian kernel is **61×61, std 3.0**; the motion kernel is 61×61, intensity 0.5, and all compared methods share the same kernel instance.

---

## 3. Step 2 — Bayer demosaicing extension

> **Scope.** Everything below is this project's own result: the paper does not cover demosaicing, so there is no published λ / ζ to copy and no published number to match. What is *not* here is a cross-method comparison — see §3.3.

### 3.1 Formulation

Bayer CFA demosaicing is masked restoration with a **channel-wise** mask: each pixel retains exactly one of its three colour channels. That makes the measurement operator a binary diagonal, so DiffPIR's inpainting data sub-problem — which already has a closed-form solution for a diagonal operator (`sub_1_analytic: true`) — applies unchanged. The extension is a mask, not a new solver.

`bayer_mask()` produces an RGGB pattern: sampling rates R/G/B = 0.25 / 0.5 / 0.25, exactly one channel active per pixel, top-left 2×2 = `[[R, G], [G, B]]` (verified numerically).

### 3.2 Results

Prior: `256x256_diffusion_uncond` (the general-image model, not the FFHQ one). Degradation: RGGB mosaic plus AWGN at σ<sub>n</sub> = 12.75/255 ≈ 0.05, added in the [-1, 1] domain and then re-masked, i.e. noise only on sampled pixels. Data: 256×256 centre crops with even-numbered origins (§4.3). Metrics: PSNR (RGB) and LPIPS (AlexNet); **SSIM is not reported — it was not computed in these runs.**

**Operating point: λ = 3, ζ = 1.0, 100 NFE.**

| Dataset | Images | PSNR | LPIPS |
|---|---|---|---|
| Kodak24 (crops) | 24 | **30.6704** | **0.1842** |
| McMaster (crops) | 18 | **31.2000** | **0.1743** |

The operating point was tuned on Kodak24 only and transferred to McMaster unchanged, where both metrics improve — so λ\* = 3 is not a Kodak-specific fit. (The absolute gap between the two sets reflects their content: these McMaster crops carry less high-frequency colour structure.)

#### 3.2.1 Where the optimum sits, and why it is not the paper's

Sweeping the data-fidelity weight λ at ζ = 1.0 gives a single-peaked curve with **λ\* = 3**, where PSNR and LPIPS are optimal simultaneously:

| λ | 1 | 2 | **3** | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PSNR | 29.2581 | 30.3022 | **30.6704** | 30.4815 | 30.1720 | 29.8508 | 29.5552 | 29.3040 | 29.0977 | 28.8538 | 28.6718 | 28.5057 |
| LPIPS | 0.2181 | 0.1873 | **0.1842** | 0.2117 | 0.2384 | 0.2587 | 0.2751 | 0.2893 | 0.2998 | 0.3113 | 0.3186 | 0.3252 |

The paper's own σ<sub>n</sub> = 0.05 tasks use λ in **7–9** (deblurring 7–8, SR 8–9). Demosaicing lands much lower, and the mechanism is in the data step itself: with ρ = λσ²/σ<sub>k</sub>² and the closed form `(mask·y + ρ·x₀)/(mask + ρ)`, a larger λ trusts the prior over the measurement. The CFA operator is **pointwise** — every retained pixel carries its own independent noise, with no neighbourhood averaging. Deblurring and super-resolution operators average, so their effective measurement noise is lower and they can afford to lean on the prior harder. A mask gives no such cushion, so λ must be smaller.

#### 3.2.2 ζ: the boundary value is the right one

ζ ∈ {0.3, 0.7, 1.0} × λ ∈ {1…12}, 36 runs, three complete curves:

| ζ | λ\* | best PSNR | best LPIPS |
|---|---|---|---|
| **1.0** | **3** | **30.6704** | **0.1842** |
| 0.7 | ≈6–7 | 28.4813 | 0.2300 |
| 0.3 | > 12 (no peak in range) | 25.8195 | 0.3141 |

Two things follow. **ζ\* = 1.0 is at the boundary**, matching what the paper uses for inpainting — masked degradations behave as one family, and demosaicing belongs to it. And **λ\* is coupled to ζ**: lowering ζ moves the optimum right and pushes the whole curve down, so "λ\* = 3" is meaningless quoted on its own. The two must be reported as a pair.

### 3.3 Baselines: not run, and the table is omitted on purpose

The comparison that would make this section complete is a **DPIR-style PnP baseline** (DRUNet denoiser, identical closed-form data step, same crops) plus **classical CFA interpolation**, which together isolate the prior as the only variable. Neither has been run yet, and a cross-method table built from numbers taken under different protocols is worse than no table, so none is given here. The same applies to the side-by-side figure (ground truth / CFA input / classical / PnP-CNN / DiffPIR).

What *is* reported above is self-contained: it does not depend on any baseline.

### 3.4 Quality vs NFE

Kodak24 crops, **operating point held fixed** at λ = 3 / ζ = 1.0:

| NFE | PSNR | LPIPS |
|---|---|---|
| 20 | 27.3189 | 0.2709 |
| 50 | 29.4979 | 0.2070 |
| 100 | **30.6704** | **0.1842** |

Halving the budget from 100 to 50 costs **1.17 dB**; going to 20 costs **3.35 dB** against the full budget.

⚠️ These are *the fixed operating point evaluated at lower NFE*, not a per-NFE re-optimised λ — the optimum was swept at 100 NFE. A per-NFE sweep would likely recover part of the gap and is not claimed here.

This is the practical point of the whole extension. A discriminative denoiser prior converges in on the order of 10–20 iterations; a diffusion prior is charged roughly one network evaluation per step and defaults to 100. On-device deployment is exactly where that difference is paid.

---

## 4. How to run

### 4.1 Environment

Developed on CSC's **Roihu**, one **GH200** (Grace Hopper, sm_90, 96 GB HBM3). Note the compute nodes are **aarch64**, not x86_64 — many PyPI packages have no prebuilt ARM wheel.

```bash
module purge
module load python-pytorch/2.13          # torch 2.13.0+cu130, CUDA 13.0, NVIDIA GH200
export PYTHONUSERBASE=/path/to/your/env
pip install --user --only-binary=:all: --no-deps lpips
```

- `--only-binary=:all:` makes pip fail fast instead of falling back to a source build on ARM.
- `--no-deps` on `lpips` is **required**: its dependency list includes torch/torchvision, and without it pip installs a second PyPI torch into `PYTHONUSERBASE` that shadows the CUDA-matched module build. The symptom is `torch.cuda.is_available()` silently becoming `False`.
- Sanity check: `torch.cuda.get_arch_list()` must contain `sm_90`.

Memory is a non-issue at 256×256; 96 GB is far more than this needs.

### 4.2 Checkpoints

```
model_zoo/256x256_diffusion_uncond.pt   # OpenAI guided-diffusion, ImageNet 256×256 (general images)
model_zoo/diffusion_ffhq_10m.pt         # FFHQ faces — used for all Step 1 numbers above
```

⚠️ The upstream `download.sh` does not check whether a file already exists and will re-download over it. An interrupted re-download truncated a complete 2.1 GB checkpoint to 190 MB. Run `ls -lh model_zoo/` first.

### 4.3 Data

Step 1 uses **`ffhq_val_100`** from HuggingFace `danaroth/benchmark_image_zhang` (the benchmark repository maintained by DiffPIR's first author) — 100 × 256×256 PNG. Those files are byte-identical to the 5 images in the repo's `demo_test/`, which confirms the split is the one used for the published evaluation.

The same repository has `imagenet_val_100`, but those are JPEGs at original resolution — **they need cropping to 256×256 before use** (`make_crops.py`).

Step 2 uses Kodak24 and McMaster, centre-cropped to 256×256 by `make_crops.py`.

### 4.4 Commands

```bash
python main_ddpir_deblur.py        # Gaussian / motion deblurring
python main_ddpir_sisr.py          # ×4 super-resolution
python main_ddpir_inpainting.py    # random / box inpainting
python main_ddpir_demosaic.py      # Bayer CFA demosaicing (this work)
```

`main_ddpir_demosaic.py` takes its six run parameters from the environment, so a sweep is a loop over submissions rather than an edited file (`scripts/31-patch-env.py` applies the same change to a stock checkout):

```bash
DP_TESTSET=kodak24_c256 DP_MODEL=256x256_diffusion_uncond \
DP_SIGMA=0.05 DP_NFE=100 DP_LAMBDA=3 DP_ZETA=1.0 \
python main_ddpir_demosaic.py        # the §3.2 operating point
```

Key `configs/sisr.yaml` values for the reproduction (defaults, intentionally unchanged):

| Parameter | Value | Meaning |
|---|---|---|
| `iter_num` | 100 | NFE; the paper's main tables are ≤100 |
| `lambda_` / `zeta` | per task, see §2.1 | the two key hyperparameters — and see the trap in §2.4 |
| `noise_level_img` | 12.75 | = 0.05 × 255 additive Gaussian noise |
| `model_name` | `diffusion_ffhq_10m` | FFHQ prior; use `256x256_diffusion_uncond` for general images |
| `sf` / `sr_mode` | 4 / `blur` | ×4 SR |
| `calc_LPIPS` | `true` | report LPIPS alongside PSNR |
| `sub_1_analytic` | `true` | closed-form data sub-problem — what makes §3.1 cheap |

### 4.5 Known breakages running 2023 code on torch 2.13

| Error | Cause | Fix |
|---|---|---|
| `UnpicklingError: Weights only load failed` | torch ≥2.6 defaults `torch.load(weights_only=True)` | pass `weights_only=False` at the checkpoint load |
| `AttributeError: module 'numpy' has no attribute 'float'` | numpy 2.x removed the `np.float` / `np.int` / `np.bool` aliases | use the builtin `float` / `int` / `bool` |
| `ModuleNotFoundError: mpi4py` | imported by `guided_diffusion/dist_util.py` | not needed for pure inference; comment out the import |

---

## 5. Attribution and scope

Built on **DiffPIR** by Yuanzhi Zhu, Kai Zhang, Jingyun Liang, Jiezhang Cao, Bihan Wen, Radu Timofte and Luc Van Gool (MIT licence) — upstream: https://github.com/yuanzhi-zhu/DiffPIR. Diffusion priors from OpenAI's guided-diffusion (ImageNet) and the FFHQ model distributed by the DiffPIR authors. `lpips` by Richard Zhang et al.

```bibtex
@inproceedings{zhu2023denoising,
  title     = {Denoising Diffusion Models for Plug-and-Play Image Restoration},
  author    = {Zhu, Yuanzhi and Zhang, Kai and Liang, Jingyun and Cao, Jiezhang
               and Wen, Bihan and Timofte, Radu and Van Gool, Luc},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (NTIRE)},
  year      = {2023}
}
```

**Scope, stated plainly:**
- This is a reproduction plus one added task. It is **not a new method and not a claim of state of the art**.
- The diffusion models are **pretrained and used at inference time**. I did not train a diffusion model.
- Step 1 reproduces published numbers; Step 2 is my own extension, reported on its own terms. **No cross-method baseline has been run yet**, so no comparison table is given (§3.3).

Modifications to upstream files are confined to: the kernel-std fix (§2.3), `bayer_mask()` in `utils_inpaint.py`, `main_ddpir_demosaic.py` (a copy of `main_ddpir_inpainting.py` differing only in its import and mask), and the torch-2.13 compatibility edits in §4.5.

Author: Wenzhu Xing.
