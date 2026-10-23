import os.path
import cv2
import logging

import numpy as np
import torch
from datetime import datetime
from collections import OrderedDict

from utils import utils_model
from utils import utils_mosaic
from utils import utils_logger
from utils import utils_image as util
from utils.utils_inpaint import mask_generator, bayer_mask

# ----------------------------------------------------------------------------
# DRUNet PnP-HQS baseline for Bayer demosaicing.
#
# Derived from main_ddpir_demosaic.py.  The degradation (Bayer mask), the data
# sub-problem, the test set and the metrics are IDENTICAL to the DiffPIR run --
# the only thing that changes is the prior: the diffusion model is replaced by
# DRUNet, and the DDIM/DDPM machinery by DPIR's HQS schedule.
#
# Reference: main_dpir_demosaick.py in cszn/DPIR (iter_num=40, sigma 49 -> 0.6,
# rho_k = 0.23 * sigma_y^2 / sigma_k^2).
# ----------------------------------------------------------------------------


def pack_cfa4(y, mask):
    """Pack a masked RGB measurement into the Nx4x(H/2)x(W/2) layout that
    utils_mosaic.dm_matlab() expects.

    Same packing as utils_mosaic.mosaic_CFA_Bayer():
        ch0 = [0::2, 0::2]  ch1 = [0::2, 1::2]  ch2 = [1::2, 0::2]  ch3 = [1::2, 1::2]
    Unlike that function this one starts from the (possibly noisy) measurement y
    rather than from the clean image, so the initialiser only ever sees data.

    !! RGGB only -- dm_matlab's fill rules assume ch0=R, ch1/ch2=G, ch3=B.
    """
    cfa = (y * mask).sum(1, keepdim=True)       # (1,1,H,W)
    return torch.cat((cfa[..., 0::2, 0::2], cfa[..., 0::2, 1::2],
                      cfa[..., 1::2, 0::2], cfa[..., 1::2, 1::2]), dim=1)


def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------

    noise_level_img         = float(os.environ.get('DP_SIGMA', '0'))   # [0,1] scale, 12.75/255 -> 0.05
    noise_level_model       = noise_level_img   # set noise level of model, default: 0
    model_name              = 'drunet_color'    # DRUNet colour denoiser (DPIR's model_zoo)
    testset_name            = os.environ.get('DP_TESTSET', 'demo_test')   # demo_test | kodak24_c256 | mcm18_c256
    iter_num                = int(os.environ.get('DP_NFE', '40'))      # DPIR default for demosaicing (24 is SISR, 8 is deblur)

    # --- HQS schedule (main_dpir_demosaick.py:61-63) ---
    modelSigma1             = 49                # sigma_1: start of the denoiser-sigma schedule, in [0,255]
    modelSigma2             = max(0.6, noise_level_model*255.)   # sigma_2: end of the schedule
    lambda_                 = float(os.environ.get('DP_LAMBDA', '0.23'))   # rho_k = lambda_*sigma_y^2/sigma_k^2; 0.23 == DPIR
    dc_mode                 = os.environ.get('DP_DC', 'loop')          # data-consistency projection x[mask]=y[mask]:
                                                                        #   loop  = every iteration (DPIR's own demosaick script)
                                                                        #   final = once at the end  (DiffPIR's "recover conditional part")
                                                                        #   none  = never            (the only correct one for sigma>0 --
                                                                        #           y is noisy there, and the closed-form step already
                                                                        #           handles it through rho)
    assert dc_mode in ('loop', 'final', 'none'), dc_mode
    x8                      = os.environ.get('DP_X8', '0') == '1'      # geometric self-ensemble; DPIR uses True, but DiffPIR
                                                # does not self-ensemble -- keep False for a fair
                                                # prior-vs-prior comparison (True costs 8x nothing extra
                                                # here, it reuses the same 40 calls, but it is an extra
                                                # trick DiffPIR does not get)

    mask_name               = 'gt_keep_masks/face/000000.png'   # mask path for loading mask img
    load_mask               = False
    mask_type               = 'bayer_RGGB'      # CFA pattern, same as the DiffPIR run
    assert mask_type == 'bayer_RGGB', 'dm_matlab / pack_cfa4 assume an RGGB CFA'

    show_img                = False             # default: False
    save_L                  = False             # save LR image
    save_E                  = True              # save estimated image
    save_LEH                = False             # save zoomed LR, E and H images
    save_progressive        = False             # save the iterates

    sigma                   = max(0.255/255., noise_level_img)  # noise level associated with condition y

    log_process             = False
    task_current            = 'demosaic'        # channel-wise CFA mask = inpainting
    generate_mode           = 'PnP-HQS'         # was DiffPIR; now DPIR-style half quadratic splitting
    n_channels              = 3                 # fixed
    cwd                     = ''
    model_zoo               = os.path.join(cwd, 'model_zoo')    # fixed
    testsets                = os.path.join(cwd, 'testsets')     # fixed
    results                 = os.path.join(cwd, 'results')      # fixed
    result_name             = f'{testset_name}_{task_current}_{generate_mode}_{mask_type}_{model_name}_sigma{noise_level_img}_NFE{iter_num}_lambda{lambda_}_x8{x8}_dc{dc_mode}'
    model_path              = os.path.join(model_zoo, model_name+'.pth')
    device                  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    calc_LPIPS              = True

    # ----------------------------------------
    # L_path, E_path, H_path, mask_path
    # ----------------------------------------

    L_path                  = os.path.join(testsets, testset_name)      # L_path, for Low-quality images
    E_path                  = os.path.join(results, result_name)        # E_path, for Estimated images
    mask_path               = os.path.join(testsets, mask_name)         # mask_path, for mask images
    util.mkdir(E_path)

    logger_name             = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
    logger                  = logging.getLogger(logger_name)

    # ----------------------------------------
    # load model
    # ----------------------------------------

    from models.network_unet import UNetRes as net
    model = net(in_nc=n_channels+1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for _, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    logger.info('model_name:{}, mask_type:{}, image sigma:{:.3f}, model sigma:{:.3f}'.format(model_name, mask_type, noise_level_img, noise_level_model))
    logger.info('generate_mode:{}, iter_num:{}, modelSigma1:{:.2f}, modelSigma2:{:.2f}, lambda:{:.3f}, x8:{}, dc:{}'.format(generate_mode, iter_num, modelSigma1, modelSigma2, lambda_, x8, dc_mode))
    if noise_level_img > 0 and dc_mode != 'none':
        logger.info('!! WARNING: sigma>0 with dc={} -- the noisy measurements are written back into 1/3 of the output, '
                    'which caps PSNR regardless of the prior. Use DP_DC=none for sigma>0.'.format(dc_mode))
    logger.info('Model path: {:s}'.format(model_path))
    logger.info(L_path)
    L_paths = util.get_image_paths(L_path)

    if calc_LPIPS:
        import lpips
        loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

    def test_rho(lambda_=lambda_):
        logger.info('lambda:{:.3f}'.format(lambda_))
        test_results = OrderedDict()
        test_results['psnr'] = []
        if calc_LPIPS:
            test_results['lpips'] = []

        for idx, img in enumerate(L_paths):

            # --------------------------------
            # (1) get img_H and img_L
            # --------------------------------

            idx += 1
            img_name, ext = os.path.splitext(os.path.basename(img))
            img_H = util.imread_uint(img, n_channels=n_channels)

            # --------------------------------
            # (2) initialize x
            # --------------------------------
            if load_mask:
                mask = util.imread_uint(mask_path, n_channels=n_channels).astype(bool)
            else:
                mask = bayer_mask(*img_H.shape[:2])

            img_L = img_H * mask / 255.   # (256,256,3)   [0,1]  -- everything below stays in [0,1]

            np.random.seed(seed=0)  # for reproducibility
            img_L += np.random.normal(0, noise_level_img, img_L.shape)  # add AWGN, [0,1] scale
            img_L = img_L * mask

            y = util.single2tensor4(img_L).to(device)   # (1,3,256,256), [0,1]
            mask = util.single2tensor4(mask.astype(np.float32)).to(device)

            # matlab (Malvar-He-Cutler) demosaicing as the initial estimate,
            # i.e. DPIR's matlab_init  (utils_mosaic.dm is the plain bilinear one)
            x = utils_mosaic.dm_matlab(pack_cfa4(y, mask))

            # --------------------------------
            # (3) get rhos and sigmas
            # --------------------------------
            # DPIR's HQS schedule: sigma_k log-spaced from modelSigma1 to modelSigma2,
            # rho_k = lambda_ * sigma_y^2 / sigma_k^2   (utils_pnp.get_rho_sigma, w=1.0)

            modelSigmaS = np.logspace(np.log10(modelSigma1), np.log10(modelSigma2), iter_num).astype(np.float32)
            sigmas = modelSigmaS / 255.
            rhos = [lambda_*(sigma**2)/(s**2) for s in sigmas]
            rhos, sigmas = torch.tensor(rhos).to(device), torch.tensor(sigmas).to(device)

            # --------------------------------
            # (4) main iterations
            # --------------------------------

            progress_img = []
            progress_seq = list(range(0, iter_num, max(1, iter_num//10)))
            if progress_seq[-1] != iter_num-1:
                progress_seq.append(iter_num-1)

            for i in range(iter_num):

                # --------------------------------
                # step 1, data sub-problem: closed-form solution
                # --------------------------------
                # same line as the DiffPIR run (main_ddpir_demosaic.py), mask is the same CFA

                x = (mask*y + rhos[i].float()*x).div(mask + rhos[i])

                # --------------------------------
                # step 2, prior sub-problem: DRUNet at sigma_k
                # --------------------------------

                x = torch.clamp(x, 0, 1)
                if x8:
                    x = util.augment_img_tensor4(x, i % 8)

                x = torch.cat((x, sigmas[i].float().repeat(1, 1, x.shape[2], x.shape[3])), dim=1)
                x = model(x)

                if x8:
                    if i % 8 == 3 or i % 8 == 5:
                        x = util.augment_img_tensor4(x, 8 - i % 8)
                    else:
                        x = util.augment_img_tensor4(x, i % 8)

                # recover the measured CFA samples (DPIR does this every iteration)
                if dc_mode == 'loop':
                    x[mask.to(torch.bool)] = y[mask.to(torch.bool)]

                # save the process
                if save_progressive and (i in progress_seq):
                    x_show = x.clone().detach().cpu().numpy()       # [0,1]
                    x_show = np.squeeze(x_show)
                    if x_show.ndim == 3:
                        x_show = np.transpose(x_show, (1, 2, 0))
                    progress_img.append(x_show)
                    if log_process:
                        logger.info('iter: {:>4d}, sigma: {:.4f}, rho: {:.4f}, np.max: {:.4f}, np.min: {:.4f}'.format(i, sigmas[i].item()*255, rhos[i].item(), np.max(x_show), np.min(x_show)))
                    if show_img:
                        util.imshow(x_show)

            if dc_mode == 'final':          # DiffPIR's "recover conditional part", once per image
                x[mask.to(torch.bool)] = y[mask.to(torch.bool)]

            x_0 = torch.clamp(x, 0, 1)      # [0,1] throughout, no /2+0.5 here

            # --------------------------------
            # (5) metrics and saving
            # --------------------------------

            img_E = util.tensor2uint(x_0)

            psnr = util.calculate_psnr(img_E, img_H, border=0)  # border=0, same as the DiffPIR run
            test_results['psnr'].append(psnr)

            if calc_LPIPS:
                img_H_tensor = np.transpose(img_H, (2, 0, 1))
                img_H_tensor = torch.from_numpy(img_H_tensor)[None,:,:,:].to(device)
                img_H_tensor = img_H_tensor / 255 * 2 -1
                # lpips.LPIPS wants [-1,1] inputs, so both sides are mapped here;
                # the pipeline itself stays in [0,1]
                lpips_score = loss_fn_vgg(x_0.detach()*2-1, img_H_tensor)
                lpips_score = lpips_score.cpu().detach().numpy()[0][0][0][0]
                test_results['lpips'].append(lpips_score)
                logger.info('{:->4d}--> {:>10s} PSNR: {:.4f}dB LPIPS: {:.4f} ave LPIPS: {:.4f}'.format(idx, img_name+ext, psnr, lpips_score, sum(test_results['lpips']) / len(test_results['lpips'])))
            else:
                logger.info('{:->4d}--> {:>10s} PSNR: {:.4f}dB'.format(idx, img_name+ext, psnr))

            if save_E:
                util.imsave(img_E, os.path.join(E_path, img_name+'_'+model_name+ext))

            if save_L:
                util.imsave(util.single2uint(img_L), os.path.join(E_path, img_name+'_L'+ext))

            if save_LEH:
                util.imsave(np.concatenate([util.single2uint(img_L), img_E, img_H], axis=1), os.path.join(E_path, img_name+model_name+'_LEH'+ext))

            if save_progressive:
                now = datetime.now()
                current_time = now.strftime("%Y_%m_%d_%H_%M_%S")
                img_total = cv2.hconcat(progress_img)
                if show_img:
                    util.imshow(img_total, figsize=(80,4))
                util.imsave(img_total*255., os.path.join(E_path, img_name+'_process_lambda_{:.3f}_{}{}'.format(lambda_, current_time, ext)))

        # --------------------------------
        # Average PSNR and LPIPS
        # --------------------------------

        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        logger.info('------> Average PSNR of ({}), sigma: ({:.3f}), lambda: ({:.3f}): {:.4f} dB'.format(testset_name, noise_level_model, lambda_, ave_psnr))

        if calc_LPIPS:
            ave_lpips = sum(test_results['lpips']) / len(test_results['lpips'])
            logger.info('------> Average LPIPS of ({}), sigma: ({:.3f}), lambda: ({:.3f}): {:.4f}'.format(testset_name, noise_level_model, lambda_, ave_lpips))

    # experiments: sweep lambda if you want to tune rho (DPIR's own value is 0.23)
    lambdas = [lambda_]
    for lambda_ in lambdas:
        test_rho(lambda_)

if __name__ == '__main__':

    main()
