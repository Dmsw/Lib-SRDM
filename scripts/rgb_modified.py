import argparse
import os

import blobfile as bf
import scipy.io as scio
import numpy as np
import torch as th
import torch.distributed as dist
import yaml
import cv2 as cv
import torch.nn.functional as F

from pysptools.abundance_maps import UCLS

import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "../")
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    spatial_model_and_diffusion_defaults,
    create_spatial_model_and_diffusion,
    spectral_model_and_diffusion_defaults,
    create_spectral_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
    )
from guided_diffusion.prior_model import PriorModels
from guided_diffusion.image_datasets import load_hsi_data
from guided_diffusion.ntire_gen import load_ntire2022
from guided_diffusion.icvl import load_icvl_test
from torchvision import utils
from measurement import rgb2hsi
from utils import calc_psnr, calc_ssim, calc_sam, tensor_1_mode_product, tensor_2_mode_product, \
    estimate_sigma, estimate_cov, calc_covariance, plot_spectrum, load_yaml
from functools import partial
from srf_tools import SRFTool
from lib_and_mask import get_lib_and_mask
from guided_diffusion.low_rank_model import LowRankGradients
import pickle

srf = SRFTool()
P = th.from_numpy(srf.load_d400_srf().astype(np.float32))


def main():
    th.manual_seed(42)
    np.random.seed(42)

    dist_util.setup_dist()
    logger.configure(dir=cfg['save_dir'])
    logger.log(cfg)
    yaml.dump(cfg, open(os.path.join(cfg['save_dir'], "config.yaml"), "w"))

    logger.log("creating model...")
    dm_model_kwargs = spatial_model_and_diffusion_defaults()
    custom_config = load_yaml(cfg['spatial_kwargs']['model_config'])
    for k, v in custom_config.items():
        dm_model_kwargs[k] = v
    model, diffusion = create_spatial_model_and_diffusion(**dm_model_kwargs)

    model.load_state_dict(
        dist_util.load_state_dict(cfg['spatial_kwargs']['model_path'], map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("loading data...")
    hsi = np.load(cfg['base_samples'])['hsi']
    target = th.from_numpy(hsi)
    hsi = hsi.transpose(1, 2, 0)
    rgb_input = srf.gen_rgb_numpy(hsi, max_value=1).transpose(2, 0, 1) * 2 - 1
    
    logger.log("creating samples...")
    count = 0
    apsnr = 0
    assim = 0
    asam = 0
    
    model_kwargs = {}
    model_kwargs['ref_img'] = th.from_numpy(rgb_input)
    
    """
    load spectral model
    """
    if cfg['spectral_kwargs'] is not None:
        logger.log("creating spectral model...")
        spectral_model_kwargs = spectral_model_and_diffusion_defaults()
        custom_config = load_yaml(cfg['spectral_kwargs']['model_config'])
        for k, v in custom_config.items():
            spectral_model_kwargs[k] = v
        _, spectral_diffusion = create_spectral_model_and_diffusion(**spectral_model_kwargs)
        rgb = model_kwargs['ref_img'].numpy().transpose(1, 2, 0)
        dm_lib, lr_lib, mask, factor = get_lib_and_mask(rgb * 0.5 + 0.5, gamma=0.6, srf=P.numpy().T, lib_path=cfg['library_kwargs']['lib_path'], sample_per_tag=cfg['library_kwargs']['samples_per_tag'], tau=cfg['tau'], logdir=os.path.join(cfg['save_dir'], f"lib-{str(count).zfill(5)}"), memory_save=True)
        print(list(dm_lib.keys()))
        if cfg['k_svd'] > 0:
            lr_grad = LowRankGradients(dm_lib, mask, factor, diffusion.alphas_cumprod, cfg['k_svd'])
        spectral_model = PriorModels(dm_lib, mask, factor, bar_alpha=spectral_diffusion.alphas_cumprod)
        spectral_model.to(dist_util.dev())

        spectral_model.eval()
    else:
        raise ValueError("spectral model config is required")

    cov = calc_covariance(model_kwargs["ref_img"], tensor_1_mode_product(target, P.T))
    model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
    cov = cov.to(dist_util.dev())
    lr_grad = lr_grad.to(dist_util.dev()) if cfg['k_svd'] > 0 else None
    # sigma = estimate_sigma(model_kwargs["ref_img"], args.k)
    # print("err:", th.mean((sigma**2 - th.diagonal(cov))**2))
    model_kwargs['ref_img'] = model_kwargs['ref_img'][None]
    sample, _ = diffusion.p_sample_loop(
        model=model,
        spectral_model=spectral_model,
        l1=cfg['l1'],
        N_switch=cfg['N_switch'],
        l3=cfg['l3'],
        spectral_diffusion=spectral_diffusion,
        shape=(cfg['n_samples'], 31, *target.shape[-2:]),
        clip_denoised=cfg['clip_denoised'],
        model_kwargs=model_kwargs,
        measure_fn=partial(rgb2hsi, P=P.to(dist_util.dev())),
        range_t=cfg['range_t'],
        progress=True,
        var=cov,
        lr_grad=lr_grad
        # var = th.diag(sigma**2).to(dist_util.dev()),      
        # var = estimate_cov(model_kwargs["ref_img"], args.k),  
    )
    # plot_uncertainty(unceratainty, os.path.join(logger.get_dir(), f"uncertainty-{str(count).zfill(5)}.png"))


    # load lib
    lib = pickle.load(open(os.path.join(cfg['library_kwargs']['lib_path'], 'lib.pkl'), 'rb'))

    sample = (th.mean(sample, dim=0, keepdim=True) + 1)/2
    rec_hsi = sample[0].cpu().numpy()
    input_spectra = rec_hsi[:, mask[cfg['target_instance']]]
    target_spectra = lib[cfg['target_spectra']].T
    print(input_spectra.shape)
    print(target_spectra.shape)
    _, S, VH = np.linalg.svd(input_spectra, full_matrices=False)
    U, _, _ = np.linalg.svd(target_spectra, full_matrices=False)
    modified_spectra = U @ np.diag(S) @ VH
    modified_rgb = srf.P @ modified_spectra
    rgb_output = rgb_input.copy()
    rgb_output[:, mask[cfg['target_instance']]] = modified_rgb.clip(0, 1)
    
    out_path = os.path.join(cfg['save_dir'], f"sample.png")
    rgb_output = (np.round(rgb_output.transpose(1, 2, 0) * 255)).astype(np.uint8)
    rgb_output = cv.cvtColor(rgb_output, cv.COLOR_RGB2BGR)
    cv.imwrite(out_path, rgb_output)

    dist.barrier()
    logger.log("sampling complete")
    # apsnr /= count
    # assim /= count
    # asam /= count
    logger.log(f"average PSNR: {apsnr}")
    logger.log(f"average SSIM: {assim}")
    logger.log(f"average SAM: {asam}")


if __name__ == "__main__":
    cfg = {
        'clip_denoised': True,
        'num_samples': 3,
        'batch_size': 1,
        'range_t': 0,
        'use_ddim': False,
        'n_samples': 1,
        'save_dir': "/home/root/project/hsi-denoising/GDM/results/TEST",
        'spatial_kwargs' : {
            'model_config': "/home/root/project/hsi-denoising/GDM/config/model_config.yaml",
            'model_path': "/home/root/project/posterior_diffusion/models/hsi256/model040000.pt",
        },
        'spectral_kwargs': {
            "model_config": "/home/root/project/hsi-denoising/GDM/config/spectral_prior.yaml",
        },
        'library_kwargs': {
            'lib_path': "/home/root/project/hsi-denoising/GDM/library/cave_all500_pretag_hq/",
            'samples_per_tag': 100,
        },
        'l1': 1.0,
        'N_switch': 50,   # which is used to be l2 but now is N_switch
        'l3': 1,
        'base_samples': "/home/root/dataset/cave/cave/fake_and_real_tomatoes_ms/fake_and_real_tomatoes_ms.npz",
        'k': 5,
        'k_svd': 4,
        'tau': 0.5,
        'cfg_path': None,
        'target_instance': 'slice-2',
        'target_spectra': 'beer',
    }
    if cfg['cfg_path'] is not None:
        cfg.update(load_yaml(cfg['cfg_path']))
    main()