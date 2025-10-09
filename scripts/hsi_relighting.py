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

from mpl_toolkits.mplot3d.art3d import PolyCollection

import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "../")
from illuminant import Illuminant

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
from guided_diffusion.icvl import load_icvl
from guided_diffusion.cave import load_cave
from torchvision import utils
from measurement import rgb2hsi
from utils import calc_psnr, calc_ssim, calc_sam, tensor_1_mode_product, tensor_2_mode_product, denoising
from functools import partial
from srf_tools import SRFTool
from lib_and_mask import get_lib_and_mask
from guided_diffusion.low_rank_model import LowRankGradients
from torchvision.transforms import ToPILImage


cfg = {
    'clip_denoised': True,
    'num_samples': 50,
    'batch_size': 1,
    'range_t': 0,
    'use_ddim': False,
    'n_samples': 1,
    "light": "F6",
    'save_dir': "/home/root/project/hsi-denoising/GDM/results/TEST/",
    'spatial_kwargs' : {
        'model_config': "/home/root/project/hsi-denoising/GDM/config/model_config.yaml",
        'model_path': "/home/root/project/hsi-denoising/GDM/models/ntire22/model100000.pt",
    },
    'spectral_kwargs': {
        "model_config": "/home/root/project/hsi-denoising/GDM/config/spectral_prior.yaml",
    },
    'library_kwargs': {
        'lib_path': "/home/root/project/hsi-denoising/GDM/library/ntire_awb/",
        'samples_per_tag': 500,
    },
    'l1': 3,
    'N_switch': 50,   # which is used to be l2 but now is N_switch
    'l3': 1,
    # 'base_samples': "/home/root/dataset/cave/rgb2hsi/test/",
    'base_samples': "/home/root/dataset/NTIRE22/",
    'k': 5,
    'k_svd': 4,
    'tau': 0.5,
    'srf_name': 'NTIRE',
    'cfg_path': None,
}

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

if cfg['cfg_path'] is not None:
    cfg.update(load_yaml(cfg['cfg_path']))


# shape 3, 31
srf = SRFTool()
srf.load_srf(cfg['srf_name'])
if cfg['light'] == "F6":
    input_illuminant = Illuminant(file="CIE_illum_FLs_1nm.csv", light_type=6)
elif cfg['light'] == "A1":
    input_illuminant = Illuminant(file="CIE_std_illum_A_1nm.csv", light_type=1, root="/home/root/dataset/illuminant/")
else:
    raise ValueError(f"Unknown light type {cfg['light']}")
target_illuminant = Illuminant(file="CIE_std_illum_D65.csv", light_type=1, root="/home/root/dataset/illuminant/")
input_srf = input_illuminant.lighting_srf(srf)
target_srf = target_illuminant.lighting_srf(srf)
del srf # prevent calling


def estimate_cov(image, k, k_m=1):
    assert k > k_m
    image = np.array(image.to("cpu"))
    image = np.transpose(image, [1, 2, 0])
    blur = cv.blur(image, (k, k))
    noise = blur - image
    noise = noise.reshape([-1, noise.shape[-1]])
    cov = noise.T @ noise / noise.shape[0] * (k**2) / (k**2 - k_m**2)
    cov[np.abs(cov) < 2e-3] = 0
    return th.from_numpy(cov.astype(np.float32)).to(dist_util.dev())


def calc_covariance(noise, true):
    noise = noise.reshape([noise.shape[0], -1])
    true = true.reshape([true.shape[0], -1])
    noise = noise - true
    cov = noise @ noise.T / noise.shape[1]
    return cov


# added
def load_cave_test(data_dir, batch_size):
    data = load_cave(
        data_root=data_dir,
        batch_size=batch_size,
        deterministic=True,
        mode="test",
    )
    for large_batch, model_kwargs in data:
        target = large_batch['input']
        rgb = input_srf.gen_rgb_torch(target*0.5+0.5, max_value=1, clip=True, normalize=False, quantize=False, awb=False)
        model_kwargs['ref_img'] = rgb[0]
        yield model_kwargs, target[0]


def load_ntire_test(data_dir, batch_size):
    data = load_ntire2022(
        data_root=data_dir,
        batch_size=batch_size,
        deterministic=True,
        mode="test",
    )
    for large_batch in data:
        target = large_batch['hsi'][..., 1:-1, :].clip(-1, 1)
        # rgb = large_batch['rgb'][0][..., 1:-1, :]
        rgb = input_srf.gen_rgb_torch(target*0.5+0.5, max_value=1, clip=True, normalize=False, quantize=False, awb=False)
        model_kwargs = {"ref_img":rgb[0]}
        yield model_kwargs, target[0]


def load_icvl_test(data_dir, batch_size):
    data = load_icvl(
        data_root=data_dir,
        deterministic=True,
        mode="test",
    )
    for large_batch in data:
        target = large_batch['hsi']
        rgb = input_srf.gen_rgb_torch(target*0.5+0.5, max_value=1, clip=True, normalize=False, quantize=False, awb=False)
        model_kwargs = {"ref_img":rgb[0]}
        yield model_kwargs, target[0]

def plot_spectrum(pred, target, save, pos=(0.5, 0.5)):
    C, H, W = pred.shape
    x, y = int(pos[0]*W), int(pos[1]*H)
    pred = pred[:, y, x]
    target = target[:, y, x]
    plt.plot(pred, label="pred")
    plt.plot(target, label="target")
    plt.legend()
    plt.savefig(save)
    plt.close()


def main():
    th.manual_seed(42)
    np.random.seed(42)

    dist_util.setup_dist()
    cfg['save_dir'] = os.path.join(cfg['save_dir'], cfg['light'])
    logger.configure(dir=cfg['save_dir'])
    logger.log(cfg)
    yaml.dump(cfg, open(os.path.join(cfg['save_dir'], "config.yaml"), "w"))

    logger.log("creating model...")
    hsi_model_kwargs = spatial_model_and_diffusion_defaults()
    custom_config = load_yaml(cfg['spatial_kwargs']['model_config'])
    for k, v in custom_config.items():
        hsi_model_kwargs[k] = v
    model, diffusion = create_spatial_model_and_diffusion(**hsi_model_kwargs)

    model.load_state_dict(
        dist_util.load_state_dict(cfg['spatial_kwargs']['model_path'], map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("loading data...")
    data = load_ntire_test(
        cfg['base_samples'],
        cfg['batch_size'],
    )
    

    logger.log("creating samples...")
    count = 0
    apsnr = 0
    assim = 0
    asam = 0
    while count * cfg['batch_size'] < cfg['num_samples']:
        model_kwargs, target = next(data)
        
        """
        load spectral model
        """
        # P = th.from_numpy(srf.load_srf('N900').astype(np.float32))
        if cfg['spectral_kwargs'] is not None:
            logger.log("creating spectral model...")
            spectral_model_kwargs = spectral_model_and_diffusion_defaults()
            custom_config = load_yaml(cfg['spectral_kwargs']['model_config'])
            for k, v in custom_config.items():
                spectral_model_kwargs[k] = v
            spectral_model_kwargs['diffusion_steps'] = hsi_model_kwargs['diffusion_steps']
            spectral_model_kwargs['rescale_timesteps'] = hsi_model_kwargs['rescale_timesteps']
            spectral_model_kwargs['timestep_respacing'] = hsi_model_kwargs['timestep_respacing']
            _, spectral_diffusion = create_spectral_model_and_diffusion(**spectral_model_kwargs)
            rgb = model_kwargs['ref_img'].numpy().transpose(1, 2, 0)
            dm_lib, lr_lib, mask, factor = get_lib_and_mask(rgb, gamma=0.6, srf=input_srf, lib_path=cfg['library_kwargs']['lib_path'], sample_per_tag=cfg['library_kwargs']['samples_per_tag'], tau=cfg['tau'], logdir=os.path.join(cfg['save_dir'], f"lib-{str(count).zfill(5)}"), memory_save=True)
            
            if cfg['k_svd'] > 0:
                lr_grad = LowRankGradients(dm_lib, mask, factor, diffusion.alphas_cumprod, cfg['k_svd'])
            spectral_model = PriorModels(dm_lib, mask, factor, bar_alpha=spectral_diffusion.alphas_cumprod)
            spectral_model.to(dist_util.dev())

            spectral_model.eval()
        else:
            raise ValueError("spectral model config is required")

        cov = calc_covariance(model_kwargs["ref_img"], tensor_1_mode_product(target*0.5+0.5, input_srf.get_P(target.device).T)) * 4
        print(cov)
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
        cov = cov.to(dist_util.dev())
        target = target.to(dist_util.dev())
        lr_grad = lr_grad.to(dist_util.dev()) if cfg['k_svd'] > 0 else None
        # sigma = estimate_sigma(model_kwargs["ref_img"], args.k)
        # print("err:", th.mean((sigma**2 - th.diagonal(cov))**2))
        model_kwargs['ref_img'] = model_kwargs['ref_img'] * 2 - th.sum(input_srf.get_P(dist_util.dev()), dim=1).view(3, 1, 1)
                
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
            measure_fn=partial(rgb2hsi, P=input_srf.get_P(dist_util.dev())),
            range_t=cfg['range_t'],
            progress=True,
            var=cov,
            lr_grad=lr_grad
            # var = th.diag(sigma**2).to(dist_util.dev()),      
            # var = estimate_cov(model_kwargs["ref_img"], args.k),  
        )
        # plot_uncertainty(unceratainty, os.path.join(logger.get_dir(), f"uncertainty-{str(count).zfill(5)}.png"))

        sample = (th.mean(sample, dim=0, keepdim=True) + 1)/2
        sample = target_srf.gen_rgb_torch(sample, max_value=1.0, normalize=True, depth=8).to(th.float32)

        target = (target[None] + 1)/2
        target = target_srf.gen_rgb_torch(target, max_value=1.0, normalize=True, depth=8).to(th.float32)

        # target = target.numpy()
        # noise = noise.cpu().numpy()
        # sample = sample.cpu().numpy()
        apsnr += calc_psnr(target, sample)
        assim += calc_ssim(target, sample)
        asam += calc_sam(target, sample)
        # plot_spectrum(sample, target, os.path.join(logger.get_dir(), f"sp-{str(count).zfill(5)}.png"))
        logger.log("PSNR:")
        # logger.log(f"count:{count}, before: {calc_psnr(target, noise)}")
        logger.log(f"count:{count}, after: {calc_psnr(target, sample)}")
        logger.log("SSIM:")
        # logger.log(f"count:{count}, before: {calc_ssim(target, noise)}")
        logger.log(f"count:{count}, after: {calc_ssim(target, sample)}")
        logger.log("SAM:")
        # logger.log(f"count:{count}, before: {calc_sam(target, noise)}")
        logger.log(f"count:{count}, after: {calc_sam(target, sample)}")
        
        topil = ToPILImage()
        path = os.path.join(logger.get_dir(), f"output-{str(count).zfill(5)}.png")
        pil = topil(sample[0])
        pil.save(path)
                
        logger.log(f"created {count * cfg['batch_size']} samples")

        # t_psnr = []
        # t_ssim = []
        # for hsi in hsi_t:
        #     hsi = (hsi+1)/2
        #     t_psnr.append(calc_psnr(target, hsi))
        #     t_ssim.append(calc_ssim(target, hsi))
        # t_psnr = np.array(t_psnr)
        # t_ssim = np.array(t_ssim)

        # np.savez(os.path.join(logger.get_dir(), f"{str(count).zfill(5)}_theory.npz"), psnr=t_psnr, ssim=t_ssim, hsi=hsi_t, rgb=rgb_t)

        count += 1

    dist.barrier()
    logger.log("sampling complete")
    apsnr /= count
    assim /= count
    asam /= count
    logger.log(f"average PSNR: {apsnr}")
    logger.log(f"average SSIM: {assim}")
    logger.log(f"average SAM: {asam}")

if __name__ == "__main__":
    main()