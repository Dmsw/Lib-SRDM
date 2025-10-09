import argparse
import os

import blobfile as bf
import scipy.io as scio
import numpy as np
import torch as th
import torch.distributed as dist
import yaml
import cv2 as cv

import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    spatial_model_and_diffusion_defaults,
    create_spatial_model_and_diffusion,
    spectral_model_and_diffusion_defaults,
    create_spectral_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
    )
from guided_diffusion.prior_model import PriorModel
from guided_diffusion.image_datasets import load_hsi_data
from torchvision import utils
from measurement import denoising
from utils import calc_psnr, calc_ssim, calc_sam
from functools import partial


def estimate_sigma(image, k, k_m=1):
    assert k > k_m
    image = np.array(image.to("cpu"))
    image = np.transpose(image, [1, 2, 0])
    blur = cv.blur(image, (k, k))
    v = np.var(blur - image, axis=(0, 1))
    std = np.sqrt(v*(k**2)/(k**2-k_m**2))
    return th.from_numpy(std).to(dist_util.dev())


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


def random_build_spectral_lib(hsi, num):
    hsi = hsi.reshape([hsi.shape[0], -1])
    idx = np.random.choice(hsi.shape[1], num, replace=False)
    return hsi[:, idx]


def calc_covariance(noise, true):
    noise = noise.reshape([noise.shape[0], -1])
    true = true.reshape([true.shape[0], -1])
    noise = noise - true
    cov = noise @ noise.T / noise.shape[1]
    return cov

# added
def load_noise_hsi(data_dir, batch_size):
    data = load_hsi_data(
        data_dir=data_dir,
        batch_size=batch_size,
        deterministic=True,
        mode="test",
    )
    for large_batch, model_kwargs in data:
        model_kwargs["ref_img"] = large_batch[1][0]
        yield model_kwargs, large_batch[0][0]


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


def load_P():
    P = scio.loadmat("/home/root/dataset/cave/P.mat")['P']
    return th.from_numpy(P.astype(np.float32)).to(dist_util.dev())


def main():
    # th.manual_seed(0)

    dist_util.setup_dist()
    logger.configure(dir=cfg['save_dir'])
    logger.log(cfg)
    yaml.dump(cfg, open(os.path.join(cfg['save_dir'], "config.yaml"), "w"))

    logger.log("creating model...")
    model_kwargs = spatial_model_and_diffusion_defaults()
    custom_config = load_yaml(cfg['spatial_kwargs']['model_config'])
    for k, v in custom_config.items():
        model_kwargs[k] = v
    model, diffusion = create_spatial_model_and_diffusion(**model_kwargs)

    model.load_state_dict(
        dist_util.load_state_dict(cfg['spatial_kwargs']['model_path'], map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("loading data...")
    data = load_noise_hsi(
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
        if cfg['spectral_kwargs'] is not None:
            logger.log("creating spectral model...")
            spectral_model_kwargs = spectral_model_and_diffusion_defaults()
            custom_config = load_yaml(cfg['spectral_kwargs']['model_config'])
            for k, v in custom_config.items():
                spectral_model_kwargs[k] = v
            _, spectral_diffusion = create_spectral_model_and_diffusion(**spectral_model_kwargs)
            lib = random_build_spectral_lib(target, cfg['library_size'])
            if cfg['k_svd'] > 0:
                U, _, _ = th.svd(lib)
                U = U[:, :cfg['k_svd']]
            spectral_model = PriorModel(lib.T, bar_alpha=spectral_diffusion.alphas_cumprod)
            spectral_model.to(dist_util.dev())

            spectral_model.eval()
        else:
            raise ValueError("spectral model config is required")
        P = load_P()

        cov = calc_covariance(model_kwargs["ref_img"], target)
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
        cov = cov.to(dist_util.dev())
        P = P.to(dist_util.dev())
        U = U.to(dist_util.dev()) if cfg['k_svd'] > 0 else None
        # sigma = estimate_sigma(model_kwargs["ref_img"], args.k)
        # print("err:", th.mean((sigma**2 - th.diagonal(cov))**2))
        model_kwargs['ref_img'] = model_kwargs['ref_img'][None]
        sample, _ = diffusion.p_sample_loop(
            model=model,
            spectral_model=spectral_model,
            l1=cfg['l1'],
            l2=cfg['l2'],
            spectral_diffusion=spectral_diffusion,
            shape=(cfg['n_samples'], 31, *target.shape[-2:]),
            clip_denoised=cfg['clip_denoised'],
            model_kwargs=model_kwargs,
            measure_fn=denoising,
            range_t=cfg['range_t'],
            progress=True,
            var=cov,
            # var = th.diag(sigma**2).to(dist_util.dev()),      
            # var = estimate_cov(model_kwargs["ref_img"], args.k),  
        )
        

        sample = (th.mean(sample, dim=0, keepdim=True) + 1)/2
        out_path = os.path.join(logger.get_dir(),
                                f"{str(count).zfill(5)}.png")
        utils.save_image(
            sample[0, 0].unsqueeze(0),
            out_path,
            nrow=1,
            normalize=True,
            range=(0, 1),
        )

        out_path = os.path.join(logger.get_dir(),
                                f"{str(count).zfill(5)}.npy")
        np.save(out_path, sample.cpu().numpy())

        target = (target + 1)/2
        out_path = os.path.join(logger.get_dir(),
                                f"target-{str(count).zfill(5)}.png")
        utils.save_image(
            target[0].unsqueeze(0),
            out_path,
            nrow=1,
            normalize=True,
            range=(0, 1),
        )

        noise = model_kwargs['ref_img'][0]
        noise = (noise + 1)/2
        out_path = os.path.join(logger.get_dir(),
                                f"noise-{str(count).zfill(5)}.png")
        utils.save_image(
            noise[0].unsqueeze(0),
            out_path,
            nrow=1,
            normalize=True,
            range=(0, 1),
        )
        target = target.numpy()
        noise = noise.cpu().numpy()
        sample = sample.cpu().numpy()[0]
        apsnr += calc_psnr(target, sample)
        assim += calc_ssim(target, sample)
        asam += calc_sam(target, sample)
        plot_spectrum(sample, target, os.path.join(logger.get_dir(), f"sp-{str(count).zfill(5)}.png"))
        logger.log("PSNR:")
        logger.log(f"count:{count}, before: {calc_psnr(target, noise)}")
        logger.log(f"count:{count}, after: {calc_psnr(target, sample)}")
        logger.log("SSIM:")
        logger.log(f"count:{count}, before: {calc_ssim(target, noise)}")
        logger.log(f"count:{count}, after: {calc_ssim(target, sample)}")
        logger.log("SAM:")
        logger.log(f"count:{count}, before: {calc_sam(target, noise)}")
        logger.log(f"count:{count}, after: {calc_sam(target, sample)}")

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


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

if __name__ == "__main__":
    cfg = {
        'clip_denoised': True,
        'num_samples': 3,
        'batch_size': 1,
        'range_t': 0,
        'use_ddim': False,
        'n_samples': 1,
        'library_size': 100,
        'save_dir': "/home/root/project/hsi-denoising/GDM/results/den_lr5/",
        'spatial_kwargs' : {
            'model_config': "/home/root/project/hsi-denoising/GDM/config/model_config.yaml",
            'model_path': "/home/root/project/posterior_diffusion/models/hsi256/model040000.pt",
        },
        'spectral_kwargs': {
            "model_config": "/home/root/project/hsi-denoising/GDM/config/spectral_prior.yaml",
        },
        'l1': 2.,
        'l2': 0.1,
        'l3': 1.,
        'base_samples': "/home/root/dataset/cave/fixedga75/",
        # 'base_samples': "/home/root/dataset/NTIRE22/",
        'k': 5,
        'k_svd': 5,
        'cfg_path': None,
    }
    if cfg['cfg_path'] is not None:
        cfg.update(load_yaml(cfg['cfg_path']))
    main()