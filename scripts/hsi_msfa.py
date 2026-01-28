import argparse
import os

import blobfile as bf
import scipy.io as scio
import numpy as np
import torch as th
import torch.distributed as dist
import yaml
import cv2 as cv

from mpl_toolkits.mplot3d.art3d import PolyCollection

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
from measurement import msfa2hsi, rgb2hsi, msfa2hsi_with_wb
from utils import calc_psnr, calc_ssim, calc_sam, tensor_1_mode_product, tensor_2_mode_product, \
    estimate_sigma, estimate_cov, calc_covariance, random_build_spectral_lib, load_yaml
from functools import partial
from msfa import MSFAModel

def plot_uncertainty(uncertainty, save):
    # def polygon_under_graph(x, y):
    #     """
    #     Construct the vertex list which defines the polygon filling the space under
    #     the (x, y) line graph. This assumes x is in ascending order.
    #     """
    #     return [(x[0], 0.), *zip(x, y), (x[-1], 0.)]
    # ax = plt.figure().add_subplot(projection='3d')
    # facecolors = plt.colormaps['viridis'](np.linspace(0, 1, len(uncertainty)))
    # n = len(uncertainty)
    # verts = []
    # for i in reversed(range(n)):
    #     # ax.fill_between(range(31), i*100, uncertainty[i], facecolor=facecolors[i], alpha=0.5)
    #     verts.append(polygon_under_graph(range(31), uncertainty[i]))
    # poly = PolyCollection(verts, facecolors=facecolors, alpha=0.5)
    # ax.add_collection3d(poly, zs=np.arange(n)*100, zdir='y')
    # ax.set_zscale('log')
    # ax.set(xlim=(0, 31), ylim=(0, n*100), xlabel='wavelength', ylabel='timestep', zlabel='uncertainty')
    # plt.savefig(save)
    # plt.close()
    
    uncertainty_map = np.stack(np.log(uncertainty))
    np.save(save.replace(".png", ".npy"), uncertainty_map)
    plt.imshow(uncertainty_map, cmap='viridis')
    plt.colorbar()
    plt.savefig(save)
    plt.close()

msfa = MSFAModel(msfa_file="/home/root/project/stdm/dataset/MSFA16.npy")

# added
def load_noise_hsi(data_dir, batch_size):
    data = load_hsi_data(
        data_dir=data_dir,
        batch_size=batch_size,
        deterministic=True,
        mode="test",
    )
    for large_batch, model_kwargs in data:
        target = large_batch[0][0]
        raw = msfa(target)
        model_kwargs['ref_img'] = raw
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


def main():
    # th.manual_seed(0)

    dist_util.setup_dist()
    logger.configure(dir=cfg['save_dir'])
    logger.log(cfg)

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
        msfa = MSFAModel(msfa_file="/home/root/project/stdm/dataset/MSFA16.npy")

        wb = msfa.wb_filter(model_kwargs["ref_img"][None])[0]
        wb_cov = calc_covariance(wb, tensor_1_mode_product(target, msfa.P()))
        cov = calc_covariance(model_kwargs["ref_img"], msfa(target))
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
        cov = cov.to(dist_util.dev())
        msfa = msfa.to(dist_util.dev())
        U = U.to(dist_util.dev()) if cfg['k_svd'] > 0 else None
        # sigma = estimate_sigma(model_kwargs["ref_img"], args.k)
        # print("err:", th.mean((sigma**2 - th.diagonal(cov))**2))
        model_kwargs['ref_img'] = model_kwargs['ref_img'][None]
        wb = wb.to(dist_util.dev())
        wb_cov = wb_cov.to(dist_util.dev())
        # model_kwargs['ref_img'] = wb.cuda()
        assert True, th.mean((model_kwargs['ref_img'] - tensor_1_mode_product(target.cuda(), msfa.P()))**2)
        sample, unceratainty = diffusion.p_sample_loop(
            model=model,
            spectral_model=spectral_model,
            l1=cfg['l1'],
            l2=cfg['l2'],
            spectral_diffusion=spectral_diffusion,
            shape=(cfg['n_samples'], 31, *target.shape[-2:]),
            clip_denoised=cfg['clip_denoised'],
            model_kwargs=model_kwargs,
            measure_fn=partial(msfa2hsi_with_wb, msfa=msfa, wb=wb, wb_cov=wb_cov),
            range_t=cfg['range_t'],
            progress=True,
            var=cov,
            U=U,
            # var = th.diag(sigma**2).to(dist_util.dev()),      
            # var = estimate_cov(model_kwargs["ref_img"], args.k),  
        )
        # plot_uncertainty(unceratainty, os.path.join(logger.get_dir(), f"uncertainty-{str(count).zfill(5)}.png"))

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
            noise.unsqueeze(0),
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
        # logger.log(f"count:{count}, before: {calc_psnr(target, noise)}")
        logger.log(f"count:{count}, after: {calc_psnr(target, sample)}")
        logger.log("SSIM:")
        # logger.log(f"count:{count}, before: {calc_ssim(target, noise)}")
        logger.log(f"count:{count}, after: {calc_ssim(target, sample)}")
        logger.log("SAM:")
        # logger.log(f"count:{count}, before: {calc_sam(target, noise)}")
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


if __name__ == "__main__":
    cfg = {
        'clip_denoised': True,
        'num_samples': 3,
        'batch_size': 1,
        'range_t': 0,
        'use_ddim': False,
        'n_samples': 1,
        'library_size': 100,
        'save_dir': "/home/root/project/hsi-denoising/GDM/results/msfa/",
        'spatial_kwargs' : {
            'model_config': "/home/root/project/hsi-denoising/GDM/config/model_config.yaml",
            'model_path': "/home/root/project/posterior_diffusion/models/hsi256/model040000.pt",
        },
        'spectral_kwargs': {
            "model_config": "/home/root/project/hsi-denoising/GDM/config/spectral_dm.yaml",
        },
        'l1': 1,
        'l2': 0.5,
        'base_samples': "/home/root/dataset/cave/fixedga75/",
        'k': 5,
        'k_svd': 5,
    }
    main()