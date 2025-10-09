import torch
from guided_diffusion.dist_util import dev
import numpy as np
import cv2 as cv
from utils import tensor_2_mode_product
import torch.nn.functional as F


def denoising(x_t, ref, bar_alpha_t, var, pred_var):
    device = x_t.device
    over_sqrt_bar_alpha_t = (1 / np.sqrt(bar_alpha_t))
    coeff = bar_alpha_t * torch.inverse(var * bar_alpha_t + (1 - bar_alpha_t) * torch.eye(var.shape[0], device=device))
    return over_sqrt_bar_alpha_t * tensor_2_mode_product(ref - over_sqrt_bar_alpha_t * x_t, coeff)


def rgb2hsi(x_t, ref, bar_alpha_t, var, pred_var, P):
    device = x_t.device
    over_sqrt_bar_alpha_t = (1 / np.sqrt(bar_alpha_t))
    assert True, f"{over_sqrt_bar_alpha_t} {x_t.shape} {ref.shape} {var.shape} {P.shape}"
    coeff = P.T @ torch.inverse(var + P @ P.T * pred_var)
    return over_sqrt_bar_alpha_t * tensor_2_mode_product(ref - over_sqrt_bar_alpha_t * tensor_2_mode_product(x_t, P.T), coeff.T)


def msfa2hsi(x_t, ref, bar_alpha_t, var, pred_var, msfa):
    device = x_t.device
    over_sqrt_bar_alpha_t = (1 / np.sqrt(bar_alpha_t))
    assert True, f"{over_sqrt_bar_alpha_t} {x_t.shape} {ref.shape} {var.shape} {P.shape}"
    P = msfa.msfa / (torch.sum(msfa.msfa**2, dim=0) * pred_var + var)
    
    return over_sqrt_bar_alpha_t * msfa.pseudo_inverse(ref - msfa(over_sqrt_bar_alpha_t * x_t), P)


def msfa2hsi_with_wb(x_t, ref, bar_alpha_t, var, pred_var, msfa, wb, wb_cov):
    g1 = msfa2hsi(x_t, ref, bar_alpha_t, var, pred_var, msfa)
    g2 = rgb2hsi(x_t, wb, bar_alpha_t, wb_cov, pred_var, msfa.P().T)
    return g1 * 0.6 + g2 * 0.4


if __name__ == "__main__":
    from msfa import MSFAModel
    x_t = torch.rand(1, 31, 256, 256)
    bar_alpha_t = 0.5
    pred_var = 0.1
    P = torch.rand(3, 31)
    msfa = MSFAModel(msfa_file="/home/root/project/stdm/dataset/MSFA16.npy")
    print(denoising(x_t, torch.rand(1, 31, 256, 256), bar_alpha_t, torch.randn(31, 31), pred_var).shape)
    print(rgb2hsi(x_t, torch.rand(1, 3, 256, 256), bar_alpha_t, torch.randn(3, 3), pred_var, P).shape)
    print(msfa2hsi(x_t, torch.rand(1, 1, 256, 256), bar_alpha_t, torch.randn(1, 1), pred_var, msfa).shape)
