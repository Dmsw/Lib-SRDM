import torch
from guided_diffusion.dist_util import dev
import numpy as np
import cv2 as cv
from utils import tensor_2_mode_product
import torch.nn.functional as F


def rgb2hsi(x_t, ref, bar_alpha_t, var, pred_var, P):
    device = x_t.device
    over_sqrt_bar_alpha_t = (1 / np.sqrt(bar_alpha_t))
    assert True, f"{over_sqrt_bar_alpha_t} {x_t.shape} {ref.shape} {var.shape} {P.shape}"
    coeff = P.T @ torch.inverse(var + P @ P.T * pred_var)
    return over_sqrt_bar_alpha_t * tensor_2_mode_product(ref - over_sqrt_bar_alpha_t * tensor_2_mode_product(x_t, P.T), coeff.T)
