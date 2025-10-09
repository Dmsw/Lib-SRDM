import torch 
from torch import nn
import torch.nn.functional as F
import numpy as np


def get_WB_filter(size):
    """make a 2D weight bilinear kernel suitable for interpolation"""
    ligne = []
    colonne = []
    half = (size+1) // 2
    for i in range(size):
        if (i + 1) <= half:
            ligne.append(i + 1)
            colonne.append(i + 1)
        else:
            ligne.append(ligne[i - 1] - 1.0)
            colonne.append(colonne[i - 1] - 1.0)
    BilinearFilter = np.zeros(size * size)
    for i in range(size):
        for j in range(size):
            BilinearFilter[(j + i * size)] = (ligne[i] * colonne[j] / (half**2))
    filter_WB = np.reshape(BilinearFilter, (size, size))
    filter_WB = torch.from_numpy(filter_WB).float()
    filter_WB = filter_WB.view(1, 1, size, size).repeat(16, 1, 1, 1)
    return filter_WB


class WBEstimator(nn.Module):
    def __init__(self, shape) -> None:
        super().__init__()
        bands, size = shape
        assert size == 16
        self.filter_WB = nn.Conv2d(16, 16, kernel_size=7, stride=1, padding=3, bias=False, groups=16)
        self.initial_estimator()
    
    def initial_estimator(self):
        self.filter_WB.weight.data = get_WB_filter(7)
        self.filter_WB.weight.requires_grad = False
    
    @torch.no_grad()
    def forward(self, cube):
        est_cube = self.filter_WB(cube)
        return est_cube
                