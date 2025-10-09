import pandas as pd
import numpy as np
import torch as th
from typing import Union
import os
import cv2
from srf_tools import SRFTool


class Illuminant:
    def __init__(self,
                 start_bands=400,
                 end_bands=700,
                 inter_bands=10,
                 light_type:Union[str, int]=0,
                 file="CIE_illum_FLs_1nm.csv",
                 root="/home/root/dataset/illuminant/LED-Acid_Fly_Research/"):
        self.filename = file
        self.fullpath = os.path.join(root, file)
        df = pd.read_csv(self.fullpath, delimiter=",", header=None, skiprows=1, index_col=0)
        new_bands = np.arange(start_bands, end_bands+inter_bands+1, inter_bands)
        df['group'] = pd.cut(df.index, bins=new_bands, right=False, labels=new_bands[:-1])
        data = df.groupby('group', observed=False).mean()
        if isinstance(light_type, str):
            assert light_type in data.columns, f"{light_type} is not in the type {data.columns}"
            illuminant = np.array(data[light_type]).astype(np.float32).flatten()
        elif isinstance(light_type, int):
            illuminant = np.array(data.loc[:, light_type]).astype(np.float32).flatten()
        illuminant = illuminant / illuminant.max()
        illuminant = illuminant.reshape([1, -1, 1, 1])
        self.illuminant = illuminant.astype(np.float32)
        assert not np.any(np.isnan(illuminant)), illuminant

    def lighting_srf(self, srf: SRFTool):
        P = srf.P   # 3 x B
        bar_P = P * self.illuminant.reshape([1, -1])
        new_srf = SRFTool()
        new_srf.set_srf(bar_P)
        return new_srf
        
    def lighting(self, hsi):
        if isinstance(hsi, th.Tensor):
            illuminant = th.from_numpy(self.illuminant).to(hsi.device)
        else:
            illuminant = self.illuminant
        return hsi * illuminant

    def smooth_illuminant(self, threshold=0.1):
        data = self.illuminant.flatten().copy()
        mask = data < threshold
        indices = np.arange(len(data))
        
        valid_indices = indices[~mask]
        valid_values = data[~mask]
        
        interpolated_values = np.interp(indices, valid_indices, valid_values)
        
        return interpolated_values.reshape(self.illuminant.shape)
    
    def delighting(self, hsi, threshold=0.1):
        if isinstance(hsi, th.Tensor):
            hsi_ = hsi.cpu().numpy()
        else:
            hsi_ = hsi
        band_idx = self.illuminant.flatten() < threshold
        hsi_ = hsi_ / self.illuminant
        hsi_ = self.interpolate(hsi_, band_idx)
        
        if isinstance(hsi, th.Tensor):
            hsi_ = th.from_numpy(hsi_).to(hsi.device)
        return hsi_
    
    def interpolate(self, hsi, band_idx):
        N, B, H, W = hsi.shape
        X = hsi.transpose(1, 0, 2, 3).reshape(B, -1)
        indices = np.arange(B)
        for i in range(X.shape[1]):
            X[:, i] = np.interp(indices, indices[~band_idx], X[~band_idx, i])
        X = X.reshape(B, N, H, W).transpose(1, 0, 2, 3)
        return X
    
    def downsampling_delighting(self, rgb, srf):
        light_rgb = th.from_numpy(srf.gen_rgb_numpy(self.illuminant[0], max_value=1)).to(rgb.device)
        return rgb / light_rgb
    
    def downsampling_lighting(self, rgb, srf):
        light_rgb = th.from_numpy(srf.gen_rgb_numpy(self.illuminant[0], max_value=1)).to(rgb.device)
        return rgb * light_rgb
    