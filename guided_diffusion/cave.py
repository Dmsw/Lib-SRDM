import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
from mpi4py import MPI


def load_cave(
    *,
    data_root,
    batch_size,
    crop_size=256,
    mode="train",
    deterministic=False,
):
    if mode == "train":
        dataset = CAVEBase(stride=8, data_root=data_root, resolution=crop_size, arg=True, shard_id=MPI.COMM_WORLD.Get_rank(), num_shards=MPI.COMM_WORLD.Get_size())
    elif mode == "test":
        dataset = CAVEBase(data_root=data_root, resolution=512, arg=False, stride=1)
    else:
        raise ValueError(f"Unknown mode {mode}")
    
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
        )
    while True:
        yield from loader


class CAVEBase(Dataset):
    def __init__(self,
                 resolution=256,
                 data_root="/home/root/project/RGB2HSI/PaDM/dataset/cave/test",
                 arg=True,
                 stride=64,
                 verbose=False,
                 shard_id=0,
                 num_shards=1,
                 ):
        super().__init__()
        self.verbose = verbose
        self.root_path = data_root
        self.files = os.listdir(data_root)[shard_id::num_shards]
        self.hypers = []
        self.rgbs = []
        self.arg = arg
        h,w = 512,512  # img shape
        self.stride = stride
        self.crop_size = resolution
        self.patch_per_line = (w-self.crop_size)//stride+1
        self.patch_per_colum = (h-self.crop_size)//stride+1
        self.patch_per_img = self.patch_per_line*self.patch_per_colum
        for f in self.files:
            path = os.path.join(self.root_path, f)
            data = np.load(path)
            hsi = data["hsi"] * 2 - 1
            rgb = data["rgb"] * 2 - 1
            self.hypers.append(hsi)
            self.rgbs.append(rgb)
        self.resolution = resolution
        self.arg = arg
        self.img_num = len(self.hypers)
        print(f"load {len(self.files)} files")

    def arguement(self, img, rotTimes, vFlip, hFlip):
        # Random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(1, 2))
        # Random vertical Flip
        for j in range(vFlip):
            img = img[:, :, ::-1].copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()
        return img

    def __len__(self):
        return self.patch_per_img*self.img_num

    def __getitem__(self, idx):
        stride = self.stride
        crop_size = self.crop_size
        img_idx, patch_idx = idx//self.patch_per_img, idx%self.patch_per_img
        h_idx, w_idx = patch_idx//self.patch_per_line, patch_idx%self.patch_per_line
        bgr = self.rgbs[img_idx].astype(np.float32)
        hyper = self.hypers[img_idx]
        bgr = bgr[:,h_idx*stride:h_idx*stride+crop_size, w_idx*stride:w_idx*stride+crop_size]
        hyper = hyper[:, h_idx * stride:h_idx * stride + crop_size,w_idx * stride:w_idx * stride + crop_size]
        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)
        if self.arg:
            bgr = self.arguement(bgr, rotTimes, vFlip, hFlip)
            hyper = self.arguement(hyper, rotTimes, vFlip, hFlip)
        return {"input": np.ascontiguousarray(hyper)}, {}


class CAVETrain(CAVEBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class CAVEValidation(CAVEBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
