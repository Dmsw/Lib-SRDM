import torch as th
import numpy as np
from torchvision.utils import save_image
import torch.nn.functional as F
from wb_estimator import WBEstimator


class MSFAModel(th.nn.Module):
    def __init__(self, 
                 bands=31,
                 size=[4, 4],
                 msfa_file="",
                 trainable=False,
                ):
        super(MSFAModel, self).__init__()
        shape = [bands, *size]
        if len(msfa_file) > 0:
            with open(msfa_file, 'rb') as f:
                msfa = np.load(f).T
                if msfa.shape != shape:
                    print(f"Warning: MSFA shape {msfa.shape} != expected shape {shape}. Reshaping.")
                    msfa = msfa.reshape(shape)
        else:
            if not trainable:
                print("Warning: MSFA is not trainable and no file is provided. Random initialization will be used.")
            msfa = np.random.rand(*shape)
        msfa /= np.sum(msfa, axis=0, keepdims=True)
        self.size = size
        self.bands = bands
        self.trainable = trainable
        self.msfa = th.nn.Parameter(th.from_numpy(msfa).float(), requires_grad=self.trainable)
        self.wb = WBEstimator([self.bands, self.size[0] * self.size[1]])
    
    def forward(self, hsi, requires_grad=False):
        if hsi.ndim == 3:
            ndim = 3
            hsi = hsi[None]
        else:
            ndim = 4
        N, B, H, W = hsi.shape
        assert B == self.bands, f"Input bands {B} != MSFA bands {self.bands}"
        raw_mossaic = th.zeros([N, 1, H, W], device=hsi.device, dtype=hsi.dtype, requires_grad=requires_grad)
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                raw_mossaic[:, :, i::self.size[0], j::self.size[1]] += th.sum(
                    hsi[:, :, i::self.size[0], j::self.size[1]] * self.msfa[:, i, j].view(1, B, 1, 1), dim=1, keepdim=True
                )
        
        return raw_mossaic[0] if ndim == 3 else raw_mossaic
        
    def pseudo_inverse(self, raw_mossaic, P):
        N, _, H, W = raw_mossaic.shape
        hsi = th.zeros([N, self.bands, H, W], device=raw_mossaic.device, dtype=raw_mossaic.dtype)
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                hsi[:, :, i::self.size[0], j::self.size[1]] = raw_mossaic[:, :, i::self.size[0], j::self.size[1]] * P[:, i, j].view(1, self.bands, 1, 1)
        
        return hsi
    
    def P(self):
        P = th.zeros([self.bands, self.size[0] * self.size[1]], device=self.msfa.device, dtype=self.msfa.dtype)
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                P[:, i*self.size[0]+j] = self.msfa[:, i, j]
        return P
    
    def raw2hard(self, raw):
        b, c, h, w = raw.size()
        assert c == 1, 'Only single channel images are supported'
        hard = th.zeros([b, self.size[0]*self.size[1], h, w], device=raw.device)
        for i in range(0, self.size[0]):
            for j in range(0, self.size[1]):
                hard[:, i*self.size[0]+j, i::self.size[0], j::self.size[1]] = raw[:, 0, i::self.size[0], j::self.size[1]]
        return hard
    
    def wb_filter(self, raw):
        cube = self.raw2hard(raw)
        return self.wb(cube)
    
    def plot_msfa(self, filename):
        import matplotlib.pyplot as plt
        
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                plt.plot(self.msfa[:, i, j])

        plt.savefig(filename)

if __name__ == "__main__":
    msfa = MSFAModel(msfa_file="/home/root/project/stdm/dataset/MSFA.npy")
    hsi = np.load("/home/root/dataset/cave/cave/fake_and_real_beers_ms/fake_and_real_beers_ms.npz")["hsi"]
    rmi = msfa(th.from_numpy(hsi[None]))
    phsi = msfa.pseudo_inverse(rmi)
    save_image(rmi, "test.png")
    msfa.plot_msfa("msfa.png")