import torch as th
import torch.nn as nn
import numpy as np


class PriorModels(nn.Module):
    def __init__(self, libs, masks, factor, bar_alpha, **kwargs):
        super().__init__()
        self.tags = libs.keys()
        self.n_models = len(self.tags)
        models = {}
        for t in self.tags:
            A = libs[t] * 2 - 1
            mask = masks[t]
            models[t] = PriorModel(A, mask, bar_alpha)

        self.models = nn.ModuleDict(models)
        self.factor = nn.Parameter(th.from_numpy(factor).float())
    
    def forward(self, x, t):
        output = th.zeros_like(x)
        for tag in self.tags:
            output += self.models[tag](x, t)
        
        return output / (self.factor + 1e-6)

    def convert_to_fp16(self):
        for m in self.models.values():
            m.convert_to_fp16(self)
        
        self.factor.half()


class PriorModel(nn.Module):
    def __init__(self, A, mask, bar_alpha, **kwargs):
        super().__init__()
        self.mask = nn.Parameter(th.from_numpy(mask).bool(), requires_grad=False)
        self.A = nn.Parameter(th.from_numpy(A).unsqueeze(0), requires_grad=False)
        self.bar_alpha = nn.Parameter(th.from_numpy(bar_alpha).float(), requires_grad=False)

    def forward(self, x, t, **kwargs):
        t = t.long()[0]
        output = th.zeros_like(x)
        x = x[:, :, self.mask] # shape B, C, N
        B, C, N = x.shape
        assert B == 1
        x = x.permute(0, 2, 1).reshape(N*B, 1, C)
        assert True, f"{x.shape} {self.A.shape} {self.bar_alpha.shape}"
        distance = (-th.norm(th.sqrt(self.bar_alpha[t]) * self.A - x, dim=2)**2 / (2*(1-self.bar_alpha[t])))
        distance = distance - th.max(distance, dim=1, keepdim=True)[0]
        exp = th.exp(distance)
        res = exp @ self.A.squeeze() / th.sum(exp, dim=1, keepdim=True)
        res = res.reshape(B, N, C).permute(0, 2, 1)

        output[:, :, self.mask] = res
        return output

    def convert_to_fp16(self):
        self.A.data = self.A.data.half()
        