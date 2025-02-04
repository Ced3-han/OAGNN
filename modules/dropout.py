import torch
from torch import nn

from .common import ScalarVector


class VectorDropout(nn.Module):
    
    def __init__(self, drop_rate):
        super().__init__()
        self.drop_rate = drop_rate
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        '''
        Args:
            x:  (*, vec_dim)
        '''
        device = self.dummy_param.device
        if not self.training:
            return x
        mask = torch.bernoulli(
            (1 - self.drop_rate) * torch.ones(x.shape[:-1], device=device)
        ).unsqueeze(-1)
        x = mask * x / (1 - self.drop_rate)
        return x


class SVDropout(nn.Module):
    
    def __init__(self, drop_rate):
        super().__init__()
        self.sdropout = nn.Dropout(drop_rate)
        self.vdropout = VectorDropout(drop_rate)

    def forward(self, x: ScalarVector) -> ScalarVector:
        return ScalarVector(
            s = self.sdropout(x.s),
            v = self.vdropout(x.v),
        )
