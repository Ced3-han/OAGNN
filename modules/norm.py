import torch
from torch import nn

from .common import ScalarVector, safe_norm


class SVLayerNorm(nn.Module):

    def __init__(self, dims):
        super().__init__()
        self.s, self.v = dims
        self.scalar_norm = nn.LayerNorm(self.s)
        
    def forward(self, x: ScalarVector) -> ScalarVector:
        s, v = x.s, x.v
        vn = safe_norm(v, dim=-1, keepdim=True, sqrt=False)
        vn = torch.sqrt(torch.mean(vn, dim=-2, keepdim=True))
        return ScalarVector(
            s = self.scalar_norm(s),
            v = v / vn,
        )
