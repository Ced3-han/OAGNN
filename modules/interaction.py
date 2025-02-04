import torch
from torch import nn

from .common import ScalarVector
from .linear import VectorLinear


class SVInteraction(nn.Module):

    def __init__(self, scalar_dims, vector_dims):
        super().__init__()
        self.s_dims, self.v_dims = scalar_dims, vector_dims
        self.s_to_v = nn.Linear(scalar_dims, vector_dims)
        self.v_to_s = VectorLinear(vector_dims, scalar_dims)

    def forward(self, x: ScalarVector) -> ScalarVector:
        s, v = x.s, x.v     # (*, s_dims), (*, v_dims, 3)
        coef_v = self.s_to_v(s).unsqueeze(-1)       # (*, v_dims, 1)
        bias_s = self.v_to_s(v, 'vector', 'dot')    # (*, s_dims)
        return ScalarVector(
            s = bias_s + s,
            v = coef_v * v,
        )
