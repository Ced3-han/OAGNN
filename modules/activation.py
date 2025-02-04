import torch
from torch import nn
from torch.nn import functional as F

from .common import ScalarVector, safe_norm


def _get_scalar_activation(name):
    if name is None:
        return nn.Identity()
    elif name == 'sigmoid':
        return torch.sigmoid
    else:
        return getattr(F, name)


def _get_vector_activation(cfg):
    if cfg is None:
        return nn.Identity()
    elif cfg[0] == 'scale':
        return VectorScaling(_get_scalar_activation(cfg[1]))
    elif cfg[0] == 'project':
        return VectorProjection(cfg[1])
    else:
        raise ValueError('Unknown vector activation class: %s' % cfg[0])
    

class VectorProjection(nn.Module):

    def __init__(self, n_dims):
        super().__init__()
        self.vecs = nn.Parameter(torch.randn([n_dims, 3]), requires_grad=True)
        nn.utils.weight_norm(self, name='vecs', dim=1)

    def forward(self, x):
        """
        Args:
            x:  Vector input, (*, n_dims, 3).
        """
        dot_prod = (self.vecs_v * x).sum(dim=-1, keepdim=True)    # (*, n_dims, 1)
        x_proj = x - dot_prod * self.vecs_v # (*, n_dims, 3)
        out = torch.where(dot_prod >= 0, x, x_proj)
        return out


class VectorScaling(nn.Module):

    def __init__(self, func=torch.sigmoid):
        super().__init__()
        self.func = func

    def forward(self, x):
        """
        Args:
            x:  Vector input, (*, n_dims, 3).
        """
        s = self.func(safe_norm(x, dim=-1, keepdim=True))
        return s * x


class SVActivation(nn.Module):

    def __init__(self, s_act, v_act):
        super().__init__()
        self.s_act = s_act
        self.v_act = v_act

    @classmethod
    def from_args(cls, scalar_act, vector_act):
        s_act = _get_scalar_activation(scalar_act)
        v_act = _get_vector_activation(vector_act)
        return cls(s_act, v_act)

    def forward(self, x: ScalarVector) -> ScalarVector:
        return ScalarVector(
            s = self.s_act(x.s) if x.s is not None else None,
            v = self.v_act(x.v) if x.v is not None else None,
        )
