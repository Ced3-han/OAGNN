import functools
import torch
from torch import nn
from torch.nn import functional as F

from .common import ScalarVector, safe_norm
from .linear import SVLinear
from .interaction import SVInteraction
from .activation import SVActivation, _get_scalar_activation


class SVPerceptron(nn.Module):

    def __init__(
        self, in_dims: tuple, out_dims: tuple, scalar_bias=True, hidden_dims=None, 
        scalar_act='relu', vector_act=['scale', 'sigmoid'],
        share_dot_cross=False, interaction=True,
    ):
        super().__init__()
        self.linear = SVLinear(in_dims, out_dims, scalar_bias, hidden_dims, share_dot_cross)
        self.interact = SVInteraction(*out_dims) if interaction else nn.Identity()
        if vector_act is not None and vector_act[0] == 'project':
            vector_act[1] = out_dims[1]
        self.act = SVActivation.from_args(scalar_act, vector_act)

    def forward(self, x: ScalarVector) -> ScalarVector:
        x = self.linear(x)
        if x.v is not None:
            x = self.interact(x)
        return self.act(x)


class GVP(nn.Module):
    '''
    Geometric Vector Perceptron. See manuscript and README.md
    for more details.
    
    :param in_dims: tuple (n_scalar, n_vector)
    :param out_dims: tuple (n_scalar, n_vector)
    :param h_dim: intermediate number of vector channels, optional
    :param activations: tuple of functions (scalar_act, vector_act)
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    '''
    def __init__(self, in_dims, out_dims, h_dim=None,
                 activations=(F.relu, torch.sigmoid), vector_gate=True):
        super(GVP, self).__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.vector_gate = vector_gate
        if self.vi: 
            self.h_dim = h_dim or max(self.vi, self.vo) 
            self.wh = nn.Linear(self.vi, self.h_dim, bias=False)
            self.ws = nn.Linear(self.h_dim + self.si, self.so)
            if self.vo:
                self.wv = nn.Linear(self.h_dim, self.vo, bias=False)
                if self.vector_gate: self.wsv = nn.Linear(self.so, self.vo)
        else:
            self.ws = nn.Linear(self.si, self.so)
        
        self.scalar_act, self.vector_act = activations
        self.dummy_param = nn.Parameter(torch.empty(0))
        
    def forward(self, x: ScalarVector) -> ScalarVector:
        x = (x.s, x.v)
        if self.vi:
            s, v = x
            v = torch.transpose(v, -1, -2)
            vh = self.wh(v)    
            vn = safe_norm(vh, dim=-2)
            s = self.ws(torch.cat([s, vn], -1))
            if self.vo: 
                v = self.wv(vh) 
                v = torch.transpose(v, -1, -2)
                if self.vector_gate: 
                    if self.vector_act:
                        gate = self.wsv(self.vector_act(s))
                    else:
                        gate = self.wsv(s)
                    v = v * torch.sigmoid(gate).unsqueeze(-1)
                elif self.vector_act:
                    v = v * self.vector_act(safe_norm(v, dim=-1, keepdim=True))
        else:
            s = self.ws(x)
            if self.vo:
                v = torch.zeros(s.shape[0], self.vo, 3, device=self.dummy_param.device)
        if self.scalar_act:
            s = self.scalar_act(s)
        
        return ScalarVector(s=s, v=v) if self.vo else ScalarVector(s=s, v=None)



def VectorPerceptron(in_dims, out_dims, mode='svp', scalar_act='relu', vector_act=['scale', 'sigmoid'], **kwargs):
    assert mode in ('svp', 'gvp')
    if mode == 'svp':
        return SVPerceptron(in_dims=in_dims, out_dims=out_dims, scalar_act=scalar_act, vector_act=vector_act)
    elif mode == 'gvp':
        if vector_act is not None:
            assert vector_act[0] == 'scale', 'GVP only supports scaling activation.'
        act_s = _get_scalar_activation(scalar_act) if scalar_act is not None else None
        act_v = _get_scalar_activation(vector_act[1]) if vector_act is not None else None
        return GVP(in_dims=in_dims, out_dims=out_dims, activations=(act_s, act_v))
    

class VectorMLP(nn.Module):
    
    def __init__(
        self, mode, in_dims: tuple, out_dims: tuple, n_layers, hidden_dims=None, 
        scalar_act='relu', vector_act=['scale', 'sigmoid'],
    ):
        super().__init__()

        assert mode in ('svp', 'gvp')
        
        # Helper functions for creating fully connected layers
        layer_act = functools.partial(VectorPerceptron, mode=mode, scalar_act=scalar_act, vector_act=vector_act)
        layer_lin = functools.partial(VectorPerceptron, mode=mode, scalar_act=None, vector_act=None)

        if hidden_dims is None:
            hidden_dims = out_dims
        module_list = []
        if n_layers == 1:
            module_list.append(layer_lin(in_dims=in_dims, out_dims=out_dims))
        else:
            module_list.append(layer_act(in_dims=in_dims, out_dims=hidden_dims))
            for _ in range(1, n_layers-1):
                module_list.append(layer_act(in_dims=hidden_dims, out_dims=hidden_dims))
            module_list.append(layer_lin(in_dims=hidden_dims, out_dims=out_dims))

        self.mlp = nn.Sequential(*module_list)

    def forward(self, x):
        return self.mlp(x)


if __name__ == '__main__':
    from .geometric import orthogonalize_matrix, apply_rotation, apply_inverse_rotation, global_to_local, local_to_global

    n = 3
    in_s, in_v = (2, 2)
    out_s, out_v = (1, 1)

    layer = VectorMLP('svp', (in_s, in_v), (out_s, out_v), n_layers=3)

    s = torch.randn([n, in_s]) * 10
    v = torch.randn([n, in_v, 3]) * 10
    x = ScalarVector(s=s, v=v)
    out_ref = layer(x)
    print(out_ref)

    rot = orthogonalize_matrix(torch.randn([1, 3, 3])).repeat(n, 1, 1)
    v_rot = apply_rotation(rot, v)

    x = ScalarVector(s=s, v=global_to_local(v_rot, rot))
    out_rot = layer(x)
    print(out_rot)

    print((out_ref.v - out_rot.v).abs().max())
