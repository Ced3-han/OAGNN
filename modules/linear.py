import copy
import math
import torch
from torch import nn
from torch.nn import functional as F

from .geometric import local_to_global, global_to_local
from .common import ScalarVector


class VectorLinear(nn.Module):

    def __init__(self, in_dims: int, out_dims: int, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.weight = nn.Parameter(torch.empty([out_dims, in_dims, 3], **factory_kwargs), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def extra_repr(self) -> str:
        return 'in_dims={}, out_dims={}'.format(self.in_dims, self.out_dims)

    def _forward_scalar(self, input):
        """
        Args:
            input:  (*, in_dims)
        Returns:
            (*, out_dims, 3)
        """
        u, v, w = torch.unbind(self.weight, dim=-1) # [(out_dims, in_dims)]_3
        a = F.linear(input, weight=u, bias=None).unsqueeze(-1)  # (*, out_dims, 1)
        b = F.linear(input, weight=v, bias=None).unsqueeze(-1)
        c = F.linear(input, weight=w, bias=None).unsqueeze(-1)
        out = torch.cat([a, b, c], dim=-1)  # (*, out_dims, 3)
        return out

    def _forward_vector_dot(self, input):
        """
        Args:
            input:  (*, in_dims, 3)
        Returns:
            (*, out_dims)
        """
        u, v, w = torch.unbind(self.weight, dim=-1) # [(out_dims, in_dims)]_3
        x, y, z = torch.unbind(input, dim=-1)       # [(*, in_dims)]_3
        a = F.linear(x, weight=u, bias=None)    # (*, out_dims)
        b = F.linear(y, weight=v, bias=None)
        c = F.linear(z, weight=w, bias=None)
        out = a + b + c
        return out

    def _forward_vector_cross(self, input):
        """
        Args:
            input:  (*, in_dims, 3)
        Returns:
            (*, out_dims, 3)
        """
        # Reshape input
        input = input.unsqueeze(-3)     # (*, 1, in_dims, 3)
        in_size = list(input.size())
        in_size[-3] = self.out_dims
        input = input.expand(in_size)   # (*, out_dims, in_dims, 3)
        # Reshape weight
        w_size = ([1] * (input.dim() - 3)) + [self.out_dims, self.in_dims, 3]
        weight = self.weight.reshape(w_size).expand_as(input)   # (*, out_dims, in_dims, 3)
        # Cross product
        out = torch.cross(weight, input, dim=-1).sum(-2)    # (*, out_dims, 3)
        return out

    def forward(self, input: torch.Tensor, input_type, mult_op='dot') -> torch.Tensor:
        """
        Args:
            input:  (*, in_dims) for scalar features and,
                    (*, in_dims, 3) for vector features.
            input_type: `scalar` or `vector`.
            mult_op:    Vector-vector mult. operator, either `dot` or `cross`.
        Returns:
            (*, out_dims, 3) for scalar-vector product and vector-vector cross product.
            (*, out_dims) for vector-vector dot product.
        """
        assert input_type in ('scalar', 'vector')
        assert mult_op in ('dot', 'cross')
        if input_type == 'scalar':
            return self._forward_scalar(input)
        elif input_type == 'vector' and mult_op == 'dot':
            return self._forward_vector_dot(input)
        elif input_type == 'vector' and mult_op == 'cross':
            return self._forward_vector_cross(input)


def vector_input_scalar_linear(lin: nn.Linear, x: torch.Tensor):
    """
    Args:
        lin:    Scalar `Linear` module.
        x:      Input tensor, (*, in_dims, 3).
    """
    assert lin.bias is None
    x = x.transpose(-1, -2)       # (*, 3, in_dims)
    y = lin(x).transpose(-1, -2)  # (*, 3, out_dims) -> (*, out_dims, 3)
    return y


def rotate_apply(layer, x, rot=None):
    if rot is not None:
        # Warning: Don't directly overwrite x.v.
        x = ScalarVector(s=x.s, v=global_to_local(x.v, rot))
    y = layer(x)
    if (rot is not None) and (y.v is not None):
        y.v = local_to_global(y.v, rot)
    return y


class SVLinear(nn.Module):

    def __init__(self, in_dims: tuple, out_dims: tuple, scalar_bias=True, hidden_dims=None, share_dot_cross=False):
        super().__init__()
        self.in_s, self.in_v = in_dims
        self.out_s, self.out_v = out_dims
        hidden_dims = hidden_dims if hidden_dims is not None else copy.deepcopy(out_dims)
        if hidden_dims[1] == 0: hidden_dims = (hidden_dims[0], in_dims[1])
        self.hid_s, self.hid_v = hidden_dims

        # Linear layers for scalar input
        self.lin_s_s = nn.Linear(self.in_s, self.hid_s, bias=scalar_bias)
        self.lin_s_v = VectorLinear(self.in_s, self.hid_s)

        # Linear layers for vector input
        self.lin_v_s = nn.Linear(self.in_v, self.hid_v, bias=False)
        self.lin_v_dot = VectorLinear(self.in_v, self.hid_v)
        self.lin_v_cro = self.lin_v_dot if share_dot_cross else VectorLinear(self.in_v, self.hid_v)

        # Layers for mixing channels
        if self.out_s > 0: self.lin_out_s = nn.Linear(self.hid_s+self.hid_v, self.out_s, bias=scalar_bias)
        if self.out_v > 0: self.lin_out_v = nn.Linear(self.hid_s+2*self.hid_v, self.out_v, bias=False)    # Linear combination
    
    def forward(self, x: ScalarVector):
        # Scalar input -> Scalar hidden, Vector hidden
        s_s_s = self.lin_s_s(x.s)   # (*, hid_s)
        s_v_v = self.lin_s_v(x.s, 'scalar')   # (*, hid_s, 3)

        # Vector input -> Vector hidden (s*v), Vector hidden (cross), Scalar hidden (dot)
        v_s_v = vector_input_scalar_linear(self.lin_v_s, x.v)   # (*, hid_v, 3)
        v_v_v = self.lin_v_cro(x.v, 'vector', 'cross')  # (*, hid_v, 3)
        v_v_s = self.lin_v_dot(x.v, 'vector', 'dot')    # (*, hid_v)

        h_s = torch.cat([s_s_s, v_v_s], dim=-1)         # (*, hid_s+hid_v)
        h_v = torch.cat([s_v_v, v_s_v, v_v_v], dim=-2)  # (*, hid_s+2*hid_v, 3)

        out_s = self.lin_out_s(h_s) if self.out_s > 0 else None
        out_v = vector_input_scalar_linear(self.lin_out_v, h_v) if self.out_v > 0 else None
        out = ScalarVector(
            s = out_s,
            v = out_v,
        )
        return out


if __name__ == '__main__':
    bsize, in_dims, out_dims = 4, 8, 16

    # SVLinear
    x = ScalarVector(
        s = torch.randn([bsize, in_dims]),
        v = torch.randn([bsize, in_dims, 3])
    )
    lin = SVLinear((in_dims, in_dims), (out_dims, out_dims))
    y = lin(x)
    print(x)
    print(y)

    # Vector Weight
    vlin = VectorLinear(in_dims, out_dims)
    weight = vlin.weight    # (out_dims, in_dims, 3)
    v_in = torch.randn([bsize, in_dims, 3])
    s_in = torch.randn([bsize, in_dims])

    s_out_ref = torch.zeros([bsize, out_dims, 3])
    v_dot_out_ref = torch.zeros([bsize, out_dims])
    v_cro_out_ref = torch.zeros([bsize, out_dims, 3])
    for b in range(bsize):
        for o in range(out_dims):
            for i in range(in_dims):
                # Scalar input
                s_out_ref[b, o] += weight[o, i] * s_in[b, i]

                # Vector input, dot product
                v_dot_out_ref[b, o] += torch.dot(weight[o, i], v_in[b, i])

                # Vector input, cross product
                v_cro_out_ref[b, o] += torch.cross(weight[o, i], v_in[b, i])


    s_out = vlin(s_in, 'scalar')
    v_dot_out = vlin(v_in, 'vector', 'dot')
    v_cro_out = vlin(v_in, 'vector', 'cross')

    assert torch.allclose(s_out_ref, s_out, atol=1e-6)
    assert torch.allclose(v_dot_out_ref, v_dot_out, atol=1e-6)
    assert torch.allclose(v_cro_out_ref, v_cro_out, atol=1e-6)
    print('[Passed] VectorLinear.')

    # Vector Input, Scalar Weight
    slin = nn.Linear(in_dims, out_dims, None)
    weight = slin.weight    # (in_dims, out_dims)
    v_in = torch.randn([bsize, in_dims, 3])

    v_out_ref = torch.zeros([bsize, out_dims, 3])
    for b in range(bsize):
        for o in range(out_dims):
            for i in range(in_dims):
                v_out_ref[b, o] += weight[o, i] * v_in[b, i]

    v_out = vector_input_scalar_linear(slin, v_in)
    assert torch.allclose(v_out_ref, v_out, atol=1e-6)
    print('[Passed] vector_input_scalar_weight.')

