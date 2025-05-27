import functools
from modules.linear import rotate_apply
import torch
from torch import nn
from torch_geometric.nn import MessagePassing

from .geometric import local_to_global, global_to_local
from .common import ScalarVector
from .perceptron import VectorMLP
from .dropout import SVDropout
from .norm import SVLayerNorm


class SVGraphConv(MessagePassing):

    def __init__(self, 
        in_dims, out_dims, edge_dims, mlp_mode='svp', n_layers=3, 
        scalar_act='relu', vector_act=['scale', 'sigmoid'], 
        share_dot_cross=False, sv_interaction=True,
        aggr='mean', 
    ):
        super().__init__(aggr=aggr, flow='target_to_source')    # i <- j
        self.in_s, self.in_v = in_dims
        self.out_s, self.out_v = out_dims
        self.edge_s, self.edge_v = edge_dims

        self.message_func = VectorMLP(
            mode = mlp_mode,
            in_dims = (2*self.in_s + self.edge_s, 2*self.in_v + self.edge_v),
            out_dims = (self.out_s, self.out_v),
            n_layers = n_layers,
            scalar_act = scalar_act,
            vector_act = vector_act,
        )

        self.dummy = nn.Parameter(torch.empty(0))
    
    def forward(self, x, edge_index, edge_attr, rot=None):
        """
        Args:
            x:  ScalarVector, {(N, in_s), (N, in_v, 3)}.
            edge_index: (2, E)
            edge_attr:  ScalarVector, {(E, in_s), (E, in_v, 3)}.
            rot:    Rigid rotation matrices, (N, 3, 3).
        Returns:
            {(E, out_s), (E, out_v, 3)}
        """
        if rot is None:
            rot = torch.eye(3).to(self.dummy.device)
            rot = rot.unsqueeze_(0).repeat(x.v.size(0), 1, 1)  # (N, 3, 3)

        out = self.propagate(
            edge_index = edge_index,
            s = x.s,    # (N, in_s)
            v = x.v.reshape([x.v.size(0), self.in_v*3]),    # (N, in_v*3)
            rot = rot.reshape([rot.size(0), 3*3]),  # (N, 3*3)
            edge_attr = edge_attr,  # {(E, edge_s), (E, edge_v, 3)}
        )   # (N, out_s+out_v*3)
        out = ScalarVector.from_tensor(out, self.out_v)
        return out

    def message(self, s_i, v_i, s_j, v_j, rot_i, rot_j, edge_attr):
        """
        Compose the message from j to i.
        Args:
            s_i:    Source node scalar features, (E, in_s).
            v_i:    Source node vector features, (E, in_v*3).
            s_j:    Target node scalar features, (E, in_s).
            v_j:    Target node vector features, (E, in_v*3).
            rot_i:  Source node orientation, (E, 3*3).
            rot_j   Target node orientation, (E, 3*3).
            edge_attr:  Edge features, ScalarVector {(E, edge_s), (E, edge_v, 3)}.
        Returns:
            (E, out_s+out_v*3)
        """
        v_i = v_i.reshape([v_i.size(0), self.in_v, 3])
        v_j = v_j.reshape([v_j.size(0), self.in_v, 3])
        rot_i = rot_i.reshape([rot_i.size(0), 3, 3])
        rot_j = rot_j.reshape([rot_j.size(0), 3, 3])

        # Scalar input to message function
        s = torch.cat([s_i, s_j, edge_attr.s], dim=-1)
        
        # Vector input to message function
        v = torch.cat([v_i, v_j, edge_attr.v], dim=-2)  # Vectors in external frame, (E, in_v*2+edge_v, 3)
        v = global_to_local(v, rot_i)
        # print('nodei_att.v (local):', v[-1,0])
        # print('edge_attr.v (local):', v[-1,-1])
        message = ScalarVector(s=s, v=v)
        message = self.message_func(message)
        message.v = local_to_global(message.v, rot_i)

        return message.to_tensor()


class SVGraphConvLayer(nn.Module):

    def __init__(self, 
        node_dims, edge_dims, mlp_mode='svp', n_message_layers=3, n_ff_layers=2, aggr='mean',
        scalar_act='relu', vector_act=['scale', 'sigmoid'], 
        drop_rate=0.1,
    ):
        super().__init__()
        self.conv = SVGraphConv(
            node_dims, node_dims, edge_dims, 
            mlp_mode=mlp_mode, n_layers=n_message_layers, aggr=aggr, 
            scalar_act=scalar_act, vector_act=vector_act,
            share_dot_cross=False, sv_interaction=True,
        )

        self.dropout_1 = SVDropout(drop_rate)
        self.layernorm_1 = SVLayerNorm(node_dims)

        self.ff_func = VectorMLP(
            mode = mlp_mode,
            in_dims = node_dims,
            out_dims = node_dims,
            n_layers = n_ff_layers,
            scalar_act=scalar_act, vector_act=vector_act,
        )
        self.dropout_2 = SVDropout(drop_rate)
        self.layernorm_2 = SVLayerNorm(node_dims)

        self.dummy = nn.Parameter(torch.empty(0))

    def forward(self, x: ScalarVector, edge_index, edge_attr:ScalarVector, rot=None):
        """
        Args:
            x:  Node features, ScalarVector, {(N, s), (N, v, 3)}.
            edge_index: Edge index, (2, E).
            edge_attr:  Edge features, {(E, s), (E, v, 3)}.
            rot:    Rigid rotation matrices, (N, 3, 3).
        Returns:
            y:  Updated node features, ScalarVector, {(N, s), (N, v, 3)}.
        """
        dh = self.conv(x, edge_index, edge_attr, rot=rot)
        x = self.layernorm_1(x + self.dropout_1(dh))

        dh = rotate_apply(self.ff_func, x, rot)
        x = self.layernorm_2(x + self.dropout_2(dh))

        return x
#################################################################################################################
# GAT
#################################################################################################################

def SAttention(target, source, edge_index_i, num_heads=4, bias=None):
    target = target.view(-1, num_heads, target.shape[-1] // num_heads)  # [N,H,C]
    source = source.view(-1, num_heads, source.shape[-1] // num_heads)  # [N,H,C]
    score = (target * source).sum(dim=-1)
    if bias is not None:
        score += bias
    score = softmax(score, edge_index_i)  # [N,H]
    att = source * score.unsqueeze(-1)  # [N,H,C]
    att = att.view(att.shape[0], -1)  # [N,CH]
    return att


def VAttention(target, source, edge_index_i, num_heads=4, bias=None):
    c = target.shape[1]
    target = target.reshape(target.shape[0], num_heads, -1)  # [N,H,C]
    source = source.reshape(source.shape[0], num_heads, -1)  # [N,H,C]
    score = (target * source).sum(dim=-1)
    if bias is not None:
        score += bias
    score = softmax(score, edge_index_i)
    att = source * score.unsqueeze(-1)  # [N,H,C]
    att = att.view(att.shape[0], -1)  # [N,HC]
    return att.view(-1, c, 3)


class GatConv(MessagePassing):

    def __init__(self,
                 in_dims, out_dims, edge_dims, mlp_mode='svp', n_layers=2,
                 scalar_act='relu', vector_act=['scale', 'sigmoid'],
                 aggr='mean',
                 ):
        super().__init__(aggr=aggr, flow='source_to_target')  # (j,i) for updating node i,之前方向都写错了
        self.in_s, self.in_v = in_dims
        self.out_s, self.out_v = out_dims
        self.edge_s, self.edge_v = edge_dims

        self.source_mlp = SVPerceptron(in_dims=in_dims, out_dims=out_dims,
                                       scalar_act=scalar_act, vector_act=vector_act)  # (out_v,out_s)
        self.target_mlp = SVPerceptron(in_dims=in_dims, out_dims=out_dims,
                                       scalar_act=scalar_act, vector_act=vector_act)  # (out_v,out_s)
        self.edge_mlp = SVPerceptron(in_dims=edge_dims, out_dims=out_dims,
                                     scalar_act=scalar_act, vector_act=vector_act)  # (out_v,out_s)
        self.out_mlp = VectorMLP(
            mode=mlp_mode,
            in_dims=(3 * self.in_s, 3 * self.in_v),
            out_dims=(self.out_s, self.out_v),
            n_layers=n_layers,
            scalar_act=scalar_act,
            vector_act=vector_act,
        )
        self.v2s = VectorLinear(self.out_v, self.out_s)
        self.s2v = VectorLinear(self.out_s, self.out_v)
        self.dummy = nn.Parameter(torch.empty(0))

    def forward(self, x, edge_index, edge_attr, rot=None):
        """
        Args:
            x:  ScalarVector, {(N, in_s), (N, in_v, 3)}.
            edge_index: (2, E)
            edge_attr:  ScalarVector, {(E, in_s), (E, in_v, 3)}.
            rot:    Rigid rotation matrices, (N, 3, 3).
        Returns:
            {(E, out_s), (E, out_v, 3)}
        """
        # print(edge_index.shape[1], edge_attr.s.shape[0])

        if edge_index.shape[1] == 0 or edge_attr.s.shape[0] == 0:
            return ScalarVector(0, 0)

        if rot is None:
            rot = torch.eye(3).to(self.dummy.device)
            rot = rot.unsqueeze_(0).repeat(x.v.size(0), 1, 1)  # (N, 3, 3)

        out = self.propagate(
            edge_index=edge_index,
            s=x.s,  # (N, in_s)
            v=x.v.reshape([x.v.size(0), self.in_v * 3]),  # (N, in_v*3)
            rot=rot.reshape([rot.size(0), 3 * 3]),  # (N, 3*3)
            edge_attr=edge_attr,  # {(E, edge_s), (E, edge_v, 3)}
        )  # (N, out_s+out_v*3)
        out = ScalarVector.from_tensor(out, self.out_v)
        return out

    def message(self, s_i, v_i, s_j, v_j, rot_i, rot_j, edge_attr, edge_index_i):
        """
        Compose the message from j to i.
        Args:
            s_i:    Target node scalar features, (E, in_s).
            v_i:    Target node vector features, (E, in_v*3).
            s_j:    Source node scalar features, (E, in_s).
            v_j:    Source node vector features, (E, in_v*3).
            rot_i:  Source node orientation, (E, 3*3).
            rot_j   Target node orientation, (E, 3*3).
            edge_attr:  Edge features, ScalarVector {(E, edge_s), (E, edge_v, 3)}.
        Returns:
            (E, out_s+out_v*3)
        """
        v_i = v_i.reshape([v_i.size(0), self.in_v, 3])
        v_j = v_j.reshape([v_j.size(0), self.in_v, 3])
        rot_i = rot_i.reshape([rot_i.size(0), 3, 3])
        rot_j = rot_j.reshape([rot_j.size(0), 3, 3])

        v_i, v_j, edge_attr.v = global_to_local(v_i, rot_i), \
                                global_to_local(v_j, rot_i), \
                                global_to_local(edge_attr.v, rot_i)
        target = self.target_mlp(ScalarVector(s_i, v_i))
        source = self.source_mlp(ScalarVector(s_j, v_j))
        s_i, v_i = target.s, target.v
        s_j, v_j = source.s, source.v
        edge = self.edge_mlp(edge_attr)
        s_e, v_e = edge.s, edge.v

        # prepare
        vfs = self.v2s(v_j, 'vector', 'dot')  # (*,out_s)
        sfv = self.s2v(s_j, 'scalar')  # (*,out_v,3)

        # update s
        s1 = SAttention(s_i, vfs, edge_index_i)
        s2 = SAttention(s_i, s_j, edge_index_i)
        s_new = torch.cat([s1, s2, s_e], dim=-1)

        # update v
        v1 = VAttention(v_i, sfv, edge_index_i)
        v2 = VAttention(v_i, v_j, edge_index_i)
        v_new = torch.cat([v1, v2, v_e], dim=1)

        # update
        message = self.out_mlp(ScalarVector(s_new, v_new))
        message.v = local_to_global(message.v, rot_i)

        return message.to_tensor()


############################
# GIN conv
############################


class GinConv(MessagePassing):
    def __init__(self,
                 in_dims, out_dims, edge_dims, mlp_mode='svp', n_layers=3,
                 scalar_act='relu', vector_act=['scale', 'sigmoid'],
                 share_dot_cross=False, sv_interaction=True,
                 aggr='add',
                 ):
        super().__init__(aggr=aggr,
                         flow='source_to_target')  # (j,i) \in \Epsilon,aggr for node i,note knn graph construction
        self.in_s, self.in_v = in_dims
        self.out_s, self.out_v = out_dims
        self.edge_s, self.edge_v = edge_dims
        self.epsilon = nn.Parameter(torch.Tensor([0.]))

        self.message_func = VectorMLP(
            mode=mlp_mode,
            in_dims=(2 * self.in_s + self.edge_s, 2 * self.in_v + self.edge_v),
            out_dims=(self.out_s, self.out_v),
            n_layers=n_layers,
            scalar_act=scalar_act,
            vector_act=vector_act,
        )

        self.dummy = nn.Parameter(torch.empty(0))

    def forward(self, x, edge_index, edge_attr, rot=None):
        """
        Args:
            x:  ScalarVector, {(N, in_s), (N, in_v, 3)}.
            edge_index: (2, E),(j,i) as knn edge
            edge_attr:  ScalarVector, {(E, in_s), (E, in_v, 3)}.
            rot:    Rigid rotation matrices, (N, 3, 3).
        Returns:
            {(E, out_s), (E, out_v, 3)}
        """
        # print(edge_index.shape[1],edge_attr.s.shape[0])

        if edge_index.shape[1] == 0 or edge_attr.s.shape[0] == 0:
            return ScalarVector(0, 0)

        # x_ = copy.deepcopy(x.to_tensor()) # tensor
        x_ = (x.to_tensor()).clone()

        if rot is None:
            rot = torch.eye(3).to(self.dummy.device)
            rot = rot.unsqueeze_(0).repeat(x.v.size(0), 1, 1)  # (N, 3, 3)

        out = self.propagate(
            edge_index=edge_index,
            s=x.s,  # (N, in_s)
            v=x.v.reshape([x.v.size(0), self.in_v * 3]),  # (N, in_v*3)
            rot=rot.reshape([rot.size(0), 3 * 3]),  # (N, 3*3)
            edge_attr=edge_attr,  # {(E, edge_s), (E, edge_v, 3)}
        )  # (N, out_s+out_v*3)

        if x_.shape == out.shape:
            out += (1 + self.epsilon) * x_

        out = ScalarVector.from_tensor(out, self.out_v)
        return out

    def message(self, s_i, v_i, s_j, v_j, rot_i, rot_j, edge_attr):
        """
        Compose the message from j to i.
        Args:
            s_i:    Target node scalar features, (E, in_s).
            v_i:    Target node vector features, (E, in_v*3).
            s_j:    Source node scalar features, (E, in_s).
            v_j:    Source node vector features, (E, in_v*3).
            rot_i:  Source node orientation, (E, 3*3).
            rot_j   Target node orientation, (E, 3*3).
            edge_attr:  Edge features, ScalarVector {(E, edge_s), (E, edge_v, 3)}.
        Returns:
            (E, out_s+out_v*3)
        """
        v_i = v_i.reshape([v_i.size(0), self.in_v, 3])
        v_j = v_j.reshape([v_j.size(0), self.in_v, 3])
        rot_i = rot_i.reshape([rot_i.size(0), 3, 3])
        rot_j = rot_j.reshape([rot_j.size(0), 3, 3])

        # Scalar input to message function
        s = torch.cat([s_i, s_j, edge_attr.s], dim=-1)

        # Vector input to message function
        v = torch.cat([v_i, v_j, edge_attr.v], dim=-2)  # Vectors in external frame, (E, in_v*2+edge_v, 3)
        v = global_to_local(v, rot_i)
        # print('nodei_att.v (local):', v[-1,0])
        # print('edge_attr.v (local):', v[-1,-1])
        message = ScalarVector(s=s, v=v)
        message = self.message_func(message)
        message.v = local_to_global(message.v, rot_i)

        return message.to_tensor()


class SVGraphConvLayer(nn.Module):

    def __init__(self,
                 node_dims, edge_dims, conv='gnn', mlp_mode='svp', n_message_layers=3, n_ff_layers=2, aggr='mean',
                 scalar_act='relu', vector_act=['scale', 'sigmoid'],
                 drop_rate=0.1, autoregressive=False
                 ):  # conv = gat or gnn
        super().__init__()
        if conv == 'gnn':
            self.conv = GnnConv(
                node_dims, node_dims, edge_dims,
                mlp_mode=mlp_mode, n_layers=n_message_layers, aggr="add" if autoregressive else aggr,
                scalar_act=scalar_act, vector_act=vector_act,
                share_dot_cross=False, sv_interaction=True,
            )
        elif conv == 'gat':
            self.conv = GatConv(node_dims, node_dims, edge_dims, mlp_mode=mlp_mode,
                                aggr="add" if autoregressive else aggr,
                                scalar_act=scalar_act, vector_act=vector_act)
        elif conv == 'gin':
            self.conv = GinConv(
                node_dims, node_dims, edge_dims,
                mlp_mode=mlp_mode, n_layers=n_message_layers, aggr="add",  # gin sum
                scalar_act=scalar_act, vector_act=vector_act,
                share_dot_cross=False, sv_interaction=True,
            )
        else:
            raise ValueError("Not Implemented")

        self.dropout_1 = SVDropout(drop_rate)
        self.layernorm_1 = SVLayerNorm(node_dims)

        self.ff_func = VectorMLP(
            mode=mlp_mode,
            in_dims=node_dims,
            out_dims=node_dims,
            n_layers=n_ff_layers,
            scalar_act=scalar_act, vector_act=vector_act,
        )
        self.dropout_2 = SVDropout(drop_rate)
        self.layernorm_2 = SVLayerNorm(node_dims)

        self.dummy = nn.Parameter(torch.empty(0))

    def forward(self, x: ScalarVector, edge_index, edge_attr: ScalarVector,
                autoregressive_x=None, node_mask=None, rot=None):
        """
        Args:
            x:  Node features, ScalarVector, {(N, s), (N, v, 3)}.
            edge_index: Edge index, (2, E).
            edge_attr:  Edge features, {(E, s), (E, v, 3)}.
            rot:    Rigid rotation matrices, (N, 3, 3).
        Returns:
            y:  Updated node features, ScalarVector, {(N, s), (N, v, 3)}.
        """
        dh = self.conv(x, edge_index, edge_attr, rot=rot)
        x = self.layernorm_1(x + self.dropout_1(dh))
        dh = rotate_apply(self.ff_func, x, rot)
        x = self.layernorm_2(x + self.dropout_2(dh))

        return x


def _test():
    from .geometric import orthogonalize_matrix, apply_rotation, apply_inverse_rotation, compose_rotation

    num_nodes = 5
    in_s, in_v = 2, 2
    out_s, out_v = 1, 1
    edge_s, edge_v = 1, 1

    edge_index = torch.LongTensor([
        [0, 1, 1, 2, 3, 3, 3, 4],
        [3, 2, 4, 3, 0, 1, 4, 0],
    ])

    s_node = torch.randn(num_nodes, in_s) * 10
    s_edge = torch.randn(edge_index.size(1), edge_s) * 10
    v_node = torch.randn(num_nodes, in_v, 3) * 10
    v_edge = torch.randn(edge_index.size(1), edge_v, 3) * 10

    frames = orthogonalize_matrix(torch.randn([num_nodes, 3, 3]))

    rot_glob = orthogonalize_matrix(torch.randn([1, 3, 3]))
    rot_node = rot_glob.repeat(num_nodes, 1, 1)
    rot_edge = rot_glob.repeat(edge_index.size(1), 1, 1)
    rot_frame = rot_node

    gconv = SVGraphConv(
        in_dims = (in_s, in_v),
        out_dims = (out_s, out_v),
        edge_dims = (edge_s, edge_v),
        n_layers = 3,
        vector_act=['project', None]
    )

    # Reference
    x = ScalarVector(s=s_node, v=v_node)
    edge_attr = ScalarVector(s=s_edge, v=v_edge)
    out_ref = gconv(x, edge_index, edge_attr, rot=frames)

    # Rotated
    x = ScalarVector(s=s_node, v=apply_rotation(rot_node, v_node))
    edge_attr = ScalarVector(s=s_edge, v=apply_rotation(rot_edge, v_edge))
    out_rot = gconv(x, edge_index, edge_attr, rot=compose_rotation(rot_frame, frames))
    out_rot.v = apply_inverse_rotation(rot_node, out_rot.v)

    print((out_ref.s - out_rot.s).abs().max())
    print((out_ref.v - out_rot.v).abs().max())
    assert torch.allclose(out_ref.s, out_rot.s, atol=1e-5, rtol=1e-4)
    assert torch.allclose(out_ref.v, out_rot.v, atol=1e-5, rtol=1e-4)
    print('[SVGraphConv] Passed invariance and equivariance tests.')


    gconv = SVGraphConvLayer(
        node_dims = (in_s, in_v),
        edge_dims = (edge_s, edge_v),
        mlp_mode = 'svp',
    )

    # Reference
    gconv.eval()    # Avoid dropout, which might lead to different output
    x = ScalarVector(s=s_node, v=v_node)
    edge_attr = ScalarVector(s=s_edge, v=v_edge)
    out_ref = gconv(x, edge_index, edge_attr, rot=frames)

    # Rotated
    gconv.eval()    # Avoid dropout, which might lead to different output
    x = ScalarVector(s=s_node, v=apply_rotation(rot_node, v_node))
    edge_attr = ScalarVector(s=s_edge, v=apply_rotation(rot_edge, v_edge))
    out_rot = gconv(x, edge_index, edge_attr, rot=compose_rotation(rot_frame, frames))
    out_rot.v = apply_inverse_rotation(rot_node, out_rot.v)

    print((out_ref.s - out_rot.s).abs().max())
    print((out_ref.v - out_rot.v).abs().max())
    assert torch.allclose(out_ref.s, out_rot.s, atol=1e-5, rtol=1e-4)
    assert torch.allclose(out_ref.v, out_rot.v, atol=1e-5, rtol=1e-4)
    print('[SVGraphConvLayer] Passed invariance and equivariance tests.')


if __name__ == '__main__':
    for i in range(100):
        print('========== Test %d ==========' % i)
        _test()
        print('\n')

