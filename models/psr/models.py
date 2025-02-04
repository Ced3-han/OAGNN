import functools
from modules.linear import rotate_apply
import numpy as np
import torch
from torch import nn
import torch_cluster
import torch_scatter

from modules.common import ScalarVector
from modules.geometric import construct_3d_basis
from modules.norm import SVLayerNorm
from modules.perceptron import VectorPerceptron
from modules.gconv import SVGraphConvLayer


def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design
    
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF


def _positional_embeddings(edge_index, num_embeddings=None, period_range=[2, 1000], device='cpu'):
    # From https://github.com/jingraham/neurips19-graph-protein-design
    d = edge_index[0] - edge_index[1]
    
    frequency = torch.exp(
        torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=device)
        * -(np.log(10000.0) / num_embeddings)
    )
    angles = d.unsqueeze(-1) * frequency
    E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
    return E


class PSRNetwork(nn.Module):

    def __init__(self, 
        node_in_dims=(26, 3), node_hid_dims=(128, 32), edge_hid_dims=(64, 16),
        num_layers=3, drop_rate=0.1, top_k=30, 
        num_rbf=16, num_positional_embeddings=16,
        perceptron_mode='svp'
    ):
        super().__init__()
        self.top_k = top_k
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.dummy = nn.Parameter(torch.empty(0))

        Perceptron_ = functools.partial(VectorPerceptron, mode=perceptron_mode)

        # Initial embedding of node features
        self.W_node = nn.Sequential(
            Perceptron_(node_in_dims, node_hid_dims, scalar_act=None, vector_act=None),
            SVLayerNorm(node_hid_dims),
            Perceptron_(node_hid_dims, node_hid_dims, scalar_act=None, vector_act=None),
        )

        # Initial embedding of edge features
        edge_in_dims = (num_rbf+num_positional_embeddings, 1)
        self.W_edge = nn.Sequential(
            SVLayerNorm(edge_in_dims),
            Perceptron_(edge_in_dims, edge_hid_dims, scalar_act=None, vector_act=None),
        )

        # Graph convolutional layers
        self.layers = nn.ModuleList(
            SVGraphConvLayer(
                node_hid_dims, 
                edge_hid_dims, 
                mlp_mode=perceptron_mode, 
                drop_rate=drop_rate
            ) for _ in range(num_layers)
        )

        # Output layers: convert vector and scalar features to scalar only features
        node_s_dim = node_hid_dims[0]
        self.W_out = nn.Sequential(
            SVLayerNorm(node_hid_dims),
            Perceptron_(node_hid_dims, (node_s_dim, 0)),
        )
        
        self.dense = nn.Sequential(
            nn.Linear(node_s_dim, node_s_dim), nn.ReLU(),
            nn.Dropout(p=drop_rate),
            nn.Linear(node_s_dim, 1),
        )


    def forward(self, batch):
        device = self.dummy.device

        # Shortcut for node features
        node_s = batch.node_s
        node_v = batch.node_v
        node_in = ScalarVector(s=node_s, v=node_v)
        
        # Get edge indices and features
        edge_index = torch_cluster.knn_graph(batch.pos_CA, k=self.top_k, batch=batch.batch, flow='target_to_source')
        pos_embeddings = _positional_embeddings(edge_index, self.num_positional_embeddings, device=device)
        E_vectors = batch.pos_CA[edge_index[0]] - batch.pos_CA[edge_index[1]]
        rbf = _rbf(E_vectors.norm(dim=-1), D_count=self.num_rbf, device=device)
        edge_s = torch.cat([rbf, pos_embeddings], dim=-1)   # (E, n_rbf+n_pos_embed)
        edge_v = _normalize(E_vectors).unsqueeze_(-2)       # (E, 1, 3)
        edge_in = ScalarVector(s=edge_s, v=edge_v)
        
        # Get rotation matrices
        R_node = construct_3d_basis(batch.pos_CA, batch.pos_C, batch.pos_N)    # (N, 3, 3)
        R_edge = R_node[edge_index[0]]  # (E, 3, 3)

        h_node = rotate_apply(self.W_node, node_in, R_node)
        h_edge = rotate_apply(self.W_edge, edge_in, R_edge)   # TODO: R_edge

        for layer in self.layers:
            h_node = layer(h_node, edge_index, h_edge, rot=R_node)     

        out = rotate_apply(self.W_out, h_node, R_node).s  # (N, node_hid_s)
        out = torch_scatter.scatter_mean(out, batch.batch, dim=0)   # Mean pooling, (G, node_hid_s).
        y = self.dense(out).squeeze(-1) + 0.5   # (G, )
        return y    


if __name__ == '__main__':
    from torch_geometric.data import Batch
    from .datasets import PSRDataset
    from modules.geometric import orthogonalize_matrix, apply_rotation, apply_inverse_rotation, compose_rotation
    dataset = PSRDataset('/home/luost/data/datasets/PSR/split-by-year/data/train', preproc_n_jobs=64)
    model = PSRNetwork(perceptron_mode='svp').to('cuda')
    batch = Batch.from_data_list([dataset[0], dataset[1], dataset[2], dataset[3]]).to('cuda')

    rot_glob = orthogonalize_matrix(torch.randn([1, 3, 3])).to('cuda')
    batch_rot = batch.clone()
    batch_rot.pos_CA = apply_rotation(rot_glob, batch.pos_CA)
    batch_rot.pos_C = apply_rotation(rot_glob, batch.pos_C)
    batch_rot.pos_N = apply_rotation(rot_glob, batch.pos_N)
    batch_rot.pos_O = apply_rotation(rot_glob, batch.pos_O)
    batch_rot.node_v = apply_rotation(rot_glob, batch.node_v)

    model.eval()
    y_ref = model(batch)
    print(y_ref)

    model.eval()
    y_rot = model(batch_rot)
    print(y_rot)

    assert torch.allclose(y_ref, y_rot, atol=1e-5, rtol=1e-4), (y_ref - y_rot).abs().max()
    print('[Model] Passed invariance test.')
