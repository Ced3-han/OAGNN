import functools
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
import torch_scatter
import torch_cluster

from modules.linear import rotate_apply
from modules.common import ScalarVector
from modules.gconv import SVGraphConvLayer
from modules.geometric import construct_3d_basis
from modules.norm import SVLayerNorm
from modules.perceptron import VectorPerceptron


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


def _repeat_graph(h_node, h_edge, edge_index, n_samples, device):
    L = h_node.s.size(0)
    h_node = ScalarVector(
        s = h_node.s.repeat(n_samples, 1),
        v = h_node.v.repeat(n_samples, 1, 1),
    )
    h_edge = ScalarVector(
        s = h_edge.s.repeat(n_samples, 1),
        v=  h_edge.v.repeat(n_samples, 1, 1),
    )
    edge_index = edge_index.expand(n_samples, -1, -1)
    offset = L * torch.arange(n_samples, device=device).view(-1, 1, 1)
    edge_index = torch.cat(tuple(edge_index + offset), dim=-1)
    return h_node, h_edge, edge_index


class SVGraphConvLayerAutoRegressive(SVGraphConvLayer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: ScalarVector, edge_index, edge_attr:ScalarVector, autoregressive_x, node_mask=None, rot=None):
        idx_i, idx_j = edge_index   # (E, ), messages flow from j to i ( j -> i )
        mask = idx_j < idx_i        # (E, )
        edge_index_forward  = edge_index[:, mask]
        edge_index_backward = edge_index[:, ~mask]
        edge_attr_forward  = edge_attr[mask]
        edge_attr_backward = edge_attr[~mask]

        dh = self.conv(x, edge_index_forward, edge_attr_forward) + self.conv(autoregressive_x, edge_index_backward, edge_attr_backward)
        count = torch_scatter.scatter_add(torch.ones_like(idx_i), idx_i, dim_size=dh.s.size(0)).clamp(min=1).unsqueeze(-1)    # (N, 1)
        dh.s = dh.s / count
        dh.v = dh.v / count.unsqueeze(-1)

        if node_mask is not None:
            x_ = x
            x = x[node_mask]
            dh = dh[node_mask]

        x = self.layernorm_1(x + self.dropout_1(dh))

        dh = rotate_apply(self.ff_func, x, rot)
        x = self.layernorm_2(x + self.dropout_2(dh))

        if node_mask is not None:
            x_.s[node_mask] = x.s
            x_.v[node_mask] = x.v
            x = x_

        return x



class CPDNetwork(nn.Module):

    def __init__(self, 
        node_in_dims=(6, 3), node_hid_dims=(128, 32), edge_hid_dims=(64, 16),
        num_layers=3, drop_rate=0.1, top_k=30, aa_embed_dim=20,
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
            Perceptron_(edge_in_dims, edge_hid_dims, scalar_act=None, vector_act=None),
            SVLayerNorm(edge_hid_dims),
            Perceptron_(edge_hid_dims, edge_hid_dims, scalar_act=None, vector_act=None),
        )

        # Encoder
        self.encoder_layers = nn.ModuleList(
            SVGraphConvLayer(
                node_hid_dims, 
                edge_hid_dims, 
                mlp_mode=perceptron_mode, 
                drop_rate=drop_rate
            ) for _ in range(num_layers)
        )

        # Embed known amino acids
        self.aa_embed = nn.Embedding(20, embedding_dim=aa_embed_dim)
        edge_dec_hid_dims = (edge_hid_dims[0]+aa_embed_dim, edge_hid_dims[1])

        # Decoder
        self.decoder_layers = nn.ModuleList(
            SVGraphConvLayerAutoRegressive(
                node_hid_dims, 
                edge_dec_hid_dims, 
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
            nn.Linear(node_s_dim, 20),
        )

    def _get_inputs(self, batch):
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
        R_node = R_node.nan_to_num()
        R_edge = R_node[edge_index[0]]  # (E, 3, 3)
        return edge_index, node_in, edge_in, R_node, R_edge

    def forward(self, batch):
        edge_index, node_in, edge_in, R_node, R_edge = self._get_inputs(batch)

        h_node = rotate_apply(self.W_node, node_in, R_node)
        h_edge = rotate_apply(self.W_edge, edge_in, R_edge)

        for layer in self.encoder_layers:
            h_node = layer(h_node, edge_index, h_edge, rot=R_node)     

        encoder_embeddings = h_node

        edge_index_i, edge_index_j = edge_index

        h_seq = self.aa_embed(batch.seq)    # (N, aa_embed_dim)
        h_seq = h_seq[edge_index_j]        # Amino acid embedding of j-nodes.
        h_seq[edge_index_j >= edge_index_i] = 0
        h_edge_autoregressive = ScalarVector(
            s = torch.cat([h_edge.s, h_seq], dim=-1),
            v = h_edge.v,
        )

        for layer in self.decoder_layers:
            h_node = layer(h_node, edge_index, h_edge_autoregressive, autoregressive_x=encoder_embeddings, rot=R_node)

        out = rotate_apply(self.W_out, h_node, R_node).s  # (N, node_hid_s)
        logits = self.dense(out)   # (N, 20)
        return logits

    def sample(self, batch, n_samples, temperature=0.1):
        device = self.dummy.device

        edge_index, node_in, edge_in, R_node, R_edge = self._get_inputs(batch)

        h_node = rotate_apply(self.W_node, node_in, R_node)
        h_edge = rotate_apply(self.W_edge, edge_in, R_edge)
        L = h_node.s.size(0)

        for layer in self.encoder_layers:
            h_node = layer(h_node, edge_index, h_edge, rot=R_node)     

        h_node, h_edge, edge_index = _repeat_graph(h_node, h_edge, edge_index, n_samples, device)

        seq = torch.zeros(n_samples * L, device=device, dtype=torch.int)
        h_seq = torch.zeros(n_samples * L, 20, device=device)
        h_node_cache = [h_node.clone() for _ in self.decoder_layers]

        idx_i, idx_j = edge_index
        for i in range(L):
            h_seq_ = h_seq[idx_j]
            h_seq_[idx_j >= idx_i] = 0
            h_edge_ = ScalarVector(
                s = torch.cat([h_edge.s, h_seq_], dim=-1),
                v = h_edge.v,
            )

            edge_mask = (idx_i % L == i)
            edge_index_ = edge_index[:, edge_mask]
            h_edge_ = h_edge_[edge_mask]
            node_mask = torch.zeros(n_samples * L, device=device, dtype=torch.bool)
            node_mask[i::L] = True

            for j, layer in enumerate(self.decoder_layers):
                out = layer(h_node_cache[j], edge_index_, h_edge_,
                            autoregressive_x=h_node_cache[0], node_mask=node_mask)
                out = out[node_mask]

                if j < len(self.decoder_layers) - 1:
                    h_node_cache[j+1].s[i::L] = out.s
                    h_node_cache[j+1].v[i::L] = out.v

            out = rotate_apply(self.W_out, h_node, R_node).s  # (N, node_hid_s)
            logits = self.dense(out)   # (N, 20)

            seq[i::L] = Categorical(logits=logits / temperature).sample()
            h_seq[i::L] = self.aa_embed(seq[i::L])
        
        return seq.view(n_samples, L)
