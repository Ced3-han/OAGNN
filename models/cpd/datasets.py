import os
import math
import json
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm.auto import tqdm
from Bio.PDB.Polypeptide import one_to_index


def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def _dihedrals(pos_N, pos_CA, pos_C, eps=1e-7):
    """
    Args:
        pos_N, pos_CA, pos_C:   (N, 3).
    Returns:
        Dihedral features, (N, 6).
    """
    X = torch.cat([pos_N.view(-1, 1, 3), pos_CA.view(-1, 1, 3), pos_C.view(-1, 1, 3)], dim=1)   # (N, 3, 3)
    X = torch.reshape(X[:, :3], [3*X.shape[0], 3])
    dX = X[1:] - X[:-1]
    U = _normalize(dX, dim=-1)
    u_2 = U[:-2]
    u_1 = U[1:-1]
    u_0 = U[2:]

    # Backbone normals
    n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
    n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

    # Angle between normals
    cosD = torch.sum(n_2 * n_1, -1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

    # This scheme will remove phi[0], psi[-1], omega[-1]
    D = F.pad(D, [1, 2]) 
    D = torch.reshape(D, [-1, 3])
    # Lift angle representations to the circle
    D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
    return D_features


def _orientations(pos_CA):
    X = pos_CA
    forward = _normalize(X[1:] - X[:-1])
    backward = _normalize(X[:-1] - X[1:])
    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])
    return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)


def _sidechains(pos_N, pos_CA, pos_C):
    X = torch.cat([pos_N.view(-1, 1, 3), pos_CA.view(-1, 1, 3), pos_C.view(-1, 1, 3)], dim=1)   # (N, 3, 3)
    n, origin, c = X[:, 0], X[:, 1], X[:, 2]
    c, n = _normalize(c - origin), _normalize(n - origin)
    bisector = _normalize(c + n)
    perp = _normalize(torch.cross(c, n))
    vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
    return vec 


class CATH_CPD_Dataset(Dataset):

    def __init__(self, root, split):
        super().__init__()
        self.data_path = os.path.join(root, 'chain_set.jsonl')
        self.split_path = os.path.join(root, 'chain_set_splits.json')
        self.cache_path = os.path.join(root, 'processed_%s.pt' % split)
        assert split in ('train', 'val', 'test')
        self.split = split
        self.dataset = None
        self._load()

    def _load(self):
        if os.path.exists(self.cache_path):
            print('[%s] Loading from cache: %s' % (self.__class__.__name__, self.cache_path))
            self.dataset = torch.load(self.cache_path)
        else:
            self.dataset = self._process()

    def _process(self):
        dataset = []
        with open(self.split_path) as f:
            data_list = json.load(f)[self.split]
        with open(self.data_path) as f:
            lines = f.readlines()

        for line in tqdm(lines, desc='Preprocess'):
            entry = json.loads(line)
            if entry['name'] not in data_list:
                continue
            coords = torch.FloatTensor(list(zip(
                entry['coords']['N'], entry['coords']['CA'], entry['coords']['C'], entry['coords']['O']
            )))  # (N, 4, 3)
            mask = coords.sum(dim=(1, 2)).isfinite()    # (N, )
            coords[~mask] = float('inf')
            pos_N, pos_CA, pos_C, pos_O = torch.unbind(coords, dim=1)

            dataset.append(Data(
                seq_fasta = entry['seq'],
                seq = torch.LongTensor([one_to_index(c) for c in entry['seq']]),
                pos_N = pos_N,
                pos_CA = pos_CA,
                pos_C = pos_C,
                pos_O = pos_O,
                num_chains = entry['num_chains'],
                name = entry['name'],
                cath = entry['CATH'],
                mask = mask,
            ))
        torch.save(dataset, self.cache_path)
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index].clone()

        dihedrals = _dihedrals(pos_N=data.pos_N, pos_CA=data.pos_CA, pos_C=data.pos_C)
        orientations = _orientations(pos_CA=data.pos_CA)
        sidechains = _sidechains(pos_N=data.pos_N, pos_CA=data.pos_CA, pos_C=data.pos_C)

        onehot = F.one_hot(data.seq, 20)
        data.node_s = torch.cat([dihedrals, onehot], dim=-1)
        data.node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)

        return data
