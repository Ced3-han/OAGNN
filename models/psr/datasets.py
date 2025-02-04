import math
import os
from joblib import Parallel, delayed
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, BatchSampler
from torch_geometric.data import Data
import atom3d.datasets
from tqdm.auto import tqdm

from utils.data import protein_df_to_structures, structure_to_seq_coords


def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def _preprocess(data):
    structure = protein_df_to_structures(data['atoms'])[0]
    bb_coords, seq_str, seq = structure_to_seq_coords(structure)
    bb_coords = torch.FloatTensor(bb_coords)    # (L, 4, 3), N, CA, C, O
    seq = torch.LongTensor(seq)   # (L, )
    pos_N, pos_CA, pos_C, pos_O = torch.unbind(bb_coords, 1)
    target_id, decoy_id = map(lambda x: x.strip("' "), data['id'].strip('()').split(','))
    return Data(
        num_nodes = seq.size(0),
        pos_N = pos_N,
        pos_CA = pos_CA,
        pos_C = pos_C,
        pos_O = pos_O,
        seq = seq,
        seq_fasta = seq_str,
        target_id = target_id,
        decoy_id = decoy_id,
        rmsd = data['scores']['rmsd'],
        tm = data['scores']['tm'],
        gdt_ts = data['scores']['gdt_ts'],
        gdt_ha = data['scores']['gdt_ha'],
    )


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


class PSRDataset(Dataset):

    def __init__(self, lmdb_path, preproc_n_jobs=32):
        super().__init__()
        self.a3d_dataset = atom3d.datasets.load_dataset(lmdb_path, 'lmdb')
        self.cache_path = os.path.join(lmdb_path, 'processed.pt')
        self.dataset = None
        self.preproc_n_jobs = preproc_n_jobs
        self.target_to_indices = {}
        self._load()

    def _load(self):
        if os.path.exists(self.cache_path):
            print('[%s] Loading from cache: %s' % (self.__class__.__name__, self.cache_path))
            self.dataset = torch.load(self.cache_path)
        else:
            self.dataset = self._process()

        for index in range(len(self.dataset)):
            data = self.dataset[index]
            if data.target_id not in self.target_to_indices:
                self.target_to_indices[data.target_id] = []
            self.target_to_indices[data.target_id].append(index)

    def _process(self):
        dataset = Parallel(n_jobs=self.preproc_n_jobs)(delayed(_preprocess)(data) for data in tqdm(self.a3d_dataset, desc='Preprocessing'))
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


class PairBatchSampler(BatchSampler):

    def __init__(self, target_to_indices, batch_size=2):
        assert batch_size % 2 == 0
        self.batch_size = batch_size
        self.target_to_indices = target_to_indices
        self.targets = []
        self.counts = []
        for k, v in self.target_to_indices.items():
            if len(v) < 2: continue
            self.targets.append(k)
            self.counts.append(len(v))

        self.counts = np.array(self.counts)
        self.total = self.counts.sum()

    def __iter__(self):
        for _ in range(len(self)):
            targets_selected = np.random.choice(self.targets, self.batch_size//2, replace=False, p=(self.counts/self.total))
            idx = []
            for target in targets_selected:
                idx += list(np.random.choice(self.target_to_indices[target], 2, replace=False))
            yield idx

    def __len__(self):
        return self.total // self.batch_size


if __name__ == '__main__':
    _ = PSRDataset('/home/luost/data/datasets/PSR/split-by-year/data/train')
