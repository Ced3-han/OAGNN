import torch
import torch.linalg


def normalize_vector(v, dim, eps=0):
    return v / (torch.linalg.norm(v, ord=2, dim=dim, keepdim=True) + eps)


def project_v2v(v, e, dim):
    """
    Description:
        Project vector `v` onto vector `e`.
    Args:
        v:  (*, 3).
        e:  (*, 3).
    """
    return (e * v).sum(dim=dim, keepdim=True) * e


def construct_3d_basis(center, p1, p2):
    """
    Args:
        center: (*, 3), usually the position of C_alpha.
        p1:     (*, 3), usually the position of C.
        p2:     (*, 3), usually the position of N.
    Returns
        A batch of orthogonal basis matrix, (*, 3, 3cols_index).
        The matrix is composed of 3 column vectors: [e1, e2, e3].
    """
    v1 = p1 - center    # (*, 3)
    e1 = normalize_vector(v1, dim=-1)

    v2 = p2 - center    # (*, 3)
    u2 = v2 - project_v2v(v2, e1, dim=-1)
    e2 = normalize_vector(u2, dim=-1)

    e3 = torch.cross(e1, e2, dim=-1)    # (*, 3)

    mat = torch.cat([
        e1.unsqueeze(-1), e2.unsqueeze(-1), e3.unsqueeze(-1)
    ], dim=-1)  # (*, 3, 3_index)
    return mat


def orthogonalize_matrix(R):
    """
    Args:
        R:  (..., 3, 3_idx)
    """
    repr_6d = torch.cat([R[..., 0], R[..., 1]], dim=-1) # (..., 6)
    return repr_6d_to_rotation_matrix(repr_6d)


def local_to_global(p, R, t=None):
    """
    Description:
        Convert local (internal) coordinates to global (external) coordinates q.
        q <- Rp + t
    Args:
        p:  Local coordinates, (N, ..., 3).
        R:  (N, 3, 3).
        t:  (N, 3).
    Returns:
        q:  Global coordinates, (N, ..., 3).
    """
    assert p.size(-1) == 3
    p_size = p.size()
    N = p_size[0]

    p = p.view(N, -1, 3).transpose(-1, -2)   # (N, *, 3) -> (N, 3, *)
    q = torch.matmul(R, p)  # (N, 3, *)
    if t is not None:
        q = q + t.unsqueeze(-1)
    q = q.transpose(-1, -2).reshape(p_size)     # (N, 3, *) -> (N, *, 3) -> (N, ..., 3)
    return q


def global_to_local(q, R, t=None):
    """
    Description:
        Convert global (external) coordinates q to local (internal) coordinates p.
        p <- R^{T}(q - t)
    Args:
        q:  Global coordinates, (N, ..., 3).
        R:  (N, 3, 3).
        t:  (N, 3).
    Returns:
        p:  Local coordinates, (N, ..., 3).
    """
    assert q.size(-1) == 3
    q_size = q.size()
    N = q_size[0]

    q = q.view(N, -1, 3).transpose(-1, -2)   # (N, *, 3) -> (N, 3, *)
    if t is not None:
        p = torch.matmul(R.transpose(-1, -2), (q - t.unsqueeze(-1)))  # (N, 3, *)        
    else:
        p = torch.matmul(R.transpose(-1, -2), q)
    p = p.transpose(-1, -2).reshape(q_size)     # (N, 3, *) -> (N, *, 3) -> (N, ..., 3)
    return p


def apply_rotation(R, v):
    return local_to_global(v, R, t=None)


def apply_inverse_rotation(R, v):
    return global_to_local(v, R, t=None)


def repr_6d_to_rotation_matrix(x):
    """
    Args:
        x:  6D representations, (..., 6).
    Returns:
        Rotation matrices, (..., 3, 3_index).
    """
    a1, a2 = x[..., 0:3], x[..., 3:6]
    b1 = normalize_vector(a1, dim=-1)
    b2 = normalize_vector(a2 - project_v2v(a2, b1, dim=-1), dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)

    mat = torch.cat([
        b1.unsqueeze(-1), b2.unsqueeze(-1), b3.unsqueeze(-1)
    ], dim=-1)  # (N, L, 3, 3_index)
    return mat


def compose_rotation(R1, R2):
    """
    Args:
        R1,R2:  (*, 3, 3).
    Returns
        R_new <- R1R2
    """
    R_new = torch.matmul(R1, R2)    # (N, L, 3, 3)
    return R_new


if __name__ == '__main__':
    # Test: local <-> global
    center = torch.randn([5, 100, 3])
    p1 = torch.randn([5, 100, 3])
    p2 = torch.randn([5, 100, 3])

    R = construct_3d_basis(center, p1, p2)
    t = center
    p = torch.randn([5, 100, 3])
    assert torch.allclose(local_to_global(R, t, global_to_local(R, t, p)), p, atol=1e-5)
    assert torch.allclose(global_to_local(R, t, local_to_global(R, t, p)), p, atol=1e-5)
    print('[Passed] Local <-> Global')
