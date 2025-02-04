import torch


class ScalarVector(object):

    def __init__(self, s, v):
        super().__init__()
        self.s = s
        self.v = v

    def to(self, *args, **kwargs):
        self.s = self.s.to(*args, **kwargs)
        self.v = self.v.to(*args, **kwargs)
        return self

    def clone(self):
        return ScalarVector(s=self.s.clone(), v=self.v.clone())

    def to_tensor(self):
        s, v = self.s, self.v
        v = torch.reshape(v, v.shape[:-2] + (v.shape[-2]*3,))
        return torch.cat([s, v], -1)

    def concat(self, other):
        return ScalarVector(
            s = torch.cat([self.s, other.s], dim=-1),
            v = torch.cat([self.v, other.v], dim=-2),
        )

    @classmethod
    def from_tensor(cls, x, nv):
        v = torch.reshape(x[..., -3*nv:], x.shape[:-1] + (nv, 3))
        s = x[..., :-3*nv]
        return cls(s=s, v=v)

    def __repr__(self) -> str:
        return 'ScalarVector(s=%r, v=%r)' % (self.s, self.v,)

    def __add__(self, other):
        return ScalarVector(
            s = self.s + other.s,
            v = self.v + other.v,
        )

    def __getitem__(self, index):
        return ScalarVector(
            s = self.s[index],
            v = self.v[index],
        )


def safe_norm(x, dim=-1, keepdim=False, eps=1e-8, sqrt=True):
    out = torch.clamp(torch.sum(torch.square(x), dim=dim, keepdim=keepdim), min=eps)
    return torch.sqrt(out) if sqrt else out
