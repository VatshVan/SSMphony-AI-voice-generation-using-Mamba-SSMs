import torch

def make_hippo_legS(N):
    n = torch.arange(N, dtype=torch.float32)
    P = torch.sqrt(1 + 2 * n)
    A = P[:, None] * P[None, :]
    A = torch.tril(A) - torch.diag(n)
    return -A


def make_s4d_diagonal_A(N):
    A = make_hippo_legS(N)
    return torch.diag(A)
