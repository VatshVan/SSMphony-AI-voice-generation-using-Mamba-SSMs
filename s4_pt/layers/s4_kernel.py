import torch

def discretize_zoh(A, B, step):
    expA = torch.exp(step * A)
    Bd = (expA - 1.0) / A * B
    return expA, Bd


def s4d_kernel_zoh(C, A, L, step):
    n = torch.arange(L, device=A.device)
    exp_term = torch.exp(n[:, None] * step * A[None, :])
    kernel = (C * (torch.exp(step * A) - 1.0) / A * exp_term).sum(dim=1)
    return kernel.real
