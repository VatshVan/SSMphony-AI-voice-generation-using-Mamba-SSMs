# import torch
# import torch.nn as nn

# class S4Layer(nn.Module):
#     def __init__(self, d_model):
#         super().__init__()
#         self.d_model = d_model

#     def forward(self, x):
#         return x

import torch
import torch.nn as nn
import torch.nn.functional as F
from s4_pt.layers.s4_kernel import s4d_kernel_zoh
from s4_pt.layers.hippo import make_s4d_diagonal_A

class S4Layer(nn.Module):
    def __init__(self, d_model, N=64, step=0.1):
        super().__init__()
        self.N = N
        self.step = step

        A = make_s4d_diagonal_A(N)
        self.A = nn.Parameter(A)

        self.B = nn.Parameter(torch.randn(N))
        self.C = nn.Parameter(torch.randn(N))
        self.D = nn.Parameter(torch.ones(1))

        self.in_proj = nn.Linear(d_model, 1)
        self.out_proj = nn.Linear(1, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, L, D = x.shape

        u = self.in_proj(x).squeeze(-1)  # (B, L)

        K = s4d_kernel_zoh(self.C, self.A, L, self.step)  # (L,)

        u_pad = torch.fft.rfft(torch.nn.functional.pad(u, (0, L)))
        k_pad = torch.fft.rfft(torch.nn.functional.pad(K, (0, L)))

        y = torch.fft.irfft(u_pad * k_pad)[:, :L]

        y = y + self.D * u
        y = y.unsqueeze(-1)

        y = self.out_proj(y)
        return self.norm(x + y)