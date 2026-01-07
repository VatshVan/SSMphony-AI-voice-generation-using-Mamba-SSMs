import torch
from s4_pt.models.s4_model import S4Model

x = torch.randn(8, 128, 64)
model = S4Model(d_model=64, num_classes=10)
y = model(x)

print(y.shape)
# Expected output: torch.Size([8, 10])  # since we have a batch size of 8 and num_classes=10