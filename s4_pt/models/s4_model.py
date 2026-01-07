import torch
import torch.nn as nn
from s4_pt.layers.s4_layer import S4Layer

class S4Model(nn.Module):
    def __init__(self, d_model, num_classes, depth=4):
        super().__init__()
        self.layers = nn.ModuleList(
            [S4Layer(d_model) for _ in range(depth)]
        )
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=1)
        return self.head(x)
