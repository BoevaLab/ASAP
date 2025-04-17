import torch
import torch.nn as nn

class UnmapPredictor(nn.Module):
    def __init__(self, channels_in):
        super().__init__()
        self.out = nn.Linear(channels_in, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        return self.activation(self.out(x))
