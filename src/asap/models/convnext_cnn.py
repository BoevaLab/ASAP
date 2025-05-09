import torch
import torch.nn as nn
from .layers.unmap_predictor import UnmapPredictor
from .layers.convnext import ConvNeXtV2Block

class ConvNeXtCNN(nn.Module):
    def __init__(self, window_size=2048, bin_size=4, nr_tracks=1, use_map=False):
        super().__init__()
        assert window_size % bin_size == 0
        nr_bins = window_size // bin_size
        self.use_map = use_map

        self.stem = nn.Sequential(
            ConvNeXtV2Block(4, 128, 15),
            nn.MaxPool1d(2),
            #nn.Dropout(0.2)
        )

        if self.use_map:
            self.unmap_predictor = UnmapPredictor(channels_in=128)

        self.core = nn.Sequential(
            ConvNeXtV2Block(128, 256, 15, groups=True),
            nn.MaxPool1d(4),
            nn.Dropout(0.2),

            ConvNeXtV2Block(256, 512, 15, groups=True),
            nn.MaxPool1d(4),
            nn.Dropout(0.2),

            ConvNeXtV2Block(512, 512, 15, groups=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
        )

        self.head = nn.Sequential(
            nn.Linear(512, int(nr_bins // 32) * 32),
            nn.ReLU(),
            nn.Linear(int(nr_bins // 32) * 32, int(nr_bins // 32)),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor, return_unmap=False) -> torch.Tensor:
        x = torch.transpose(x, -1, -2)
        x = self.stem(x)
        if return_unmap:
            u = self.unmap_predictor(torch.transpose(x, dim0=-1, dim1=-2))
        x = self.core(x)
        x = torch.transpose(x, -1, -2)
        x = self.head(x)
        x = x.reshape((x.shape[0], 512, 1))
        if return_unmap:
            return x, u
        return x
