import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.unmap_predictor import UnmapPredictor
from .layers.convnext import ConvNeXtV2Block

class ConvNeXtLSTM(nn.Module):
    def __init__(self, window_size: int = 2048, bin_size: int = 4, nr_tracks=1, use_map: bool = False):
        super().__init__()
        assert window_size % bin_size == 0, f'Window size {window_size} must be divisible by bin size {bin_size}'
        self.nr_bins = window_size // bin_size
        self.use_map = use_map

        #self.conv1 = nn.Conv1d(in_channels=4, out_channels=256, kernel_size=26)
        self.conv1 = ConvNeXtV2Block(4, 256, kernel_size=26, inv_bottleneckscale=4)
        self.maxpool = nn.MaxPool1d(kernel_size=4)
        self.dropout1 = nn.Dropout(p=0.2)

        if self.use_map:
            self.unmap_predictor = UnmapPredictor(channels_in=256)

        self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True)

        self.fc1 = nn.Linear(256 * 2, 256)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, self.nr_bins)

    def forward(self, x, return_unmap=False) -> torch.Tensor:
        x = torch.transpose(x, -1, -2)
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        # x = self.dropout1(x)
        if return_unmap:
            u = self.unmap_predictor(torch.transpose(x, dim0=-1, dim1=-2))

        x = torch.transpose(x, -1, -2)
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take only the last output of the sequence
        x = self.dropout1(x)

        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.softplus(self.fc2(x))

        x = x.unsqueeze(-1)
        if return_unmap:
            return x, u
        return x