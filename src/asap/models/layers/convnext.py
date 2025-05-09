import torch
import torch.nn as nn
import torch.nn.functional as F
from .grn import GRN1d

class ConvNeXtV2Block(nn.Module):
    """Adapted from ConvNeXt"""
    def __init__(
        self,
        channels_in,
        channels_out,
        kernel_size,
        groups=False,
        inv_bottleneckscale=4,
        grn=True,
        dilation_rate=1,
    ):
        super().__init__()
        self.res_early = channels_in == channels_out
        self.inv_bottleneckwidth = int(inv_bottleneckscale * channels_out)
        self.dwconv = nn.Conv1d(
            channels_in,
            channels_out,
            kernel_size=kernel_size,
            padding="same",
            groups=channels_in if groups else 1,
            dilation=dilation_rate,
        )  # depthwise conv
        self.norm = nn.LayerNorm(channels_out, eps=1e-6)
        self.pwconv1 = nn.Linear(
            channels_out, self.inv_bottleneckwidth
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        if grn:
            self.grn = GRN1d(self.inv_bottleneckwidth)
        else:
            self.grn = nn.Identity()
        self.pwconv2 = nn.Linear(self.inv_bottleneckwidth, channels_out)

    def forward(self, x: torch.Tensor):
        if self.res_early:
            x_ = x
            x = self.dwconv(x)
        else:
            x = self.dwconv(x)
            x_ = x

        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)

        x = x.permute(0, 2, 1)
        x = x_ + x
        return x
