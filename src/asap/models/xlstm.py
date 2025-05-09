import torch
from torch import nn
from .layers.unmap_predictor import UnmapPredictor
from .layers.convnext import ConvNeXtV2Block

from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)


class ConvNeXt_XLSTM(nn.Module):
    def __init__(
        self,
        channels=128,
        stem_k_size=15,
        conv_inv_bottleneckscale=4,
        conv_grn=True,
        conv_tower_k_size=7,
        conv_tower_layers=4,
        core_layers=4,
        core_inv_bottleneckscale=2,
        nr_tracks=1,
        use_map=False
    ) -> None:
        super().__init__()
        self.use_map = use_map

        self.stem = Stem(
            channels_in=4,
            channels_out=channels,
            k_size=stem_k_size,
            inv_bottleneckscale=conv_inv_bottleneckscale,
        )

        if self.use_map:
            self.unmap_predictor = UnmapPredictor(channels_in=channels)
        self.conv_tower = ConvTower(
            channels,
            k_size=conv_tower_k_size,
            inv_bottleneckscale=conv_inv_bottleneckscale,
            grn=conv_grn,
            n_layers=conv_tower_layers,
        )

        xlstm_cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="cuda",
                    num_heads=4,
                    conv1d_kernel_size=4,
                    bias_init="powerlaw_blockdependent",
                ),
                feedforward=FeedForwardConfig(proj_factor=core_inv_bottleneckscale, act_fn="gelu"),
            ),
            context_length=512,
            num_blocks=core_layers,
            embedding_dim=channels,
            slstm_at=[1],

        )
        
        self.core = (
            xLSTMBlockStack(
                xlstm_cfg
            )
            if core_layers
            else nn.Identity()
        )

        self.out = OutBlock(channels, nr_tracks)

    def forward(self, x: torch.Tensor, return_unmap=False) -> torch.Tensor:
        x = torch.transpose(x, dim0=-1, dim1=-2)
        x = self.stem(x)
        if return_unmap:
            u = self.unmap_predictor(torch.transpose(x, dim0=-1, dim1=-2))
        x = self.conv_tower(x)
        x = torch.transpose(x, dim0=-1, dim1=-2)
        x = self.core(x)
        x = self.out(x)
        if return_unmap:
            return x, u
        return x


class Stem(nn.Module):
    def __init__(self, channels_in, channels_out, k_size, inv_bottleneckscale) -> None:
        super().__init__()
        self.conv1 = ConvNeXtV2Block(
            channels_in,
            channels_out,
            kernel_size=k_size,
            inv_bottleneckscale=inv_bottleneckscale,
        )
        self.pool = nn.MaxPool1d(4)
        '''
        self.conv2 = ConvNeXtV2Block(
            channels_out,
            channels_out,
            kernel_size=k_size,
            inv_bottleneckscale=inv_bottleneckscale,
            groups=True,
        )
        '''

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.pool(x)
        #x = self.conv2(x)
        #x = self.pool(x)
        return x


class ConvTower(nn.Module):
    def __init__(self, channels, k_size, inv_bottleneckscale, grn, n_layers=2):
        super().__init__()
        self.conv_list = nn.Sequential(
            *[
                ConvNeXtV2Block(
                    channels,
                    channels,
                    k_size,
                    inv_bottleneckscale=inv_bottleneckscale,
                    grn=grn,
                    groups=True,
                )
                for i in range(n_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_list(x)
        return x


class OutBlock(nn.Module):
    def __init__(self, channels, nr_tracks: int = 1) -> None:
        super().__init__()
        self.linear_out = nn.Linear(
            in_features=channels,
            out_features=nr_tracks,
        )
        self.activation = nn.Softplus()

    def forward(self, x: torch.Tensor):
        x = self.linear_out(x)
        x = self.activation(x)
        return x
