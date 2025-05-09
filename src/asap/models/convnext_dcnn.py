import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from .layers.unmap_predictor import UnmapPredictor
from .layers.convnext import ConvNeXtV2Block

DEFAULT = {
    'window_size': 2048,
    'bin_size': 4,
    'residual_blocks': 11,
    'dilation_mult': 1.5,
    'filters0': 256,
    'filters1': 128,
    'filters3': 2048,
    'kernel0': 15,
    'kernel1': 3,
    'kernel2': 1,
    'dropout': 0.3,
    'final_dropout': 0.05,
    'use_map': False
}



class ConvNeXtDCNN(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        config = DEFAULT.copy()
        config.update(kwargs)
        print('Using basenji with config:', config)

        window = config['window_size']
        self.use_map = config['use_map']
        self.init_conv = ConvNeXtV2Block(channels_in=4, channels_out=config['filters0'],
                                   kernel_size=config['kernel0'])
        self.init_pool = nn.MaxPool1d(kernel_size=2)

        if self.use_map:
            self.unmap_predictor = UnmapPredictor(channels_in=config['filters0'])

        self.core = BasenjiCoreBlock(nr_tracks=1, window=window, filters_in=config['filters0'],
                      nr_res_blocks=config['residual_blocks'],
                      rate_mult=config['dilation_mult'],
                      bin_size=config['bin_size'],
                      filters1=config['filters1'],
                      filters3=config['filters3'],
                      kernel1=config['kernel1'],
                      kernel2=config['kernel2'],
                      dropout=config['dropout'],
                      final_dropout=config['final_dropout'])
    
    def forward(self, x:torch.Tensor, return_unmap=False) -> torch.Tensor:
        x = torch.transpose(x, dim0=-1, dim1=-2)
        x = self.init_conv(x)
        x = F.pad(x, (1, 0))
        x = self.init_pool(x)
        if return_unmap:
            u = F.max_pool1d(x, 2)
            u = self.unmap_predictor(torch.transpose(u, dim0=-1, dim1=-2))
        x = self.core(x)
        if return_unmap:
            return x, u
        return x


class BasenjiCoreBlock(nn.Module):
    def __init__(self, nr_tracks: int, window: int, filters_in,
                  nr_res_blocks: int = 11, rate_mult: float = 1.5, bin_size: int = 100, filters1: int = 128,
                  filters3: int = None, kernel1: int = 3, kernel2: int = 1, dropout: float = 0.3,
                  final_dropout: float = 0.05):
        super().__init__()
        if not filters3:
            filters3 = window
        dconv_blocks = []
        conv_blocks = []
        dilation_rate = 1.0
        self.nr_res_blocks = nr_res_blocks
        self.dropout = nn.Dropout(p=dropout)
        for _ in range(self.nr_res_blocks):
            d_conv_block = ConvBlock(filters_in, filters1, kernel_size=kernel1, dilation_rate=int(np.round(dilation_rate)))
            dconv_blocks.append(d_conv_block)
            conv_block = ConvBlock(filters1, filters_in,  kernel_size=kernel2, bn_gamma='zeros')
            conv_blocks.append(conv_block)
            dilation_rate *= rate_mult

        self.dconv_blocks = nn.ModuleList(dconv_blocks)
        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.final_conv = ConvBlock(filters_in, channels_out=filters3)
        self.final_dropout = nn.Dropout(p=final_dropout)
        pool_size = bin_size // 2
        self.pool = nn.AvgPool1d(kernel_size=pool_size, stride=pool_size)

        self.linear_out = nn.Linear(
            in_features=filters3,
            out_features=nr_tracks,
        )
        self.activation = nn.Softplus()
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        for i in range(self.nr_res_blocks):
            last_block_x = x
            # dilated conv
            x = self.dconv_blocks[i](x)

            # normal conv
            x = self.conv_blocks[i](x)
            x = self.dropout(x)

            # add residual
            x = x + last_block_x
        x = self.final_conv(x)
        x = self.final_dropout(x)
        x = self.pool(x)
        x = torch.transpose(x, -2, -1)
        x = self.linear_out(x)
        x = self.activation(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, channels_in, channels_out:int=1 , kernel_size:int=1 , dilation_rate:int=1, bn_gamma=None) -> None:
        super().__init__()
        self.activation = nn.GELU()
        self.conv = nn.Conv1d(
            channels_in,
            channels_out,
            kernel_size,
            bias=False, # no need if batchnorm after conv layer
            dilation=dilation_rate,
            padding='same'
        )
        self.bn = nn.BatchNorm1d(channels_out, momentum=0.1)
        if bn_gamma is not None:
            if bn_gamma == 'zeros':
                self.bn.weight = nn.Parameter(torch.zeros_like(self.bn.weight))
            elif bn_gamma == 'ones':
                # default of BatchNorm1d
                # but let's be explicit
                self.bn.weight = nn.Parameter(torch.ones_like(self.bn.weight))

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(x)
        x = self.conv(x)
        x = self.bn(x)
        return x
