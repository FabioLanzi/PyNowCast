# -*- coding: utf-8 -*-
# ---------------------

import torch
from torch import nn


class BasicConv2D(nn.Module):
    """
    Basic 2D Convolution with optional batch normalization and activation function
    """


    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1,
                 with_batch_norm=False, activation='LeakyReLU'):
        # type: (int, int, int, int, float, bool, str) -> None
        """
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param kernel_size: kernel size of the 2D convolution
        :param padding: zero-padding added to both sides of the input (default = 0)
        :param stride: stride of the convolution (default = 1)
            * NOTE: if `stride` is < 1, a trasnpsposed 2D convolution with stride=1/`stride`
            is used instead of a normal 2D convolution
        :param with_batch_norm: do you want use batch normalization?
        :param activation: activation function you want to add after the convolution
            * values in {'ReLU', 'LeakyReLU', 'Linear', 'Sigmoid', 'Tanh'}
        """
        super().__init__()

        self.with_batch_norm = with_batch_norm

        # 2D convolution
        if stride >= 1:
            self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=int(stride),
                padding=padding,
                bias=(not self.with_batch_norm)
            )
        else:
            self.conv = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=int(1 / stride),
                padding=padding,
                output_padding=padding,
                bias=(not self.with_batch_norm)
            )

        # batch normalization
        if with_batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)

        # activation function
        assert activation in ['ReLU', 'LeakyReLU', 'Linear', 'Sigmoid', 'Tanh']
        if activation == 'ReLU':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'Linear':
            self.activation = lambda x: x
        elif activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'Tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.LeakyReLU(inplace=True)


    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        x = self.conv(x)
        if self.with_batch_norm:
            x = self.bn(x)
        x = self.activation(x)
        return x
