# -*- coding: utf-8 -*-
# ---------------------

import torch
from torch import nn

from models.basic_conv2d import BasicConv2D


class SingleResidual(nn.Module):
    """
    A single residual block with 2 convolutional layers

    x →|---→[layer1]---→[layer2]----↓
       |                           [+]---→ y
       |----------------------------↑
    """


    def __init__(self, in_channels, kernel_size=3, padding=1, with_batch_norm=False, activation='LeakyReLU'):
        # type: (int, int, int, bool, str) -> None
        """
        :param in_channels: number of input (and output) channels
        :param kernel_size: kernel size of the 2D convolution of `layer2`
        :param padding: zero-padding added to both sides of the input (default = 0)
        :param with_batch_norm: do you want use batch normalization?
        :param activation: activation function you want to add after the convolutions
            * values in {'ReLU', 'LeakyReLU', 'Linear', 'Sigmoid', 'Tanh'}
        """
        super().__init__()

        self.layer1 = BasicConv2D(
            in_channels=in_channels, out_channels=in_channels // 2, kernel_size=1, stride=1,
            padding=0, activation=activation, with_batch_norm=with_batch_norm)

        self.layer2 = BasicConv2D(
            in_channels=in_channels // 2, out_channels=in_channels, kernel_size=kernel_size, stride=1,
            padding=padding, activation=activation, with_batch_norm=with_batch_norm
        )


    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        residual = x
        y = self.layer1(x)
        y = self.layer2(y)
        return y + residual


# ---------


class MultipleResidualBlocks(nn.Module):
    """
    A sequence of `num_blocks` cascade residual blocks
    """


    def __init__(self, in_channels, num_blocks, kernel_size=3, padding=1,
                 with_batch_norm=False, activation='LeakyReLU'):
        # type: (int, int, int, int, bool, str) -> None
        """
        :param in_channels: number of input (and output) channels
        :param num_blocks: total number of single residual blocks in the macro block
        :param kernel_size: kernel size of the 2D convolutions of `layer2` of each residual block
        :param padding: zero-padding added to both sides of the input (default = 0)
        :param with_batch_norm: do you want use batch normalization?
        :param activation: activation function you want to add after the convolutions
            * values in {'ReLU', 'LeakyReLU', 'Linear', 'Sigmoid', 'Tanh'}
        """
        super().__init__()

        layers = []
        for i in range(num_blocks):
            layers.append(SingleResidual(
                in_channels=in_channels, kernel_size=kernel_size, padding=padding,
                with_batch_norm=with_batch_norm, activation=activation)
            )

        self.residuals = nn.Sequential(*layers)


    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        return self.residuals(x)
