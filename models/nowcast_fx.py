# -*- coding: utf-8 -*-
# ---------------------

from typing import Optional

import torch
from google_drive_downloader import GoogleDriveDownloader as gdd
from path import Path
from torch import nn

from models.base_model import BaseModel
from models.basic_conv2d import BasicConv2D
from models.residuals import MultipleResidualBlocks


PRETRAINED_WEIGHTS_URL = '1qms6RNZVlAPrjOnD4kjovJ8HcmNxdZ_4'


class NowCastFX(BaseModel):
    """
    NowCastFX = [NowCast]ing [F]eature e[X]tractor
    """


    def __init__(self, in_channels=3, pretrained=True):
        # type: (int, bool) -> None
        """
        This Autoencoder is vaguely inspired by DarkNet.

        :param in_channels: number of input channels
            * NOTE: in_channels = out_channels
        """
        super().__init__()

        self.encoder = nn.Sequential(
            # first conv
            BasicConv2D(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1),
            # down 1
            BasicConv2D(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=2),
            MultipleResidualBlocks(in_channels=32, num_blocks=1),
            # down 2
            BasicConv2D(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2),
            MultipleResidualBlocks(in_channels=64, num_blocks=2),
            # down 3
            BasicConv2D(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2),
            MultipleResidualBlocks(in_channels=128, num_blocks=4),
            # down 4
            BasicConv2D(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=2),
            MultipleResidualBlocks(in_channels=128, num_blocks=4),
            # last conv
            BasicConv2D(in_channels=128, out_channels=3, kernel_size=3, padding=1, activation='Sigmoid')
        )

        self.decoder = nn.Sequential(
            # first conv
            BasicConv2D(in_channels=3, out_channels=128, kernel_size=3, padding=1),
            # up 1
            MultipleResidualBlocks(in_channels=128, num_blocks=4),
            BasicConv2D(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=0.5),
            # up 2
            MultipleResidualBlocks(in_channels=128, num_blocks=4),
            BasicConv2D(in_channels=128, out_channels=64, kernel_size=3, padding=1, stride=0.5),
            # up 3
            MultipleResidualBlocks(in_channels=64, num_blocks=2),
            BasicConv2D(in_channels=64, out_channels=32, kernel_size=3, padding=1, stride=0.5),
            # up 4
            MultipleResidualBlocks(in_channels=32, num_blocks=1),
            BasicConv2D(in_channels=32, out_channels=16, kernel_size=3, padding=1, stride=0.5),
            # last conv
            BasicConv2D(in_channels=16, out_channels=in_channels, kernel_size=3, padding=1, activation='Linear')
        )

        self.kaiming_init(activation='LeakyReLU')
        if pretrained:
            weights_file_path = (Path(__file__).parent / 'nowcast_fx.pth')
            if not weights_file_path.exists():
                gdd.download_file_from_google_drive(
                    file_id='1qms6RNZVlAPrjOnD4kjovJ8HcmNxdZ_4',
                    dest_path=weights_file_path,
                )
            self.load_w(weights_file_path)


    def encode(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        return self.encoder(x)


    def decode(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        return self.decoder(x)


    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        code = self.encoder(x)
        return self.decoder(code)


    def flat_features(self, x, fixed_size=None, trainable=True):
        # type: (torch.Tensor, Optional[int], bool) -> torch.Tensor
        if trainable:
            x = self.encoder(x).view(x.shape[0], -1)
        else:
            with torch.no_grad():
                x = self.encoder(x).view(x.shape[0], -1)

        if fixed_size is not None:
            x = nn.AdaptiveAvgPool1d(fixed_size)(x.unsqueeze(1)).squeeze(1)

        return x


# ---------

def debug(batch_size=2, device=None):
    #  type: (int, Optional[str]) -> None
    import torchsummary

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x = torch.rand((batch_size, 3, 256, 341)).to(device)

    model = NowCastFX(pretrained=True).to(device)
    model.requires_grad(True)
    model.train()

    torchsummary.summary(model=model, input_size=x.shape[1:], device=str(device))

    print(f'\n▶ batch size: {batch_size}')
    print(f'▶ device: \'{device}\'')

    print('\n▶ ENCODING')
    y = model.encode(x)
    print(f'├── input shape: {tuple(x.shape)}')
    print(f'└── output shape: {tuple(y.shape)}')

    print('\n▶ DECODING')
    xd = model.decode(y)
    print(f'├── input shape: {tuple(y.shape)}')
    print(f'└── output shape: {tuple(xd.shape)}')

    print('\n▶ FORWARD')
    y = model.forward(x)
    print(f'├── input shape: {tuple(x.shape)}')
    print(f'└── output shape: {tuple(y.shape)}')


if __name__ == '__main__':
    debug(batch_size=2)
