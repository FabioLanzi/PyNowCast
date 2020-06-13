import torch
from path import Path
from torch import nn

from models import BaseModel


class BasicConv2D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1, with_batch_norm=True):
        super().__init__()

        self.with_batch_norm = with_batch_norm

        # 2D convolution
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=(not self.with_batch_norm)
        )

        # batch normalization
        self.bn = nn.BatchNorm2d(out_channels)

        # activation: LeakyReLU
        self.activation = nn.LeakyReLU(inplace=True)


    def forward(self, x):
        x = self.conv(x)
        if self.with_batch_norm:
            x = self.bn(x)
        x = self.activation(x)
        return x


class SingleResidual(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        t = in_channels
        self.layer1 = BasicConv2D(in_channels=t, out_channels=t // 2, kernel_size=1, padding=0)
        self.layer2 = BasicConv2D(in_channels=t // 2, out_channels=t, kernel_size=3, padding=1)


    def forward(self, x):
        residual = x
        y = self.layer1(x)
        y = self.layer2(y)
        return y + residual


class DarkResidual(nn.Module):

    def __init__(self, in_channels, num_blocks):
        super().__init__()
        layers = []
        for i in range(num_blocks):
            layers.append(SingleResidual(in_channels=in_channels))
        self.residuals = nn.Sequential(*layers)


    def forward(self, x):
        return self.residuals(x)


class Darknet53(BaseModel):

    def __init__(self, pretrained=True):
        super(Darknet53, self).__init__()

        self.conv_features = nn.Sequential(
            # shape: (3, H, W)
            BasicConv2D(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            # shape: (32, H, W)
            BasicConv2D(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2),
            DarkResidual(in_channels=64, num_blocks=1),
            # shape: (64, H/2, W/2)
            BasicConv2D(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2),
            DarkResidual(in_channels=128, num_blocks=2),
            # shape: (128, H/4, W/4)
            BasicConv2D(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=2),
            DarkResidual(in_channels=256, num_blocks=8),
            # shape: (256, H/8, W/8)
            BasicConv2D(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=2),
            DarkResidual(in_channels=512, num_blocks=8),
            # shape: (512, H/16, W/16)
            BasicConv2D(in_channels=512, out_channels=1024, kernel_size=3, padding=1, stride=2),
            DarkResidual(in_channels=1024, num_blocks=4)
            # shape: (1024, H/32, W/32)
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        if pretrained:
            self.load_w(Path(__file__).parent / 'darknet.pth')


    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        x = self.conv_features(x)
        return x


    def features(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        return self.forward(x)


    def flat_features(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        x = self.conv_features(x)
        x = self.global_avg_pool(x)
        return x.view(-1, 1024)


def main():
    batch_size = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Darknet53().to(device)
    model.load_w('darknet.pth')

    print(model)
    print(f'* number of parameters: {model.n_param}')

    x = torch.rand((batch_size, 3, 256, 341)).to(device)

    import time

    t = time.time()
    y = model.forward(x)
    t = time.time() - t

    print(f'* input shape: {tuple(x.shape)}')
    print(f'* output shape: {tuple(y.shape)}')
    print(f'* forward time: {t:.4f} s with a batch size of {batch_size}')


if __name__ == '__main__':
    main()
