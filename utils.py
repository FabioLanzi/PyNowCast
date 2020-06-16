# -*- coding: utf-8 -*-
# ---------------------

import matplotlib


matplotlib.use('Agg')

from matplotlib import figure
from matplotlib import cm
import numpy as np
import PIL
from PIL.Image import Image
from path import Path
from torchvision.transforms import ToTensor, RandomHorizontalFlip
from torch import Tensor
import torch
from typing import *
import torchvision


RandomHorizontalFlip = RandomHorizontalFlip

TRANSFORM = torchvision.transforms.Compose([
    torchvision.transforms.CenterCrop((900, 1280)),
    torchvision.transforms.Resize(290),
    torchvision.transforms.ToTensor()
])


def pre_process_img(img):
    w, h = img.size
    scale_factor = 256 / min(h, w)
    h = int(round(h * scale_factor))
    w = int(round(w * scale_factor))
    new_h = int(round(h/16 + 0.5))*16
    new_w = int(round(w/16 + 0.5))*16

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((new_h, new_w)),
        torchvision.transforms.ToTensor()
    ])
    img = transform(img)

    return img


def imread(path):
    # type: (Union[Path, str]) -> Image
    with open(path, 'rb') as f:
        with PIL.Image.open(f) as img:
            return img.convert('RGB')


def pyplot_to_numpy(figure):
    # type: (figure.Figure) -> np.ndarray
    figure.canvas.draw()
    x = np.fromstring(figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    x = x.reshape(figure.canvas.get_width_height()[::-1] + (3,))
    return x


def pyplot_to_tensor(figure):
    # type: (figure.Figure) -> Tensor
    x = pyplot_to_numpy(figure=figure)
    x = ToTensor()(x)
    return x


def apply_colormap_to_tensor(x, cmap='jet', range=(None, None)):
    # type: (Tensor, str, Optional[Tuple[float, float]]) -> Tensor
    """
    :param x: Tensor with shape (1, H, W)
    :param cmap: name of the color map you want to apply
    :param range: tuple of (minimum possible value in x, maximum possible value in x)
    :return: Tensor with shape (3, H, W)
    """
    cmap = cm.ScalarMappable(cmap=cmap)
    cmap.set_clim(vmin=range[0], vmax=range[1])
    try:
        x = x.cpu().numpy()
    except:
        x = x.detatch().cpu().numpy()
    x = x.squeeze()
    x = cmap.to_rgba(x)[:, :, :-1]
    return ToTensor()(x)


def main():
    x = torch.zeros((1, 100, 100)).cuda()
    x = apply_colormap_to_tensor(x, cmap='rainbow')
    print(x.shape)


if __name__ == '__main__':
    main()
