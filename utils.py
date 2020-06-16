# -*- coding: utf-8 -*-
# ---------------------


from typing import Union

import PIL
import torch
import torchvision
from PIL.Image import Image
from path import Path


def pre_process_img(img):
    # type: (Image) -> torch.Tensor
    """
    Apply pre-processing to input image (PIL image)
    >> the smaller dimension (H or W) will be resized to 256
    >> the other dimension will be scaled accordingly keeping the aspect ratio as far as possible
       with the constraint that is divisible by 16
    >> the image is also converted to torch.Tensor and the values are normalized between 0 and 1
    :param img: image without pre processing
    :return: pre-processed image
    """
    w, h = img.size
    scale_factor = 256 / min(h, w)
    h = int(round(h * scale_factor))
    w = int(round(w * scale_factor))
    new_h = int(round(h / 16 + 0.5)) * 16
    new_w = int(round(w / 16 + 0.5)) * 16

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((new_h, new_w)),
        torchvision.transforms.ToTensor()
    ])
    img = transform(img)

    return img


def imread(path):
    # type: (Union[Path, str]) -> Image
    """
    Read image (PIL Image) from a given path.
    :param path: image path
    :return: PIL image
    """
    with open(path, 'rb') as f:
        with PIL.Image.open(f) as img:
            return img.convert('RGB')
