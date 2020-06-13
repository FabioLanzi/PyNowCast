# -*- coding: utf-8 -*-
# ---------------------

from abc import ABCMeta
from abc import abstractmethod
from typing import *

import torch
from path import Path
from torch import nn


class BaseModel(nn.Module, metaclass=ABCMeta):

    def __init__(self):
        super().__init__()


    @abstractmethod
    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        """
        Defines the computation performed at every call.
        Should be overridden by all subclasses.
        """
        ...


    @property
    def n_param(self):
        # type: (BaseModel) -> int
        """
        :return: number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


    @property
    def is_cuda(self):
        # type: () -> bool
        """
        :return: `True` if the model is on Cuda; `False` otherwise
        """
        return next(self.parameters()).is_cuda


    def save_w(self, path):
        # type: (Union[str, Path]) -> None
        """
        save model weights in the specified path
        """
        torch.save(self.state_dict(), path)


    def load_w(self, path):
        # type: (Union[str, Path]) -> None
        """
        load model weights from the specified path
        """
        self.load_state_dict(torch.load(path))


    def requires_grad(self, flag):
        # type: (bool) -> None
        """
        :param flag: True if the model requires gradient, False otherwise
        """
        for p in self.parameters():
            p.requires_grad = flag
