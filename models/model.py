# -*- coding: utf-8 -*-
# ---------------------

import torch
from torch import nn

from models import BaseModel
from models.darknet import Darknet53
from models.nowcast_fx import NowCastFX

class RainModel(BaseModel):

    def __init__(self):
        super().__init__()

        self.f_extractor = NowCastFX(pretrained=True)
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(768, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )


    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        x = self.f_extractor.flat_features(x, trainable=False)
        return self.fc(x)


# ---------

def main():
    batch_size = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = RainModel().to(device)
    print(model)
    print(f'* number of parameters: {model.n_param}')

    x = torch.rand((batch_size, 3, 256, 256)).to(device)

    import time

    t = time.time()
    y = model.forward(x)
    t = time.time() - t

    print(f'* input shape: {tuple(x.shape)}')
    print(f'* output shape: {tuple(y.shape)}')
    print(f'* forward time: {t:.4f} s with a batch size of {batch_size}')


if __name__ == '__main__':
    main()
