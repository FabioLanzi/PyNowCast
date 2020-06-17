# -*- coding: utf-8 -*-
# ---------------------

import torch
from torch import nn

from models.base_model import BaseModel
from models.nowcast_fx import NowCastFX


FEATURE_VECTOR_SIZE = 768


class NCClassifier(BaseModel):

    def __init__(self, n_classes=2):
        super().__init__()

        self.extractor = NowCastFX(pretrained=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(FEATURE_VECTOR_SIZE, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, n_classes),
            nn.Softmax(dim=1)
        )


    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        x = self.get_features(x)
        return self.classifier.forward(x)


    def get_features(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        return self.extractor.flat_features(x, fixed_size=FEATURE_VECTOR_SIZE)


# ---------

def debug(n_classes=3, batch_size=2, device=None):
    #  type: (int, int, Optional[str]) -> None
    import torchsummary

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x = torch.rand((batch_size, n_classes, 256, 352)).to(device)

    model = NCClassifier(n_classes=n_classes).to(device)
    model.requires_grad(True)
    model.train()

    torchsummary.summary(model=model, input_size=x.shape[1:], device=str(device))

    print(f'\n▶ number of classes: \'{n_classes}\'')
    print(f'▶ batch size: {batch_size}')
    print(f'▶ device: \'{device}\'')

    print('\n▶ FORWARD')
    y = model.forward(x)
    print(f'├── input shape: {tuple(x.shape)}')
    print(f'└── output shape: {tuple(y.shape)}')


if __name__ == '__main__':
    debug()
