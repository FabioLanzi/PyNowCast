# -*- coding: utf-8 -*-
# ---------------------

from typing import List
from typing import Tuple

import click
import numpy as np
import torch
from torch import nn

import utils
from models import NCClassifier


H1 = 'path of the image you want to classify'
H2 = 'path of the `.pync` containing model weights'
H3 = 'optional list of sensor(s) data -> ";"-separated values (no spaces) | example: "1.5;4;7.1;100"'
H4 = 'device you want to use | example: "cuda", "cuda:0", "cuda:1", ...'


def classify(img_path, pync_file_path, sensor_data, device):
    # type: (str, str, str, str) -> Tuple[List[str], np.ndarray]
    """
    :param img_path: path of the image you want to classify
    :param pync_file_path: path of the `.pync` containing model weights
    :param sensor_data: optional list of sensor(s) data
        >> ";"-separated values (no spaces)
        >> example: "1.5;4;7.1;100"
    :param device: device you want to use
        >> example: "cuda", "cuda:0", "cuda:1", ..
    :return: list of all classes and index of predicted class in that list
    """

    # read data from pnc_file
    pnc_dict = torch.load(pync_file_path)
    classes = pnc_dict['classes']
    sensor_data_len = pnc_dict['sensor_data_len']
    s_min, s_max = pnc_dict['sdata_range']

    # init nowcasting model
    model = NCClassifier(n_classes=len(classes), sensor_data_len=sensor_data_len)
    model.load_state_dict(pnc_dict['model_weights'])
    model.requires_grad(False)
    model.to(device)

    # read input image
    img = utils.imread(img_path)
    img = utils.pre_process_img(img).unsqueeze(0)
    img = img.to(device).float()

    # parse sensor(s) data
    if sensor_data is not None:
        sdata = []
        for x in sensor_data.split(';'):
            try:
                sdata.append(float(x))
            except ValueError:
                sdata.append(None)
        sdata = np.array(sdata, dtype=np.float)
        sdata = (sdata - s_min) / (s_max - s_min)
        sdata[np.isnan(sdata)] = -1
        sdata = torch.tensor(sdata).unsqueeze(0).to(device).float()
        assert len(sdata) == sensor_data_len, \
            'ERROR: the length of the input `sensor_data` array does not match ' \
            'the length specified in the` .pync` file.'
    else:
        sdata = None

    # predict class using nowcasting model
    y_pred = model.forward(img, sensor_data=sdata)
    probabilities = nn.Softmax(dim=1)(y_pred)
    probabilities = probabilities.squeeze().to('cpu').numpy()

    return classes, probabilities


@click.command()
@click.option('--img_path', type=click.Path(exists=True), required=True, help=H1)
@click.option('--pync_file_path', type=click.Path(exists=True), required=True, help=H2)
@click.option('--sensor_data', type=str, default=None, show_default=True, help=H3)
@click.option('--device', type=str, default='cuda', show_default=True, help=H4)
def main(img_path, pync_file_path, sensor_data, device):
    # type: (str, str, str, str) -> None

    print(f'\n▶ Classifying image \'{img_path}\'')

    classes, probabilities = classify(img_path, pync_file_path, sensor_data, device)
    selected_class = np.argmax(probabilities)

    # print formatted results
    fmt1 = len(str(len(classes) - 1))
    fmt2 = np.max([len(c) for c in classes]) + 4
    for i in range(len(classes)):
        c = f'[{classes[i]}]'
        p = probabilities[i] * 100
        branch = '└──' if i == len(classes) - 1 else '├──'
        print('{branch}[{i:0{q}d}]{c:─>{fmt}}: {p:.2f} %'.format(
            branch=branch, i=i, c=c, q=fmt1, fmt=fmt2, p=p
        ), end=' ❮⬤\n' if i == selected_class else '\n')


if __name__ == '__main__':
    main()
