# -*- coding: utf-8 -*-
# ---------------------

import json
from typing import Tuple

import torch
from path import Path
from torch.utils.data import Dataset

import utils


class NowCastDS(Dataset):
    """
    Dataset class used to train and test nowcasting models.
    """


    def __init__(self, ds_root_path, mode, create_cache=False):
        # type: (str, str, bool) -> None
        """
        :param ds_root_path: absolute path of the root directory of your dataset
        :param mode: dataset working mode; values in {'train', 'test'}
        :param create_cache: if `True`, the image paths are saved in a JSON file to speed up subsequent loads
        """

        self.ds_root_path = Path(ds_root_path)

        self.mode = mode
        assert mode in {'train', 'test'}, f'mode must be one "train" or "test"'

        self.classes = [str(d.basename()) for d in (self.ds_root_path / self.mode).dirs()]
        self.classes.sort()

        cache_path = self.ds_root_path / self.mode / 'cache.json'
        if cache_path.exists() and not create_cache:
            print('▶ Loading dataset from cache file. Plase wait')
            with open(cache_path, 'r') as cache_file:
                self.img_paths = json.load(cache_file)
        else:
            self.img_paths = []
            print('▶ Loading dataset: it may take some time. Please wait')
            for class_dir in (self.ds_root_path / self.mode).dirs():
                print(f'└── Loading class \'{class_dir.basename()}\'', end='')
                for img_path in class_dir.files('*.jpg'):
                    s = img_path.split('/')
                    self.img_paths.append(f'{s[-2]}/{s[-1]}')
                print(f'\r├── Loading class \'{class_dir.basename()}\'')

        print('└── The dataset has been loaded')

        if create_cache:
            with open(cache_path, 'w') as cache_file:
                json.dump(self.img_paths, cache_file)
            print(f'▶ Created cahce file \'{cache_path}\'')


    def __len__(self):
        # type: () -> int
        """
        :return: dataset len (total number of images)
        """
        return len(self.img_paths)


    def __getitem__(self, i):
        # type: (int) -> Tuple[torch.Tensor, int]
        """
        :param i: image index
        :return: (img, img_class)
            >> img: RGB image -> torch.Tensor with shape (3, H, W) and values in range [0, 1]
            >> img_class: int value representing the image class
        """

        # select image
        img_path = self.ds_root_path / self.mode / self.img_paths[i]
        img = utils.imread(img_path)

        # read class
        img_class = self.classes.index(img_path.split('/')[-2])

        # aplly pre-processing:
        # >> resize and transform to `torch.Tensor`
        img = utils.pre_process_img(img)

        return img, img_class


    def class_int2str(self, class_int_value):
        # type: (int) -> str
        """
        :param class_int_value:
        :return:
        """
        return self.classes[class_int_value]


def main():
    ds = NowCastDS(
        ds_root_path=Path(__file__).parent / 'example_ds',
        mode='train',
        create_cache=True
    )

    for i in range(max(1, len(ds) // 2)):
        img, img_class = ds[i]
        print(
            f'\n▶ Dataset sample #{i}\n'
            f'├── image.shape: {tuple(img.shape)}\n'
            f'├── class_int_value: {img_class}\n'
            f'└── class_name: \'{ds.class_int2str(img_class)}\''
        )


if __name__ == '__main__':
    main()
