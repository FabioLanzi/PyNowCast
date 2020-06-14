# -*- coding: utf-8 -*-
# ---------------------

import click
import termcolor
from path import Path
import numpy as np
import constants


RED_BALL = termcolor.colored('⬤', 'red')
YELLOW_BALL = termcolor.colored('⬤', 'yellow')
GREEN_BALL = termcolor.colored('⬤', 'green')


def count_files(directory):
    # type: (Path) -> int
    count = 0
    for file in directory.files():
        if file.basename() != 'sensors.json':
            count += 1
    return count


def has_sensors_file(directory):
    # type: (Path) -> bool
    return (directory / 'sensors.json').exists()


def has_errors(train_dir, test_dir):
    # type: (Path, Path) -> bool
    required_dirs = [train_dir, test_dir]
    for required_dir in required_dirs:
        if not required_dir.exists():
            print(f'{RED_BALL} ERROR: \'{required_dir.abspath()}\' does not exist')
            return True

    train_classes = set([str(d.basename()) for d in train_dir.dirs()])
    test_classes = set([str(d.basename()) for d in test_dir.dirs()])

    if len(train_classes - test_classes) != 0:
        print(f'{RED_BALL} ERROR: train classes {train_classes} do not match test classes {test_classes}')
        return True

    if len(train_classes) < 2 or len(test_classes) < 2:
        print(f'{RED_BALL} ERROR: you must have at least 2 classes; you currently have {len(train_classes)}')
        return True

    n1 = np.sum([has_sensors_file(d) for d in train_dir.dirs()])
    n2 = np.sum([has_sensors_file(d) for d in test_dir.dirs()])
    if (n1 != 0 or n2 != 0) and (n1 != len(train_classes) or n2 != len(test_classes)):
        print(f'{RED_BALL} ERROR: one or more class directories does not contain the `sensors.json` file\n'
              f'\t>> this file is optional, but if a class contains it, then all the others must do the same')
        return True

    return False


def has_warnings(train_dir):
    # type: (Path) -> bool
    n_training_elements = count_files(train_dir)
    if count_files(train_dir) < constants.SUGGESTED_TRAIN_LEN:
        print(f'{YELLOW_BALL} WARNING: your training set seems a little small ({n_training_elements} elements)')
        return True
    return False


@click.command()
@click.argument('ds_path', type=click.Path(exists=True))
def main(ds_path):
    # type: (str) -> None

    ds_path = Path(ds_path)
    train_dir = ds_path / 'train'
    test_dir = ds_path / 'test'

    # find errors
    if has_errors(train_dir=train_dir, test_dir=test_dir):
        exit(-1)

    # find warnings
    if has_warnings(train_dir=train_dir):
        pass
    else:
        print(f'{GREEN_BALL} OK: the structure of your dataset seems good :)')


if __name__ == '__main__':
    main()
