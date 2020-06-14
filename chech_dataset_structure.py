# -*- coding: utf-8 -*-
# ---------------------

import click
import termcolor
from path import Path


@click.command()
@click.argument('ds_path', type=click.Path(exists=True))
def main(ds_path):
    # type: (str) -> None

    RED_BALL = termcolor.colored('⬤', 'red')
    YELLOW_BALL = termcolor.colored('⬤', 'yellow')
    GREEN_BALL = termcolor.colored('⬤', 'green')

    ds_path = Path(ds_path)

    required_dirs = [ds_path / 'train', ds_path / 'test']
    for required_dir in required_dirs:
        if not required_dir.exists():
            print(f'{RED_BALL} ERROR: \'{required_dir.abspath()}\' does not exist')
            exit(-1)

    train_classes = set([str(d.basename()) for d in required_dirs[0].dirs()])
    test_classes = set([str(d.basename()) for d in required_dirs[1].dirs()])

    if len(train_classes - test_classes) != 0:
        print(f'{RED_BALL} ERROR: train classes {train_classes} do not match test classes {test_classes}')
        exit(-1)

    if len(train_classes) < 2 or len(test_classes) < 2:
        print(f'{RED_BALL} ERROR: you must have at least 2 classes; you currently have {len(train_classes)}')
        exit(-1)

    print(f'{GREEN_BALL} OK: the structure of your dataset seems good :)')

if __name__ == '__main__':
    main()
