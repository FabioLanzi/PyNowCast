# -*- coding: utf-8 -*-
# ---------------------

from time import time

import click
import numpy as np
import torch
from path import Path
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import conf
from dataset.nowcast_ds import NowCastDS
from models import NCClassifier
from progress_bar import ProgressBar


class Trainer(object):

    def __init__(self, exp_name, ds_root_path, device):
        # type: (str, str, str) -> None

        self.exp_name = exp_name
        self.device = device

        # init train loader
        training_set = NowCastDS(ds_root_path=ds_root_path, mode='train', create_cache=True)
        self.train_loader = DataLoader(
            dataset=training_set, batch_size=conf.FX_BATCH_SIZE,
            num_workers=conf.FX_N_WORKERS, shuffle=True, pin_memory=True,
        )

        # init test loader
        test_set = NowCastDS(ds_root_path=ds_root_path, mode='test', create_cache=True)
        self.test_loader = DataLoader(
            dataset=test_set, batch_size=2,
            num_workers=0, shuffle=False, pin_memory=True,
        )

        # init model
        n_classes = len(training_set.classes)
        self.model = NCClassifier(n_classes=n_classes)
        self.model = self.model.to(device)

        # init optimizer
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=conf.FX_LR)

        # init logging stuffs
        self.log_path = Path(__file__).parent / 'log' / exp_name
        print(f'▶ You can monitor training progress with Tensorboard')
        if not self.log_path.exists():
            self.log_path.makedirs()
        print(f'└── tensorboard --logdir={self.log_path.parent}\n')
        self.sw = SummaryWriter(self.log_path)
        self.train_losses = []
        self.test_losses = []

        # starting values
        self.epoch = 0
        self.best_test_accuracy = None
        self.patience = conf.FX_PATIENCE

        # init progress bar
        self.progress_bar = ProgressBar(max_step=len(self.train_loader), max_epoch=conf.FX_EPOCHS)

        # possibly load checkpoint
        self.load_ck()


    def load_ck(self):
        """
        load training checkpoint
        """
        ck_path = self.log_path / 'training.ck'
        if ck_path.exists():
            ck = torch.load(ck_path)
            print(f'[loading checkpoint \'{ck_path}\']')
            self.epoch = ck['epoch']
            self.progress_bar.current_epoch = self.epoch
            self.model.load_state_dict(ck['model'])
            self.optimizer.load_state_dict(ck['optimizer'])
            self.best_test_accuracy = self.best_test_accuracy


    def save_ck(self):
        """
        save training checkpoint
        """
        ck = {
            'epoch': self.epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_test_loss': self.best_test_accuracy
        }
        torch.save(ck, self.log_path / 'training.ck')


    def train(self):
        """
        train model for one epoch on the Training-Set.
        """
        start_time = time()
        self.model.train()

        for step, sample in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            x, y_true = sample
            x = x.to(self.device)
            y_true = y_true.to(self.device)

            y_pred = self.model.forward(x)
            loss = nn.NLLLoss()(torch.log(y_pred), y_true)
            loss.backward()
            self.train_losses.append(loss.item())

            self.optimizer.step(None)

            # print an incredible progress bar
            print(f'\r{self.progress_bar} │ Loss: {np.mean(self.train_losses):.6f}', end='')
            self.progress_bar.inc()

        # log average loss of this epoch
        mean_epoch_loss = np.mean(self.train_losses)
        self.sw.add_scalar(tag='train_loss', scalar_value=mean_epoch_loss, global_step=self.epoch)
        self.train_losses = []

        # log epoch duration
        print(f' │ T: {time() - start_time:.2f} s')


    def test(self):
        """
        test model on the Test-Set
        """
        self.model.eval()

        num = 0
        den = 0
        for step, sample in enumerate(self.test_loader):
            x, y_true = sample
            x = x.to(self.device)
            y_true = y_true.to(self.device)

            y_pred = self.model.forward(x)
            y_pred = torch.argmax(y_pred, dim=0)

            den += y_true.shape[0]
            num += torch.sum(y_pred == y_true).item()

        # log average loss on test set
        test_accuracy = num / den
        print(f'\t● Accuracy on TEST-set: {test_accuracy*100:.2f} │ patience: ', end='')
        self.sw.add_scalar(tag='test_accuracy', scalar_value=test_accuracy, global_step=self.epoch)

        # save best model and update training patience
        if self.best_test_accuracy is None or test_accuracy > self.best_test_accuracy:
            self.best_test_accuracy = test_accuracy
            self.patience = conf.FX_PATIENCE
            torch.save(self.model.state_dict(), self.log_path / 'best.pth')
        else:
            self.patience = self.patience - 1
        print(f'{self.patience}/{conf.FX_PATIENCE}')


    def run(self):
        """
        start model training procedure (train > test > checkpoint > repeat)
        """
        for _ in range(self.epoch, conf.FX_EPOCHS):
            self.train()

            with torch.no_grad():
                self.test()

            self.epoch += 1
            self.save_ck()


H1 = 'experiment name: string without spaces'
H2 = 'absolute path of your dataset root directory'
H3 = 'device used to train the model; example: "cuda", "cuda:0", "cuda:1", ...'


@click.command()
@click.option('--exp_name', type=str, required=True, help=H1)
@click.option('--ds_root_path', type=click.Path(exists=True), required=True, help=H2)
@click.option('--device', type=str, default='cuda', show_default=True, help=H3)
def main(exp_name, ds_root_path, device):
    # type: (str, str, str) -> None
    trainer = Trainer(
        exp_name=exp_name,
        ds_root_path=ds_root_path,
        device=device
    )
    trainer.run()


if __name__ == '__main__':
    main()
