#!/usr/bin/env python
# coding: utf-8

import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST as Dataset
from torchvision.transforms import transforms

import pytorch_lightning as pl


class SugoiModel(pl.LightningModule):

    def __init__(self):
        super(SugoiModel, self).__init__()

        # construct model here
        self.l1 = torch.nn.Linear(28*28, 512)
        self.l2 = torch.nn.Linear(512, 10)

    def forward(self, x):
        h = x.view(x.size(0), -1)
        h = torch.relu(self.l1(h))
        h = self.l2(h)
        return h

    def training_step(self, batch, batch_nb):
        # REQUIRED
        x, y = batch
        y_hat = self.forward(x)
        return {'loss': F.cross_entropy(y_hat, y)}

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        return {'val_loss': F.cross_entropy(y_hat, y)}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'avg_val_loss': avg_loss}

    def configure_optimizers(self):
        # REQUIRED
        return [torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)]

    @pl.data_loader
    def tng_dataloader(self):
        # REQUIRED
        return DataLoader(
            Dataset(
                os.path.join(os.getcwd(), 'datasets'),
                train=True,
                download=True,
                transform=transforms.ToTensor()),
            batch_size=32
        )

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(
            Dataset(
                os.path.join(os.getcwd(), 'datasets'),
                train=False,
                download=True,
                transform=transforms.ToTensor()),
            batch_size=32
        )


from pytorch_lightning import Trainer
from test_tube import Experiment

# prepare model to train
model = SugoiModel()

# prepare trainer
exp = Experiment(save_dir=os.path.join(os.getcwd(), 'logs'))
trainer = Trainer(experiment=exp)

# view tensorflow logs 
print('View tensorboard logs by running\ntensorboard --logdir %s' % os.getcwd())
print('and going to http://localhost:6006 on your browser')

# train
trainer.fit(model)
