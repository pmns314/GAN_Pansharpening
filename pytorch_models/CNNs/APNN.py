import os
import shutil
from abc import ABC

import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset.DatasetPytorch import DatasetPytorch
from constants import ROOT_DIR
from pytorch_models.CNNs.CnnInterface import CnnInterface


class APNN(CnnInterface, ABC):
    def __init__(self, channels, device="cpu", name="APNN"):
        super(APNN, self).__init__(device, name)
        self._model_name = name
        self.channels = channels
        self.conv1 = nn.Conv2d(in_channels=channels + 1, out_channels=64, kernel_size=(9, 9), padding='same',
                               padding_mode='replicate', bias=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(5, 5), padding='same',
                               padding_mode='replicate', bias=True)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=channels, kernel_size=(5, 5), padding='same',
                               padding_mode='replicate', bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, ms, pan):
        inputs = torch.cat([ms, pan], 1)
        rs = self.conv1(inputs)
        rs = self.relu(rs)
        rs = self.conv2(rs)
        rs = self.relu(rs)
        out = self.conv3(rs)

        # Skip connection converts the model in a residual model
        out = ms + out
        return out

    def train_step(self, dataloader):
        self.train(True)

        loss_batch = 0
        for batch, data in enumerate(dataloader):
            pan, ms, ms_lr, gt = data

            if len(pan.shape) == 3:
                pan = torch.unsqueeze(pan, 0)
            gt = gt.to(self.device)
            pan = pan.to(self.device)
            ms = ms.to(self.device)

            # Compute prediction and loss
            pred = self(ms, pan)
            loss = self.loss_fn(pred, gt)
            loss /= (32 * 32)

            # Backpropagation
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            loss = loss.item()
            torch.cuda.empty_cache()

            loss_batch += loss

        return loss_batch / (len(dataloader))

    def validation_step(self, dataloader):
        self.train(False)
        self.eval()
        running_vloss = 0.0
        i = 0
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                pan, ms, ms_lr, gt = data
                if len(pan.shape) == 3:
                    pan = torch.unsqueeze(pan, 0)
                gt = gt.to(self.device)
                pan = pan.to(self.device)
                ms = ms.to(self.device)

                voutputs = self(ms, pan)
                vloss = self.loss_fn(voutputs, gt)
                running_vloss += vloss.item()

        avg_vloss = running_vloss / (i + 1)

        return avg_vloss

    def generate_output(self, pan, **kwargs):
        return self(kwargs['ms'], pan)


