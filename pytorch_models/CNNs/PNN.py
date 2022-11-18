from abc import ABC

import torch
from torch import nn

from pytorch_models.CNNs.CnnInterface import CnnInterface


class PNN(CnnInterface, ABC):
    def __init__(self, channels, device="cpu", name="PNN"):
        super(PNN, self).__init__(device, name)
        self._model_name = name
        self.channels = channels
        self.conv1 = nn.Conv2d(in_channels=channels + 1, out_channels=64, kernel_size=(9, 9), padding='same', bias=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(5, 5), padding='same', bias=True)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=channels, kernel_size=(5, 5), padding='same', bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pan, ms):
        inputs = torch.cat([ms, pan], 1)
        rs = self.conv1(inputs)
        rs = self.relu(rs)
        rs = self.conv2(rs)
        rs = self.relu(rs)
        out = self.conv3(rs)
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
            pred = self.generate_output(pan, ms=ms)
            loss = self.loss_fn(pred, gt)

            # Backpropagation
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            loss = loss.item()
            torch.cuda.empty_cache()

            loss_batch += loss
        return loss_batch / len(dataloader)

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

                voutputs = self.generate_output(pan, ms=ms)
                vloss = self.loss_fn(voutputs, gt)
                running_vloss += vloss.item()

        avg_vloss = running_vloss / (i + 1)

        return avg_vloss

    def generate_output(self, pan, **kwargs):
        ms = kwargs['ms']
        return self(pan, ms)
