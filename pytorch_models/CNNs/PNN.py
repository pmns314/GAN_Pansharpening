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

    def generate_output(self, pan, evaluation=True, **kwargs):
        ms = kwargs['ms']
        if evaluation:
            self.eval()
            with torch.no_grad():
                return self(pan, ms)
        return self(pan, ms)

    def compile(self, loss_fn=None, optimizer=None):
        self.loss_fn = loss_fn if loss_fn is not None else torch.nn.L1Loss(reduction='mean')
        self.opt = optimizer if optimizer is not None else torch.optim.Adam(self.parameters())
