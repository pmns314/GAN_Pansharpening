from abc import ABC

import torch
from torch import nn

from pytorch_models.CNNs.CnnInterface import CnnInterface


def frobenius_loss(y_true, y_pred):
    tensor = y_pred - y_true
    norm = torch.norm(tensor, p="fro")
    return torch.mean(torch.square(norm))


class DiCNN(CnnInterface, ABC):

    def __init__(self, channels, device="cpu", internal_channels=64, name="DiCNN"):
        super(DiCNN, self).__init__(device, name)
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=channels + 1, out_channels=internal_channels, kernel_size=3, stride=1, padding='same',
                      bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=internal_channels, out_channels=internal_channels, kernel_size=3, stride=1,
                      padding='same',
                      bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=internal_channels, out_channels=channels, kernel_size=3, stride=1, padding='same',
                      bias=True),
            nn.ReLU()
        )

    def forward(self, pan, ms):
        inputs = torch.cat([ms, pan], 1)
        output = self.backbone(inputs)
        output = torch.add(output, ms)
        return output

    def generate_output(self, pan, **kwargs):
        ms = kwargs['ms']
        return self(pan, ms)

    def compile(self, loss_fn=None, optimizer=None):
        self.loss_fn = loss_fn if loss_fn is not None else frobenius_loss
        self.opt = optimizer if optimizer is not None else torch.optim.Adam(self.parameters())
