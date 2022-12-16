from abc import ABC

import torch
from torch import nn

from pytorch_models.CNNs.CnnInterface import CnnInterface


class FusionNet(CnnInterface, ABC):
    class ResidualBlock(nn.Module):
        def __init__(self, channels):
            super(FusionNet.ResidualBlock, self).__init__()
            self.backbone = nn.Sequential(
                nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 3), padding="same", bias=True),
                nn.ReLU(),
                nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 3), padding="same", bias=True),
                nn.ReLU()
            )
            self.relu = nn.ReLU()

        def forward(self, x):
            out = self.backbone(x)
            out = torch.add(out, x)
            out = self.relu(out)
            return out

    def __init__(self, channels, device="cpu", internal_channels=32, name="FusionNet"):
        super(FusionNet, self).__init__(device, name)
        self._model_name = name
        self.channels = channels
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=internal_channels, kernel_size=(3, 3), padding='same',
                               padding_mode='replicate', bias=True)
        backbone = [FusionNet.ResidualBlock(internal_channels) for _ in range(4)]
        self.backbone = nn.Sequential(*backbone)
        self.conv3 = nn.Conv2d(in_channels=internal_channels, out_channels=channels, kernel_size=(3, 3), padding='same',
                               padding_mode='replicate', bias=True)
        self.relu = nn.ReLU()

    def forward(self, pan, ms):
        pan_rep = pan.repeat(1, self.channels, 1, 1)
        rs = torch.sub(pan_rep, ms)
        rs = self.conv1(rs)
        rs = self.relu(rs)
        rs = self.backbone(rs)
        rs = self.conv3(rs)

        out = torch.add(rs, ms)
        return out

    def generate_output(self, pan, evaluation=True, **kwargs):
        ms = kwargs['ms']
        if evaluation:
            self.eval()
            with torch.no_grad():
                return self(pan, ms)
        return self(pan, ms)

    def compile(self, loss_fn=None, optimizer=None):
        self.loss_fn = loss_fn if loss_fn is not None else torch.nn.MSELoss(reduction='mean')
        self.opt = optimizer if optimizer is not None else torch.optim.Adam(self.parameters())
