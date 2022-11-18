import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from pytorch_models.CNNs.CnnInterface import CnnInterface


class ConvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv20 = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=(7, 7), stride=(1, 1), padding=3,
                                bias=True)

        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        rs1 = self.conv20(x)
        rs = self.relu1(rs1)
        return rs


class DRPNN(CnnInterface):
    def __init__(self, channels, internal_channels=64, device="cpu", name="DRPNN"):
        super(DRPNN, self).__init__(device, name)
        self.conv2_pre = nn.Conv2d(in_channels=channels + 1, out_channels=internal_channels,
                                   kernel_size=(7, 7), stride=(1, 1), padding=3,
                                   bias=True)
        self.relu = nn.ReLU(inplace=True)

        self.backbone = nn.Sequential(
            ConvBlock(internal_channels),
            ConvBlock(internal_channels),
            ConvBlock(internal_channels),
            ConvBlock(internal_channels),
            ConvBlock(internal_channels),
            ConvBlock(internal_channels),
            ConvBlock(internal_channels),
            ConvBlock(internal_channels),
        )

        self.conv2_post = nn.Conv2d(in_channels=internal_channels, out_channels=channels + 1,
                                    kernel_size=(7, 7), stride=(1, 1), padding=3,
                                    bias=True)

        self.conv2_final = nn.Conv2d(in_channels=channels + 1, out_channels=channels,
                                     kernel_size=(7, 7), stride=(1, 1), padding=3,
                                     bias=True)

    def forward(self, pan, ms):

        input_data = torch.cat([ms, pan], 1)

        rs1 = self.conv2_pre(input_data)
        rs1 = self.relu(rs1)

        rs1 = self.backbone(rs1)

        rs1 = self.conv2_post(rs1)

        rs = torch.add(input_data, rs1)

        rs = self.conv2_final(rs)
        return rs

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

                voutputs = self.generate_output(pan, ms=ms)
                vloss = self.loss_fn(voutputs, gt)
                running_vloss += vloss.item()

        avg_vloss = running_vloss / (i + 1)

        return avg_vloss

    def generate_output(self, pan, **kwargs):
        ms = kwargs['ms']
        return self(pan, ms)
