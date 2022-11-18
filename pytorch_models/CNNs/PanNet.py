import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from pytorch_models.CNNs.CnnInterface import CnnInterface


class Resblock(nn.Module):
    def __init__(self):
        super().__init__()

        channel = 64
        self.conv20 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1,
                                bias=True)
        self.conv21 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1,
                                bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        rs1 = self.conv20(x)
        rs1 = self.relu1(rs1)  # Bsx32x64x64
        rs1 = self.conv21(rs1)  # Bsx32x64x64
        rs = torch.add(x, rs1)  # Bsx32x64x64
        rs = self.relu2(rs)
        return rs


class PanNet(CnnInterface):
    def __init__(self, channels, internal_channels=32, device="cpu", name="PanNet"):
        super(PanNet, self).__init__(device, name)

        # ConvTranspose2d: output = (input - 1)*stride + outpading - 2*padding + kernelsize
        self.deconv = nn.ConvTranspose2d(in_channels=channels, out_channels=channels, kernel_size=8, stride=4,
                                         padding=2, bias=True)
        self.conv1 = nn.Conv2d(in_channels=channels + 1, out_channels=internal_channels, kernel_size=3, stride=1,
                               padding=1,
                               bias=True)
        self.res1 = Resblock()
        self.res2 = Resblock()
        self.res3 = Resblock()
        self.res4 = Resblock()

        self.conv3 = nn.Conv2d(in_channels=internal_channels, out_channels=channels, kernel_size=3, stride=1, padding=1,
                               bias=True)

        self.relu = nn.ReLU(inplace=True)

        self.backbone = nn.Sequential(  # method 2: 4 resnet repeated blocks
            self.res1,
            self.res2,
            self.res3,
            self.res4
        )

    def forward(self, pan, ms):

        output_deconv = self.deconv(ms)
        input = torch.cat([output_deconv, pan], 1)  # Bsx9x64x64
        rs = self.relu(self.conv1(input))  # Bsx32x64x64

        rs = self.backbone(rs)  # ResNet's backbone!

        output = self.conv3(rs)  # Bsx8x64x64
        return output

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
