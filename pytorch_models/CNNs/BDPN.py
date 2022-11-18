from abc import ABC

import torch
from torch import nn

from pytorch_models.CNNs.CnnInterface import CnnInterface


def charbonnier_loss(y_true, y_pred):
    epsilon = 1e-6
    x = y_true - y_pred

    # loss = sqrt(x**2 + eps**2)
    loss = torch.sqrt(torch.square(x) + epsilon)
    # Mean over batch
    loss = torch.mean(torch.mean(loss, [1, 2, 3]))
    return loss


class BDPN(CnnInterface, ABC):
    class ResBlock(nn.Module):
        def __init__(self, channel):
            super(BDPN.ResBlock, self).__init__()

            self.conv20 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1,
                                    bias=True)
            self.conv21 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1,
                                    bias=True)
            self.prelu = nn.PReLU(num_parameters=1, init=0.2)

        def forward(self, x):
            rs1 = self.prelu(self.conv20(x))  # Bsx32x64x64
            rs1 = self.conv21(rs1)  # Bsx32x64x64
            rs = torch.add(x, rs1)  # Bsx32x64x64

            return rs

    def __init__(self, channels, device="cpu", internal_channels=64, name="BDPN"):
        super(BDPN, self).__init__(device, name)
        self._model_name = name
        self.channels = channels
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=internal_channels, kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=True)
        backbone1 = []
        for _ in range(10):
            backbone1.append(self.ResBlock(internal_channels))
        self.backbone1 = nn.Sequential(*backbone1)

        backbone2 = []
        for _ in range(10):
            backbone2.append(self.ResBlock(internal_channels))
        self.backbone2 = nn.Sequential(*backbone2)

        self.conv3 = nn.Conv2d(in_channels=internal_channels, out_channels=channels, kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=True)
        self.conv4 = nn.Conv2d(in_channels=channels, out_channels=channels * 4, kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=True)
        self.conv5 = nn.Conv2d(in_channels=channels, out_channels=channels * 4, kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=True)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pixshuf = nn.PixelShuffle(upscale_factor=2)
        self.prelu = nn.PReLU(num_parameters=1, init=0.2)

    def forward(self, pan, ms):
        # ========A): pan feature (extraction)===========
        # --------pan feature (stage 1:)------------
        pan_feature = self.conv1(pan)  # Nx64x64x64
        rs = pan_feature  # Nx64x64x64

        rs = self.backbone1(rs)  # Nx64x64x64

        pan_feature1 = torch.add(pan_feature, rs)  # Bsx64x64x64
        pan_feature_level1 = self.conv3(pan_feature1)  # Bsx8x64x64
        pan_feature1_out = self.maxpool(pan_feature1)  # Bsx64x32x32

        # --------pan feature (stage 2:)------------
        rs = pan_feature1_out  # Bsx64x32x32

        rs = self.backbone2(rs)  # Nx64x32x32, ????

        pan_feature2 = torch.add(pan_feature1_out, rs)  # Bsx64x32x32
        pan_feature_level2 = self.conv3(pan_feature2)  # Bsx8x32x32

        # ========B): ms feature (extraction)===========
        # --------ms feature (stage 1:)------------
        ms_feature1 = self.conv4(ms)  # x= ms(Nx8x16x16); ms_feature1 =Nx32x16x16
        ms_feature_up1 = self.pixshuf(ms_feature1)  # Nx8x32x32
        ms_feature_level1 = torch.add(pan_feature_level2, ms_feature_up1)  # Nx8x32x32

        # --------ms feature (stage 2:)------------
        ms_feature2 = self.conv5(ms_feature_level1)  # Nx32x32x32
        ms_feature_up2 = self.pixshuf(ms_feature2)  # Nx8x64x64
        output = torch.add(pan_feature_level1, ms_feature_up2)  # Nx8x64x64

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
            ms_lr = ms_lr.to(self.device)

            # Compute prediction and loss
            pred = self.generate_output(pan, ms_lr=ms_lr)
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
                ms_lr = ms_lr.to(self.device)

                # Compute prediction and loss
                voutputs = self.generate_output(pan, ms_lr=ms_lr)
                vloss = self.loss_fn(voutputs, gt)
                running_vloss += vloss.item()

        avg_vloss = running_vloss / (i + 1)

        return avg_vloss

    def generate_output(self, pan, **kwargs):
        ms_lr = kwargs['ms_lr']
        return self(pan, ms_lr)
