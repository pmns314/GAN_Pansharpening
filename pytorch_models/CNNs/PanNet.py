import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


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


class PanNet(nn.Module):
    def __init__(self, spectral_num, criterion, channel=32, reg=True):
        super(PanNet, self).__init__()
        self.criterion = criterion
        self.reg = reg

        # ConvTranspose2d: output = (input - 1)*stride + outpading - 2*padding + kernelsize
        self.deconv = nn.ConvTranspose2d(in_channels=spectral_num, out_channels=spectral_num, kernel_size=8, stride=4,
                                         padding=2, bias=True)
        self.conv1 = nn.Conv2d(in_channels=spectral_num + 1, out_channels=channel, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.res1 = Resblock()
        self.res2 = Resblock()
        self.res3 = Resblock()
        self.res4 = Resblock()

        self.conv3 = nn.Conv2d(in_channels=channel, out_channels=spectral_num, kernel_size=3, stride=1, padding=1,
                               bias=True)

        self.relu = nn.ReLU(inplace=True)

        self.backbone = nn.Sequential(  # method 2: 4 resnet repeated blocks
            self.res1,
            self.res2,
            self.res3,
            self.res4
        )

        self.apply(init_weights)
        # init_weights(self.backbone, self.deconv, self.conv1, self.conv3)  # state initialization, important!

    def forward(self, x, y):  # x= hp of ms; y = hp of pan

        output_deconv = self.deconv(x)
        input = torch.cat([output_deconv, y], 1)  # Bsx9x64x64
        rs = self.relu(self.conv1(input))  # Bsx32x64x64

        rs = self.backbone(rs)  # ResNet's backbone!

        output = self.conv3(rs)  # Bsx8x64x64
        return output

    def train_step(self, data, *args, **kwargs):
        log_vars = {}
        gt, lms, ms_hp, pan_hp = data['gt'].cuda(), data['lms'].cuda(), \
                                 data['ms_hp'].cuda(), data['pan_hp'].cuda()
        hp_sr = self(ms_hp, pan_hp)
        sr = lms + hp_sr  # output:= lms + hp_sr
        loss = self.criterion(sr, gt, *args, **kwargs)
        # return sr, loss
        log_vars.update(loss=loss['loss'])
        return {'loss': loss['loss'], 'log_vars': log_vars}

    def val_step(self, data, *args, **kwargs):
        # gt, lms, ms, pan = data
        gt, lms, ms_hp, pan_hp = data['gt'].cuda(), data['lms'].cuda(), \
                                 data['ms'].cuda(), data['pan'].cuda()
        hp_sr = self(ms_hp, pan_hp)
        sr = lms + hp_sr  # output:= lms + hp_sr
        return sr, gt


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    model = Resblock().to(device)
    print(model)

    from torchsummary import summary
    summary(model,(64,64,9))
