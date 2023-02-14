import torch
from torch import nn

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
    """ Implementation of the PanNet network"""
    def __init__(self, channels, internal_channels=32, device="cpu", name="PanNet"):
        """ Constructor of the class

        Parameters
        ----------
        channels : int
            number of channels accepted as input
        device : str, optional
            the device onto which train the network (either cpu or a cuda visible device).
            Default is 'cpu'
        name : str, optional
            the name of the network. Default is 'PanNet'
        internal_channels : int
            number of internal channels. default is 32
        """
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
        """ Forwards the input data through the network

        Parameters
        ----------
        pan : tensor
            the panchromatic image
        ms : tensor
            the multi spectral image
        """
        output_deconv = self.deconv(ms)
        input = torch.cat([output_deconv, pan], 1)  # Bsx9x64x64
        rs = self.relu(self.conv1(input))  # Bsx32x64x64

        rs = self.backbone(rs)  # ResNet's backbone!

        output = self.conv3(rs)  # Bsx8x64x64
        return output

    def compile(self, loss_fn=None, optimizer=None):
        """ Define loss function and optimizer """
        self.loss_fn = loss_fn if loss_fn is not None else torch.nn.MSELoss(reduction='mean')
        self.opt = optimizer if optimizer is not None else torch.optim.Adam(self.parameters())
