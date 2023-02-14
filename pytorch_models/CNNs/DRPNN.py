import torch
from torch import nn

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
    """ Implementation of the DRPNN network"""
    def __init__(self, channels, internal_channels=64, device="cpu", name="DRPNN"):
        """ Constructor of the class

        Parameters
        ----------
        channels : int
            number of channels accepted as input
        device : str, optional
            the device onto which train the network (either cpu or a cuda visible device).
            Default is 'cpu'
        name : str, optional
            the name of the network. Default is 'DRPNN'
        internal_channels : int
            number of internal channels. default is 64
        """
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
        """ Forwards the input data through the network

        Parameters
        ----------
        pan : tensor
            the panchromatic image
        ms : tensor
            the multi spectral image
        """
        input_data = torch.cat([ms, pan], 1)

        rs1 = self.conv2_pre(input_data)
        rs1 = self.relu(rs1)

        rs1 = self.backbone(rs1)

        rs1 = self.conv2_post(rs1)

        rs = torch.add(input_data, rs1)

        rs = self.conv2_final(rs)
        return rs

    def compile(self, loss_fn=None, optimizer=None):
        """ Define loss function and optimizer """
        self.loss_fn = loss_fn if loss_fn is not None else torch.nn.MSELoss(reduction='mean')
        self.opt = optimizer if optimizer is not None else torch.optim.Adam(self.parameters())
