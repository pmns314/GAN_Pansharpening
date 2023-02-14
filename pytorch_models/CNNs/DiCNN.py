from abc import ABC

import torch
from torch import nn

from pytorch_models.CNNs.CnnInterface import CnnInterface
from pytorch_models.Losses import FrobeniusLoss


class DiCNN(CnnInterface, ABC):
    """ Implementation of the DiCNN network"""
    def __init__(self, channels, device="cpu", internal_channels=64, name="DiCNN"):
        """ Constructor of the class

        Parameters
        ----------
        channels : int
            number of channels accepted as input
        device : str, optional
            the device onto which train the network (either cpu or a cuda visible device).
            Default is 'cpu'
        name : str, optional
            the name of the network. Default is 'DiCNN'
        """
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
        """ Forwards the input data through the network

        Parameters
        ----------
        pan : tensor
            the panchromatic image
        ms : tensor
            the multi spectral image
        """
        inputs = torch.cat([ms, pan], 1)
        output = self.backbone(inputs)
        output = torch.add(output, ms)
        return output

    def compile(self, loss_fn=None, optimizer=None):
        """ Define loss function and optimizer """
        self.loss_fn = loss_fn if loss_fn is not None else FrobeniusLoss()
        self.opt = optimizer if optimizer is not None else torch.optim.Adam(self.parameters())
