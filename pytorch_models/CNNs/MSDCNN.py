from abc import ABC

import torch
from torch import nn

from pytorch_models.CNNs.CnnInterface import CnnInterface


class MSDCNN(CnnInterface, ABC):
    """ Implementation of the MSDCNN network"""
    class MSResB(nn.Module):
        def __init__(self, in_channels, out_channels):
            super(MSDCNN.MSResB, self).__init__()

            assert in_channels / 3 == out_channels
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding="same",
                          bias=True),
                nn.ReLU()
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(5, 5), padding="same",
                          bias=True),
                nn.ReLU()
            )
            self.conv3 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(7, 7), padding="same",
                          bias=True),
                nn.ReLU()
            )
            self.relu = nn.ReLU()

        def forward(self, inputs):
            x = self.conv1(inputs)
            y = self.conv2(inputs)
            z = self.conv3(inputs)

            concat = torch.cat([x, y, z], 1)
            out = torch.add(inputs, concat)
            out = self.relu(out)
            return out

    def __init__(self, channels, device="cpu", name="MSDCNN"):
        """ Constructor of the class

        Parameters
        ----------
        channels : int
            number of channels accepted as input
        device : str, optional
            the device onto which train the network (either cpu or a cuda visible device).
            Default is 'cpu'
        name : str, optional
            the name of the network. Default is 'MSDCNN'
        """
        super(MSDCNN, self).__init__(device, name)
        self._model_name = name
        self.channels = channels

        self.deep_features = nn.Sequential(
            nn.Conv2d(in_channels=channels + 1, out_channels=60, kernel_size=(7, 7), padding='same', bias=True),
            nn.ReLU(),
            MSDCNN.MSResB(60, 20),
            nn.Conv2d(in_channels=60, out_channels=30, kernel_size=(3, 3), padding='same', bias=True),
            nn.ReLU(),
            MSDCNN.MSResB(30, 10),
            nn.Conv2d(in_channels=30, out_channels=channels, kernel_size=(5, 5), padding='same', bias=True)
        )

        self.shallow_features = nn.Sequential(
            nn.Conv2d(in_channels=channels + 1, out_channels=64, kernel_size=(9, 9), padding='same', bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1), padding='same', bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=channels, kernel_size=(5, 5), padding='same', bias=True)
        )

        self.relu = nn.ReLU()

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
        deep_f = self.deep_features(inputs)
        shallow_f = self.shallow_features(inputs)
        out = torch.add(deep_f, shallow_f)
        out = self.relu(out)
        return out

    def compile(self, loss_fn=None, optimizer=None):
        """ Define loss function and optimizer """
        self.loss_fn = loss_fn if loss_fn is not None else torch.nn.MSELoss(reduction='mean')
        self.opt = optimizer if optimizer is not None else torch.optim.Adam(self.parameters())
