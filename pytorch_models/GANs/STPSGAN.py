import torch
from torch import nn

from pytorch_models.GANs.PSGAN import PSGAN


class STPSGAN(PSGAN):
    """ Stacked-PSGAN Implementation"""
    def __init__(self, channels, device='cpu', name="STPSGAN"):
        """ Constructor of the class"""
        super().__init__(channels, device, name)

    class Generator(nn.Module):
        """ Generator of STPSGAN """
        def __init__(self, channels, pad_mode, name="Gen"):
            super().__init__()
            self._model_name = name
            self.channels = channels

            # Pan || ms
            # BxC+1xHxW ---> Bx32xHxW
            self.main_stream = nn.Sequential(
                nn.Conv2d(in_channels=channels + 1, out_channels=32, kernel_size=(3, 3), padding='same',
                          padding_mode=pad_mode, bias=True),
                nn.LeakyReLU(negative_slope=.2),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding='same',
                          padding_mode=pad_mode, bias=True)
            )

            # Bx32xHxW ---> Bx128xH/2xW/2
            self.enc = nn.Sequential(
                nn.LeakyReLU(negative_slope=.2),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2), padding_mode=pad_mode, bias=True,
                          stride=(2, 2), padding=(0, 0)
                          ),
                nn.LeakyReLU(negative_slope=.2),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding='same',
                          padding_mode=pad_mode, bias=True),
                nn.LeakyReLU(negative_slope=.2),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding='same',
                          padding_mode=pad_mode, bias=True),
            )

            # Bx128xH/2xW/2 ---> Bx128xH/2xW/2
            self.dec = nn.Sequential(
                nn.LeakyReLU(negative_slope=.2),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3),
                          stride=(2, 2), padding=(1, 1),
                          padding_mode=pad_mode, bias=True),
                nn.LeakyReLU(negative_slope=.2),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), padding='same',
                          padding_mode=pad_mode, bias=True),
                nn.LeakyReLU(negative_slope=.2),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding='same',
                          padding_mode=pad_mode, bias=True),
                nn.LeakyReLU(negative_slope=.2),
                nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(2, 2), bias=True,
                                   stride=(2, 2), padding=(0, 0))
            )

            # enc || dec
            # Bx256xH/2xW/2 ---> Bx128xHxW
            self.common = nn.Sequential(
                nn.LeakyReLU(negative_slope=.2),
                nn.Conv2d(in_channels=128 + 128, out_channels=128, kernel_size=(3, 3), padding=(1, 1),
                          padding_mode=pad_mode, bias=True),
                nn.LeakyReLU(negative_slope=.2),
                nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(2, 2), bias=True,
                                   stride=(2, 2), padding=(0, 0))
            )

            # common || main
            # Bx256xHxW ---> BxCxHxW
            self.final_part = nn.Sequential(
                nn.LeakyReLU(negative_slope=.2),
                nn.Conv2d(in_channels=128 + 32, out_channels=64, kernel_size=(3, 3), padding='same',
                          padding_mode=pad_mode, bias=True),
                nn.LeakyReLU(negative_slope=.2),
                nn.Conv2d(in_channels=64, out_channels=channels, kernel_size=(3, 3), padding='same',
                          padding_mode=pad_mode, bias=True)
            )

            self.relu = nn.ReLU(inplace=True)

        def forward(self, pan, ms):
            conc1 = torch.cat([pan, ms], 1)
            main = self.main_stream(conc1)
            enc = self.enc(main)
            dec = self.dec(enc)
            conc2 = torch.cat([dec, enc], 1)
            common = self.common(conc2)
            conc3 = torch.cat([common, main], 1)

            out = self.final_part(conc3)
            out = self.relu(out)
            return out
