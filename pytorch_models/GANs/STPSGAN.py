import torch
from torch import nn

from pytorch_models.GANs.PSGAN import PSGAN
from utils import calc_padding_conv2dtranspose, calc_padding_conv2d

if __name__ == '__main__':
    print(calc_padding_conv2dtranspose(8, 2, 2, 16))


class STPSGAN(PSGAN):

    def __init__(self, channels, device='cpu', name="STPSGAN"):
        super().__init__(channels, device, name)

    class Generator(nn.Module):
        def __init__(self, channels, name="Gen"):
            super().__init__()
            self._model_name = name
            self.channels = channels

            self.main_stream = nn.Sequential(
                nn.Conv2d(in_channels=channels + 1, out_channels=32, kernel_size=(3, 3), padding='same', bias=True),
                nn.LeakyReLU(negative_slope=.2),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding='same', bias=True)
            )

            self.enc = nn.Sequential(
                nn.LeakyReLU(negative_slope=.2),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2), stride=(2, 2),
                          padding=(0, 0), bias=True),
                nn.LeakyReLU(negative_slope=.2),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding='same', bias=True),
                nn.LeakyReLU(negative_slope=.2),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding='same', bias=True),
            )

            self.dec = nn.Sequential(
                nn.LeakyReLU(negative_slope=.2),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(2, 2),
                          padding=(1, 1), bias=True),
                nn.LeakyReLU(negative_slope=.2),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), padding='same', bias=True),
                nn.LeakyReLU(negative_slope=.2),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding='same', bias=True),
                nn.LeakyReLU(negative_slope=.2),
                nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(2, 2), stride=(2, 2),
                                   padding=(0, 0), bias=True)
            )

            self.common = nn.Sequential(
                nn.LeakyReLU(negative_slope=.2),
                nn.Conv2d(in_channels=128+128, out_channels=128, kernel_size=(3, 3), padding=(1, 1), bias=True),
                nn.LeakyReLU(negative_slope=.2),
                nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(2, 2), stride=(2, 2),
                                   padding=(0, 0), bias=True)
            )

            self.final_part = nn.Sequential(
                nn.LeakyReLU(negative_slope=.2),
                nn.Conv2d(in_channels=128+32, out_channels=64, kernel_size=(3, 3), padding='same',
                          bias=True),
                nn.LeakyReLU(negative_slope=.2),
                nn.Conv2d(in_channels=64, out_channels=channels, kernel_size=(3, 3), padding='same',
                          bias=True)
            )

            self.relu = nn.ReLU(inplace=True)

        def forward(self, ms, pan):
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
