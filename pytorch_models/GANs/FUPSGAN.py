import os
import shutil
from abc import ABC

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.nn import LeakyReLU
from constants import EPS
from pytorch_models.GANs.PSGAN import PSGAN


class FUPSGAN(PSGAN):
    def __init__(self, channels, device='cpu', name="FUPSGAN"):
        super().__init__(channels, device=device, name=name)

    # ------------------------------- Specific GAN methods -----------------------------
    class Generator(nn.Module):
        def __init__(self, channels, name="Gen"):
            super().__init__()
            self._model_name = name
            self.channels = channels

            # Bx1xHxW  ---> Bx32xHxW
            self.pan_enc_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding='same', bias=True)
            # Bx32xHxW  ---> Bx32xHxW
            self.pan_enc_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding='same', bias=True)
            # Bx32xHxW  ---> Bx64xH/2xW/2
            self.pan_enc_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2), stride=(2, 2),
                                       padding=(0, 0), bias=True)

            # BxCxH/4xW/4  ---> Bx32xH/4xW/4
            self.ms_enc_1 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=(3, 3), padding='same', bias=True)
            # Bx32xH/4xW/4 ---> Bx32xHxW
            self.ms_enc_2 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(4, 4), stride=(4, 4),
                                               padding=(0, 0), bias=True)
            # Bx32xHxW  ---> Bx64xH/2xW/2
            self.ms_enc_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2), stride=(2, 2),
                                      padding=(0, 0), bias=True)
            # Pan || Ms_lr
            # Bx128xH/2xW/2 ---> Bx128xH/2xW/2
            self.enc = nn.Sequential(
                LeakyReLU(negative_slope=.2),
                nn.Conv2d(in_channels=64 + 64, out_channels=128, kernel_size=(3, 3), padding='same', bias=True),
                LeakyReLU(negative_slope=.2),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding='same', bias=True)
            )

            # Bx128xH/2xW/2 ---> Bx128xH/2xW/2
            self.dec = nn.Sequential(
                LeakyReLU(negative_slope=.2),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=(2, 2),
                          bias=True),
                LeakyReLU(negative_slope=.2),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), padding='same', bias=True),
                LeakyReLU(negative_slope=.2),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding='same', bias=True),

                LeakyReLU(negative_slope=.2),
                nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(2, 2), stride=(2, 2),
                                   padding=(1, 1), bias=True)
            )

            # enc || dec
            # Bx256xH/2xW/2 ---> Bx128xHxW
            self.common = nn.Sequential(
                LeakyReLU(negative_slope=.2),
                nn.Conv2d(in_channels=128 + 128, out_channels=128, kernel_size=(3, 3), padding=(1, 1), bias=True),
                LeakyReLU(negative_slope=.2),
                nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(2, 2), stride=(2, 2),
                                   padding=(0, 0), bias=True)
            )

            # common || pan_enc_2 || ms_enc_2
            # Bx192xHxW ---> BxCxHxW
            self.final_part = nn.Sequential(
                LeakyReLU(negative_slope=.2),
                nn.Conv2d(in_channels=128 + 32 + 32, out_channels=64, kernel_size=(3, 3), padding='same',
                          bias=True),
                LeakyReLU(negative_slope=.2),
                nn.Conv2d(in_channels=64, out_channels=channels, kernel_size=(3, 3), padding='same',
                          bias=True)
            )

            self.relu = nn.ReLU(inplace=True)
            self.lrelu = LeakyReLU(negative_slope=.2)

        def forward(self, ms_lr, pan):
            pan1 = self.lrelu(self.pan_enc_1(pan))
            pan2 = self.pan_enc_2(pan1)
            pan3 = self.lrelu(pan2)
            pan3 = self.pan_enc_3(pan3)

            ms1 = self.lrelu(self.ms_enc_1(ms_lr))
            ms2 = self.ms_enc_2(ms1)
            ms3 = self.lrelu(ms2)
            ms3 = self.ms_enc_3(ms3)

            conc1 = torch.cat([pan3, ms3], 1)
            enc = self.enc(conc1)
            dec = self.dec(enc)
            conc2 = torch.cat([dec, enc], 1)
            common = self.common(conc2)
            conc3 = torch.cat([common, pan2, ms2], 1)

            out = self.final_part(conc3)
            out = self.relu(out)
            return out

    def loss_generator(self, **kwargs):
        pan = kwargs['pan']
        ms = kwargs['ms']
        ms_lr = kwargs['ms_lr']
        gt = kwargs['gt']
        outputs = self.generator(ms_lr, pan)
        predict_fake = self.discriminator(ms, outputs)
        # From Code
        # gen_loss_GAN = tf.reduce_mean(-tf.math.log(predict_fake + EPS))
        # gen_loss_L1 = tf.reduce_mean(tf.math.abs(gt - outputs))
        # gen_loss = gen_loss_GAN * self.alpha + gen_loss_L1 * self.beta

        # From Formula
        gen_loss_GAN = torch.mean(-torch.log(predict_fake + EPS))  # Inganna il discriminatore
        gen_loss_L1 = torch.mean(torch.abs(gt - outputs))  # Avvicina la risposta del generatore alla ground truth
        gen_loss = self.alpha * gen_loss_GAN + self.beta * gen_loss_L1

        return gen_loss

    def generate_output(self, pan, **kwargs):
        ms_lr = kwargs['ms_lr']
        return self.generator(ms_lr, pan)
