from abc import ABC

import numpy as np
import torch
from torch import nn, optim
from torch.nn import LeakyReLU

from constants import EPS
from pytorch_models.GANs.GanInterface import GanInterface
from pytorch_models.GANs.PanGan import PanGan


class PSGAN(GanInterface, ABC):
    def __init__(self, channels, device='cpu', name="PSGAN", pad_mode="replicate"):
        super().__init__(device, name)
        self.channels = channels
        self.alpha = 1
        self.beta = 100
        self.generator = self.Generator(channels, pad_mode)
        self.discriminator = self.Discriminator(channels, pad_mode)
        self.best_losses = [np.inf, np.inf]
        self.gen_opt = optim.Adam(self.generator.parameters())
        self.disc_opt = optim.Adam(self.discriminator.parameters())

    # ------------------------------- Specific GAN methods -----------------------------
    class Generator(nn.Module):
        def __init__(self, channels, pad_mode="replicate", name="Gen"):
            super().__init__()
            self._model_name = name
            self.channels = channels

            # Bx1xHxW  ---> Bx32xHxW
            self.pan_enc_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding='same',
                                       padding_mode=pad_mode, bias=True)
            # Bx32xHxW  ---> Bx32xHxW
            self.pan_enc_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding='same',
                                       padding_mode=pad_mode, bias=True)
            # Bx32xHxW  ---> Bx64xH/2xW/2
            self.pan_enc_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2),
                                       stride=(2, 2), padding=(0, 0),
                                       padding_mode=pad_mode, bias=True)

            # BxCxHxW  ---> Bx32xHxW
            self.ms_enc_1 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=(3, 3), padding='same',
                                      padding_mode=pad_mode, bias=True)
            # Bx32xHxW  ---> Bx32xHxW
            self.ms_enc_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding='same',
                                      padding_mode=pad_mode, bias=True)
            # Bx32xHxW  ---> Bx64xH/2xW/2
            self.ms_enc_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2),
                                      stride=(2, 2), padding=(0, 0),
                                      padding_mode=pad_mode, bias=True)
            # Pan || Ms
            # Bx128xH/2xW/2 ---> Bx128xH/2xW/2
            self.enc = nn.Sequential(
                LeakyReLU(negative_slope=.2),
                nn.Conv2d(in_channels=64 + 64, out_channels=128, kernel_size=(3, 3), padding='same',
                          padding_mode=pad_mode, bias=True),
                LeakyReLU(negative_slope=.2),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding='same',
                          padding_mode=pad_mode, bias=True)
            )

            # Bx128xH/2xW/2 ---> Bx128xH/2xW/2
            self.dec = nn.Sequential(
                LeakyReLU(negative_slope=.2),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3),
                          stride=(2, 2), padding=(2, 2),
                          padding_mode=pad_mode, bias=True),
                LeakyReLU(negative_slope=.2),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), padding='same',
                          padding_mode=pad_mode, bias=True),
                LeakyReLU(negative_slope=.2),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding='same',
                          padding_mode=pad_mode, bias=True),

                LeakyReLU(negative_slope=.2),
                nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(2, 2), bias=True,
                                   stride=(2, 2), padding=(1, 1))
            )

            # enc || dec
            # Bx256xH/2xW/2 ---> Bx128xHxW
            self.common = nn.Sequential(
                LeakyReLU(negative_slope=.2),
                nn.Conv2d(in_channels=128 + 128, out_channels=128, kernel_size=(3, 3), padding=(1, 1),
                          padding_mode=pad_mode, bias=True),
                LeakyReLU(negative_slope=.2),
                nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(2, 2), bias=True,
                                   stride=(2, 2), padding=(0, 0))
            )
            # common || pan_enc_2 || ms_enc_2
            # Bx192xHxW ---> BxCxHxW
            self.final_part = nn.Sequential(
                LeakyReLU(negative_slope=.2),
                nn.Conv2d(in_channels=128 + 32 + 32, out_channels=64, kernel_size=(3, 3), padding='same',
                          padding_mode=pad_mode, bias=True),
                LeakyReLU(negative_slope=.2),
                nn.Conv2d(in_channels=64, out_channels=channels, kernel_size=(3, 3), padding='same',
                          padding_mode=pad_mode, bias=True)
            )

            self.relu = nn.ReLU(inplace=True)
            self.lrelu = LeakyReLU(negative_slope=.2)

        def forward(self, pan, ms):
            pan1 = self.lrelu(self.pan_enc_1(pan))
            pan2 = self.pan_enc_2(pan1)
            pan3 = self.lrelu(pan2)
            pan3 = self.pan_enc_3(pan3)

            ms1 = self.lrelu(self.ms_enc_1(ms))
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

    class Discriminator(nn.Module):
        def __init__(self, channels, pad_mode="replicate", name="Disc"):
            super().__init__()
            self._model_name = name
            self.channels = channels

            self.backbone = nn.Sequential(
                nn.Conv2d(in_channels=channels * 2, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=(2, 2),
                          padding_mode=pad_mode, bias=True),
                LeakyReLU(negative_slope=.2),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(2, 2),
                          padding_mode=pad_mode, bias=True),
                LeakyReLU(negative_slope=.2),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=(2, 2),
                          padding_mode=pad_mode, bias=True),
                LeakyReLU(negative_slope=.2),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2),
                          padding_mode=pad_mode, bias=True),
                LeakyReLU(negative_slope=.2),
            )

            self.out_conv = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=(3, 3), stride=(1, 1),
                                      padding='same', padding_mode=pad_mode, bias=True)
            self.sigmoid = nn.Sigmoid()

        def forward(self, generated, target):
            inputs = torch.cat([generated, target], 1)  # B x 2*C x H x W
            out = self.backbone(inputs)
            out = self.out_conv(out)
            out = self.sigmoid(out)
            return out

    def loss_generator(self, **kwargs):
        ms = kwargs['ms']
        pan = kwargs['pan']
        gt = kwargs['gt']
        outputs = self.generator(pan, ms)
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

    def loss_discriminator(self, ms, gt, output):

        predict_fake = self.discriminator(ms, output)
        predict_real = self.discriminator(ms, gt)

        # From Formula
        # mean[ 1 - log(fake) + log(real) ]
        # return torch.mean(
        #     1 - torch.log(predict_fake + EPS) + torch.log(predict_real + EPS)
        # )

        # From Code
        # return tf.reduce_mean(
        #     -(
        #             tf.math.log(predict_real + EPS) + tf.math.log(1 - predict_fake + EPS)
        #     )
        # )
        return torch.mean(
            -(
                    torch.log(predict_real + EPS) + torch.log(1 - predict_fake + EPS)
            )
        )

    # -------------------------------- Interface Methods ------------------------------
    def train_step(self, dataloader):
        self.train(True)

        loss_g_batch = 0
        loss_d_batch = 0
        for batch, data in enumerate(dataloader):
            pan, ms, ms_lr, gt = data

            gt = gt.to(self.device)
            pan = pan.to(self.device)
            ms = ms.to(self.device)
            ms_lr = ms_lr.to(self.device)
            if len(pan.shape) != len(ms.shape):
                pan = torch.unsqueeze(pan, 0)

            # Generate Data for Discriminators Training
            with torch.no_grad():
                generated_HRMS = self.generate_output(pan, ms=ms, ms_lr=ms_lr)

            # ------------------- Training Discriminator ----------------------------
            self.discriminator.train(True)
            self.generator.train(False)
            self.discriminator.zero_grad()

            # Compute prediction and loss
            loss_d = self.loss_discriminator(ms, gt, generated_HRMS)

            # Backpropagation
            self.disc_opt.zero_grad()
            loss_d.backward()
            self.disc_opt.step()

            loss = loss_d.item()
            torch.cuda.empty_cache()

            loss_d_batch += loss

            # ------------------- Training Generator ----------------------------
            self.discriminator.train(False)
            self.generator.train(True)
            self.generator.zero_grad()

            # Compute prediction and loss
            loss_g = self.loss_generator(ms=ms, pan=pan, gt=gt, ms_lr=ms_lr)

            # Backpropagation
            self.gen_opt.zero_grad()
            loss_g.backward()
            self.gen_opt.step()

            loss = loss_g.item()
            torch.cuda.empty_cache()

            loss_g_batch += loss

        return {"Gen loss": loss_g_batch / len(dataloader),
                "Disc loss": loss_d_batch / len(dataloader)
                }

    def validation_step(self, dataloader):
        self.eval()
        self.discriminator.eval()
        self.generator.eval()

        gen_loss = 0.0
        disc_loss = 0.0
        i = 0
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                pan, ms, ms_lr, gt = data

                gt = gt.to(self.device)
                pan = pan.to(self.device)
                ms = ms.to(self.device)
                ms_lr = ms_lr.to(self.device)
                if len(pan.shape) == 3:
                    pan = torch.unsqueeze(pan, 0)

                generated = self.generate_output(pan, ms=ms, ms_lr=ms_lr)

                dloss = self.loss_discriminator(ms, gt, generated)
                disc_loss += dloss.item()

                gloss = self.loss_generator(ms=ms, pan=pan, gt=gt, ms_lr=ms_lr)
                gen_loss += gloss.item()

        return {"Gen loss": gen_loss / len(dataloader),
                "Disc loss": disc_loss / len(dataloader)
                }

    def save_model(self, path):
        torch.save({
            'gen_state_dict': self.generator.state_dict(),
            'disc_state_dict': self.discriminator.state_dict(),
            'gen_optimizer_state_dict': self.gen_opt.state_dict(),
            'disc_optimizer_state_dict': self.disc_opt.state_dict(),
            'gen_best_loss': self.best_losses[0],
            'disc_best_loss': self.best_losses[1],
            'best_epoch': self.best_epoch,
            'tot_epochs': self.tot_epochs
        }, f"{path}")

    def load_model(self, path):
        trained_model = torch.load(f"{path}", map_location=torch.device(self.device))
        self.generator.load_state_dict(trained_model['gen_state_dict'])
        self.discriminator.load_state_dict(trained_model['disc_state_dict'])
        self.gen_opt.load_state_dict(trained_model['gen_optimizer_state_dict'])
        self.disc_opt.load_state_dict(trained_model['disc_optimizer_state_dict'])
        self.best_losses = [trained_model['gen_best_loss'], trained_model['disc_best_loss']]
        self.best_epoch = trained_model['best_epoch']
        self.tot_epochs = trained_model['tot_epochs']

    def generate_output(self, pan, **kwargs):
        ms = kwargs['ms']
        return self.generator(pan, ms)

    def set_optimizers_lr(self, lr):
        for g in self.gen_opt.param_groups:
            g['lr'] = lr
        for g in self.disc_opt.param_groups:
            g['lr'] = lr


if __name__ == '__main__':
    print("ff")
