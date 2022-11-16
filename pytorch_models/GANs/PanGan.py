from abc import ABC

import torch
from numpy import log2
from torch import nn, optim

from constants import EPS
from pytorch_models.GANs.GanInterface import GanInterface
from torch.nn.functional import interpolate
import numpy as np
import matplotlib.pyplot as plt


def downsample(img, new_shape):
    return interpolate(img, new_shape, mode='bilinear', antialias=True)


def high_pass(img, device='cpu'):
    blur_kerel = np.zeros(shape=(1, img.shape[1], 3, 3), dtype=np.float32)
    value = np.array([[1., 1., 1.], [1., -8., 1.], [1., 1., 1.]])
    blur_kerel[:, :, :, :] = value
    img_hp = torch.nn.functional.conv2d(img, torch.from_numpy(blur_kerel).to(device), stride=(1, 1), padding='same')
    return img_hp


class PanGan(GanInterface, ABC):
    def __init__(self, channels, device="cpu", name="PanGan", train_spat_disc=False, use_highpass=True):
        super(PanGan, self).__init__(name=name, device=device)

        self.generator = PanGan.Generator(channels)
        self.spatial_discriminator = PanGan.Discriminator(1)
        self.spectral_discriminator = PanGan.Discriminator(channels)

        # Discriminator
        self.a = 0  # Label for Original
        self.b = 1  # Label for Fake

        # Generator
        self.c = 1  # Label For Fake
        self.d = 1

        self.alpha = .002
        self.beta = .001
        self.mu = 5

        self.best_losses = [np.inf, np.inf, np.inf]
        self.mse = torch.nn.MSELoss(reduction='mean')
        self.use_spatial = train_spat_disc
        self.use_highpass = use_highpass

        self.optimizer_gen = optim.Adam(self.generator.parameters(), lr=.001)
        self.optimizer_spatial_disc = optim.Adam(self.spatial_discriminator.parameters(), lr=.001)
        self.optimizer_spectral_disc = optim.Adam(self.spectral_discriminator.parameters(), lr=.001)

    # ------------------------- Specific GAN Methods -----------------------------------
    class Generator(nn.Module):
        def __init__(self, in_channels):
            super(PanGan.Generator, self).__init__()

            # BxC+1xHxW ---> Bx64xHxW
            self.block_1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels + 1, out_channels=64, kernel_size=(9, 9), padding="same",
                          stride=(1, 1), bias=True, padding_mode="replicate"),
                nn.BatchNorm2d(64, eps=1e-5, momentum=.9),
                nn.ReLU()
            )

            # Bx64+C+1xHxW ---> Bx32xHxW
            self.block_2 = nn.Sequential(
                nn.Conv2d(in_channels=64 + in_channels + 1, out_channels=32, kernel_size=(5, 5), padding="same",
                          stride=(1, 1), bias=True, padding_mode="replicate"),
                nn.BatchNorm2d(32, eps=1e-5, momentum=.9),
                nn.ReLU()
            )

            # Bx32+64+C+1  ---> BxCxHxW
            self.block_3 = nn.Sequential(
                nn.Conv2d(in_channels=32 + 64 + in_channels + 1, out_channels=in_channels, kernel_size=(5, 5),
                          padding="same", stride=(1, 1), bias=True, padding_mode="replicate"),
                nn.Tanh()

            )

        def forward(self, pan, ms):
            input_block_1 = torch.cat([ms, pan], 1)
            output_block_1 = self.block_1(input_block_1)

            input_block_2 = torch.cat([input_block_1, output_block_1], 1)
            output_block_2 = self.block_2(input_block_2)

            input_block_3 = torch.cat([input_block_1, output_block_1, output_block_2], 1)
            output_block_3 = self.block_3(input_block_3)

            return output_block_3

    class Discriminator(nn.Module):
        class ConvBlock(nn.Module):
            def __init__(self, in_channels, out_channels, stride=2, batch_normalization=True):
                super(PanGan.Discriminator.ConvBlock, self).__init__()
                self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3),
                                       stride=(stride, stride), padding=(1, 1), bias=True, padding_mode="replicate")
                nn.init.trunc_normal_(self.conv2.weight, std=1e-3)
                nn.init.constant_(self.conv2.bias, 0.0)
                self.lrelu = nn.LeakyReLU(negative_slope=.2)
                self.bn = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=.9)
                self.batch_normalization = batch_normalization

            def forward(self, input_data):
                rs = self.conv2(input_data)
                if self.batch_normalization:
                    rs = self.bn(rs)
                rs = self.lrelu(rs)
                return rs

        def __init__(self, channels):
            super(PanGan.Discriminator, self).__init__()
            self.strides2_stack = nn.Sequential(
                self.ConvBlock(channels, 16, stride=1, batch_normalization=False),
                self.ConvBlock(16, 32),
                self.ConvBlock(32, 64),
                self.ConvBlock(64, 128),
                self.ConvBlock(128, 256)
            )
            self.finale_conv = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=(4, 4), stride=(1, 1),
                                         padding=(0, 0), bias=True, padding_mode="replicate")
            nn.init.trunc_normal_(self.finale_conv.weight, std=1e-3)
            nn.init.constant_(self.finale_conv.bias, 0.0)
            self.bn = nn.BatchNorm2d(1, eps=1e-5, momentum=.9)
            self.lrelu = nn.LeakyReLU(negative_slope=.2)

        def forward(self, input_data):
            rs = self.strides2_stack(input_data)
            rs = self.finale_conv(rs)
            if rs.shape[0] > 1:
                rs = self.bn(rs)
            rs = self.lrelu(rs)
            return rs

    def discriminator_spatial_loss(self, pan, generated):
        averaged = torch.mean(generated, 1, keepdim=True)

        # Loss of discriminator classifying original PAN as 1 (original label is 0)
        spatial_pos = self.spatial_discriminator(pan)
        spatial_pos_loss = self.mse(spatial_pos, torch.ones_like(spatial_pos) * self.b)

        # Loss of discriminator classifying mean of generated image as 0 (fake label is 1)
        spatial_neg = self.spatial_discriminator(averaged)
        spatial_neg_loss = self.mse(spatial_neg, torch.ones_like(spatial_neg) * self.a)
        return spatial_pos_loss + spatial_neg_loss

    def discriminator_spectral_loss(self, ms, generated):

        # Loss of discriminator classifying original MS as 1 (original label is 0)
        spectrum_pos = self.spectral_discriminator(ms)
        spectrum_pos_loss = self.mse(spectrum_pos, torch.ones_like(spectrum_pos) * self.b)

        # Loss of discriminator classifying generated image as 0 (fake label is 1)
        spectrum_neg = self.spectral_discriminator(generated)
        spectrum_neg_loss = self.mse(spectrum_neg, torch.ones_like(spectrum_neg) * self.a)

        return spectrum_pos_loss + spectrum_neg_loss

    def generator_loss(self, pan, ms, generated):

        averaged = torch.mean(generated, 1, keepdim=True)

        # Spatial Loss
        L_spatial = 0
        if self.use_spatial:
            if self.use_highpass:
                details_generated = high_pass(averaged, self.device)
                details_original = high_pass(pan, self.device)
                L_spatial_base = self.mse(details_original, details_generated)  # g spatial loss
            else:
                L_spatial_base = self.mse(pan, averaged)  # g spatial loss
            # L_spatial_base = torch.mean(torch.square(torch.linalg.norm(averaged - pan)))  # g spatial loss
            spatial_neg = self.spatial_discriminator(averaged)
            L_adv2 = self.mse(spatial_neg, torch.ones_like(spatial_neg) * self.d)  # spatial_loss_ad
            L_spatial = 5 * L_spatial_base + 5 * L_adv2

        # Spectral Loss
        L_spectral_base = self.mse(generated, ms)  # g spectrum loss
        # L_spectral_base = torch.mean(torch.square(torch.linalg.norm(generated - ms)))  # g spectrum loss
        spectrum_neg = self.spectral_discriminator(generated)
        L_adv1 = self.mse(spectrum_neg, torch.ones_like(spectrum_neg) * self.c)  # spectrum loss ad
        L_spectral = 1 * L_spectral_base + 1 * L_adv1

        # return L_adv1 + L_spectral_base
        return L_spatial + L_spectral

    # ------------------------- Concrete Interface Methods -----------------------------

    def train_step(self, dataloader):
        self.train(True)

        loss_g_batch = 0
        loss_d_spec_batch = 0
        loss_d_spat_batch = 0
        for batch, data in enumerate(dataloader):
            pan, ms, ms_lr, gt = data

            gt = gt.to(self.device)
            pan = pan.to(self.device)
            ms = ms.to(self.device)
            # ms_lr = ms_lr.to(self.device)

            if len(pan.shape) != len(ms.shape):
                pan = torch.unsqueeze(pan, 0)

            # Generate Data for Discriminators Training
            with torch.no_grad():
                generated_HRMS = self.generate_output(pan, ms=ms)

            # ------------------- Training Discriminators ----------------------------
            self.spatial_discriminator.train(True)
            self.spectral_discriminator.train(True)
            self.generator.train(False)
            self.spatial_discriminator.zero_grad()
            self.spectral_discriminator.zero_grad()

            if self.use_spatial:
                # Spatial Discriminator
                self.optimizer_spatial_disc.zero_grad()
                loss_spatial = self.discriminator_spatial_loss(pan, generated_HRMS)
                loss_spatial.backward()
                self.optimizer_spatial_disc.step()
                loss = loss_spatial.item()
                torch.cuda.empty_cache()
                loss_d_spat_batch += loss

            # Spectral Discriminator
            self.optimizer_spectral_disc.zero_grad()
            loss_spectral = self.discriminator_spectral_loss(ms, generated_HRMS)
            loss_spectral.backward()
            self.optimizer_spectral_disc.step()
            loss = loss_spectral.item()
            torch.cuda.empty_cache()
            loss_d_spec_batch += loss

            # # ------------------- Training Generator ----------------------------
            self.spatial_discriminator.train(False)
            self.spectral_discriminator.train(False)
            self.generator.train(True)
            self.generator.zero_grad()

            # Compute prediction and loss
            self.optimizer_gen.zero_grad()
            generated = self.generator(pan, ms)
            loss_generator = self.generator_loss(pan, ms, generated)

            # Backpropagation
            loss_generator.backward()
            self.optimizer_gen.step()

            loss = loss_generator.item()
            torch.cuda.empty_cache()

            loss_g_batch += loss

        return {"Gen loss": loss_g_batch / len(dataloader),
                "Spat Disc loss": loss_d_spat_batch / len(dataloader),
                "Spec Disc loss": loss_d_spec_batch / len(dataloader)
                }

    def validation_step(self, dataloader):
        self.eval()
        self.spatial_discriminator.eval()
        self.spectral_discriminator.eval()
        self.generator.eval()

        loss_g_batch = 0
        loss_d_spec_batch = 0
        loss_d_spat_batch = 0
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                pan, ms, ms_lr, gt = data

                gt = gt.to(self.device)
                pan = pan.to(self.device)
                ms = ms.to(self.device)
                ms_lr = ms_lr.to(self.device)
                if len(pan.shape) != len(ms.shape):
                    pan = torch.unsqueeze(pan, 0)
                generated_HRMS = self.generate_output(pan, ms=ms)

                if self.use_spatial:
                    d_spat_loss = self.discriminator_spatial_loss(pan, generated_HRMS)
                    loss_d_spat_batch += d_spat_loss.item()

                d_spec_loss = self.discriminator_spectral_loss(ms, generated_HRMS)
                loss_d_spec_batch += d_spec_loss.item()

                gloss = self.generator_loss(pan, ms, generated_HRMS)
                loss_g_batch += gloss.item()

        return {"Gen loss": loss_g_batch / len(dataloader),
                "Spat Disc loss": loss_d_spat_batch / len(dataloader),
                "Spec Disc loss": loss_d_spec_batch / len(dataloader)
                }

    def save_model(self, path):
        torch.save({'gen_state_dict': self.generator.state_dict(),
                    'spat_disc_state_dict': self.spatial_discriminator.state_dict(),
                    'spec_disc_state_dict': self.spectral_discriminator.state_dict(),
                    'gen_optimizer_state_dict': self.optimizer_gen.state_dict(),
                    'spat_disc_optimizer_state_dict': self.optimizer_spatial_disc.state_dict(),
                    'spec_disc_optimizer_state_dict': self.optimizer_spectral_disc.state_dict(),
                    'gen_best_loss': self.best_losses[0],
                    'spat_disc_best_loss': self.best_losses[1],
                    'spec_disc_best_loss': self.best_losses[2],
                    'best_epoch': self.best_epoch,
                    'tot_epochs': self.tot_epochs
                    }, f"{path}")

    def load_model(self, path):
        trained_model = torch.load(f"{path}", map_location=torch.device(self.device))
        self.generator.load_state_dict(trained_model['gen_state_dict'])
        self.spatial_discriminator.load_state_dict(trained_model['spat_disc_state_dict'])
        self.spectral_discriminator.load_state_dict(trained_model['spec_disc_state_dict'])
        self.optimizer_gen.load_state_dict(trained_model['gen_optimizer_state_dict'])
        self.optimizer_spatial_disc.load_state_dict(trained_model['spat_disc_optimizer_state_dict'])
        self.optimizer_spectral_disc.load_state_dict(trained_model['spec_disc_optimizer_state_dict'])
        self.tot_epochs = trained_model['tot_epochs']
        self.best_epoch = trained_model['best_epoch']
        self.best_losses = [trained_model['gen_best_loss'],
                            trained_model['spat_disc_best_loss'],
                            trained_model['spec_disc_best_loss']]

    def generate_output(self, pan, **kwargs):
        return self.generator(pan, kwargs['ms'])

    def set_optimizers_lr(self, lr):
        for g in self.optimizer_gen.param_groups:
            g['lr'] = lr
        for g in self.optimizer_spatial_disc.param_groups:
            g['lr'] = lr
        for g in self.optimizer_spectral_disc.param_groups:
            g['lr'] = lr
