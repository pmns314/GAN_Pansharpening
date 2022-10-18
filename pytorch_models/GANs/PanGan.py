from abc import ABC

import torch
from torch import nn, optim

from pytorch_models.GANs.GanInterface import GanInterface
from torch.nn.functional import interpolate
import numpy as np


def downsample(img, new_shape):
    return interpolate(img, new_shape, mode='bilinear', antialias=True)


def high_pass(img, device='cpu'):
    blur_kerel = np.zeros(shape=(1, img.shape[1], 3, 3), dtype=np.float32)
    value = np.array([[1., 1., 1.], [1., -8., 1.], [1., 1., 1.]])
    blur_kerel[:, :, :, :] = value
    img_hp = torch.nn.functional.conv2d(img, torch.from_numpy(blur_kerel).to(device), stride=(1, 1), padding='same')
    return img_hp


class PanGan(GanInterface, ABC):
    def __init__(self, channels, device="cpu", name="PanGan"):
        super(PanGan, self).__init__(name=name, device=device)

        self.generator = PanGan.Generator(channels)
        self.spatial_discriminator = PanGan.Discriminator(1)
        self.spectral_discriminator = PanGan.Discriminator(channels)

        self.a = .2
        self.b = .8
        self.c = 1
        self.d = 1

        self.alpha = .002
        self.beta = .001
        self.mu = 5

        self.best_losses = [np.inf, np.inf, np.inf]
        self.mse = torch.nn.MSELoss(reduction='mean')
        self.optimizer_gen = optim.Adam(self.generator.parameters(), lr=.001)
        self.optimizer_spatial_disc = optim.Adam(self.spatial_discriminator.parameters(), lr=.001)
        self.optimizer_spectral_disc = optim.Adam(self.spectral_discriminator.parameters(), lr=.001)

    def set_optimizers_lr(self, lr):
        self.optimizer_gen = optim.Adam(self.generator.parameters(), lr=lr)
        self.optimizer_spatial_disc = optim.Adam(self.spatial_discriminator.parameters(), lr=lr)
        self.optimizer_spectral_disc = optim.Adam(self.spectral_discriminator.parameters(), lr=lr)

    # ------------------------- Specific GAN Methods -----------------------------------
    class Generator(nn.Module):
        def __init__(self, in_channels):
            super(PanGan.Generator, self).__init__()

            # BxC+1xHxW ---> Bx64xHxW
            self.block_1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels + 1, out_channels=64, kernel_size=(9, 9), padding="same", bias=True),
                nn.ReLU(),
                nn.BatchNorm2d(64, eps=1e-5, momentum=.9)
            )

            # Bx64+C+1xHxW ---> Bx32xHxW
            self.block_2 = nn.Sequential(
                nn.Conv2d(in_channels=64 + in_channels + 1, out_channels=32, kernel_size=(9, 9), padding="same",
                          bias=True),
                nn.ReLU(),
                nn.BatchNorm2d(32, eps=1e-5, momentum=.9)
            )

            # Bx32+64+C+1  ---> BxCxHxW
            self.block_3 = nn.Sequential(
                nn.Conv2d(in_channels=32 + 64 + in_channels + 1, out_channels=in_channels, kernel_size=(5, 5),
                          padding="same",
                          bias=True),
                nn.ReLU(),
                nn.BatchNorm2d(in_channels, eps=1e-5, momentum=.9)
            )

        def forward(self, pan, ms):
            input_block_1 = torch.cat([pan, ms], 1)
            output_block_1 = self.block_1(input_block_1)

            input_block_2 = torch.cat([ms, output_block_1, pan], 1)
            output_block_2 = self.block_2(input_block_2)

            input_block_3 = torch.cat([input_block_1, output_block_1, output_block_2], 1)
            output_block_3 = self.block_3(input_block_3)

            return output_block_3

    class Discriminator(nn.Module):
        class ConvBlock(nn.Module):
            def __init__(self, in_channels, out_channels, batch_normalization=True):
                super(PanGan.Discriminator.ConvBlock, self).__init__()
                self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3),
                                       stride=(2, 2), padding=(2, 2), bias=True)
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
                self.ConvBlock(channels, 16, batch_normalization=False),
                self.ConvBlock(16, 32),
                self.ConvBlock(32, 64),
                self.ConvBlock(64, 128),
                self.ConvBlock(128, 256)
            )
            self.last_conv2 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=(4, 4), stride=(1, 1),
                                        padding=(1, 1), bias=True)
            self.lrelu = nn.LeakyReLU(negative_slope=.2)

        def forward(self, input_data):
            rs = self.strides2_stack(input_data)
            rs = self.last_conv2(rs)
            rs = self.lrelu(rs)
            return rs

    def discriminator_spatial_loss(self, pan, generated):
        averaged = torch.mean(generated, 1, keepdim=True)
        spatial_neg_loss = self.mse(self.spatial_discriminator(pan) - self.b)
        spatial_pos_loss = self.mse(self.spatial_discriminator(averaged) - self.a)
        return spatial_pos_loss + spatial_neg_loss

    def discriminator_spectral_loss(self, ms, generated):
        spectrum_neg_loss = self.mse(self.spectral_discriminator(ms) - self.b)
        spectrum_pos_loss = self.mse(self.spectral_discriminator(generated) - self.a)
        return spectrum_pos_loss + spectrum_neg_loss

    def generator_loss(self, pan, ms_lr, generated):
        # Spectral Loss
        downsampled = downsample(generated, (ms_lr.shape[2:]))
        L_spectral_base = self.mse(downsampled - ms_lr)
        L_adv1 = self.mse(self.spectral_discriminator(generated) - self.c)
        L_spectral = L_spectral_base + self.alpha * L_adv1

        # Spatial Loss
        averaged = torch.mean(generated, 1, keepdim=True)
        details_generated = high_pass(averaged, self.device)
        details_original = high_pass(pan, self.device)
        L_spatial_base = self.mu * self.mse(details_generated - details_original)
        L_adv2 = self.mse(self.spatial_discriminator(averaged) - self.d)
        L_spatial = L_spatial_base + self.beta * L_adv2

        return 5 * L_adv2 + L_adv1 + 5 * L_spatial_base + L_spectral_base

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
            ms_lr = ms_lr.to(self.device)

            # Generate Data for Discriminators Training
            with torch.no_grad():
                generated_HRMS = self.generate_output(pan, ms=ms)

            # ------------------- Training Discriminators ----------------------------
            self.spatial_discriminator.train(True)
            self.spectral_discriminator.train(True)
            self.generator.train(False)
            self.spatial_discriminator.zero_grad()
            self.spectral_discriminator.zero_grad()

            # Spatial Discriminator
            loss_spatial = self.discriminator_spatial_loss(pan, generated_HRMS)
            self.optimizer_spatial_disc.zero_grad()
            loss_spatial.backward()
            self.optimizer_spatial_disc.step()
            loss = loss_spatial.item()
            torch.cuda.empty_cache()
            loss_d_spat_batch += loss

            # Spectral Discriminator
            loss_spectral = self.discriminator_spectral_loss(ms, generated_HRMS)
            self.optimizer_spectral_disc.zero_grad()
            loss_spectral.backward()
            self.optimizer_spectral_disc.step()
            loss = loss_spectral.item()
            torch.cuda.empty_cache()
            loss_d_spec_batch += loss

            # ------------------- Training Generator ----------------------------
            self.spatial_discriminator.train(False)
            self.spectral_discriminator.train(False)
            self.generator.train(True)
            self.generator.zero_grad()

            # Compute prediction and loss
            generated = self.generator(pan, ms)
            loss_generator = self.generator_loss(pan, ms_lr, generated)

            # Backpropagation
            self.optimizer_gen.zero_grad()
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

                generated_HRMS = self.generate_output(pan, ms=ms)

                d_spat_loss = self.discriminator_spatial_loss(pan, generated_HRMS)
                loss_d_spat_batch += d_spat_loss.item()

                d_spec_loss = self.discriminator_spectral_loss(ms, generated_HRMS)
                loss_d_spec_batch += d_spec_loss.item()

                gloss = self.generator_loss(pan, ms_lr, generated_HRMS)
                loss_g_batch += gloss.item()

        return {"Gen loss": loss_g_batch / len(dataloader),
                "Spat Disc loss": loss_d_spat_batch / len(dataloader),
                "Spec Disc loss": loss_d_spec_batch / len(dataloader)
                }

    def save_checkpoint(self, path, curr_epoch):
        torch.save({'gen_state_dict': self.generator.state_dict(),
                    'spat_disc_state_dict': self.spatial_discriminator.state_dict(),
                    'spec_disc_state_dict': self.spectral_discriminator.state_dict(),
                    'gen_optimizer_state_dict': self.optimizer_gen.state_dict(),
                    'spat_disc_optimizer_state_dict': self.optimizer_spatial_disc.state_dict(),
                    'spec_disc_optimizer_state_dict': self.optimizer_spectral_disc.state_dict(),
                    'gen_best_loss': self.best_losses[0],
                    'spat_disc_best_loss': self.best_losses[1],
                    'spec_disc_best_loss': self.best_losses[2]
                    }, f"{path}/checkpoint_{curr_epoch}.pth")

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
                    'tot_epochs': self.pretrained_epochs
                    }, f"{path}/model.pth")

    def load_model(self, path, lr=None):
        trained_model = torch.load(f"{path}/model.pth", map_location=torch.device(self.device))
        self.generator.load_state_dict(trained_model['gen_state_dict'])
        self.spatial_discriminator.load_state_dict(trained_model['spat_disc_state_dict'])
        self.spectral_discriminator.load_state_dict(trained_model['spec_disc_state_dict'])
        self.optimizer_gen.load_state_dict(trained_model['gen_optimizer_state_dict'])
        self.optimizer_spatial_disc.load_state_dict(trained_model['spat_disc_optimizer_state_dict'])
        self.optimizer_spectral_disc.load_state_dict(trained_model['spec_disc_optimizer_state_dict'])
        self.pretrained_epochs = trained_model['tot_epochs']
        self.best_epoch = trained_model['best_epoch']
        self.best_losses = [trained_model['gen_best_loss'],
                            trained_model['spat_disc_best_loss'],
                            trained_model['spec_disc_best_loss']]
        if lr is not None:
            for g in self.gen_opt.param_groups:
                g['lr'] = lr
            for g in self.disc_opt.param_groups:
                g['lr'] = lr

    def generate_output(self, pan, **kwargs):
        return self.generator(pan, kwargs['ms'])
