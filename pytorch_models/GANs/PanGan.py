from abc import ABC

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.nn.functional import interpolate

from pytorch_models.GANs.GanInterface import GanInterface
from pytorch_models.adversarial_losses import LsganLoss


def downsample(img, new_shape):
    return interpolate(img, new_shape, mode='bilinear', antialias=True)


kernel = np.array([[1., 1., 1.], [1., -8., 1.], [1., 1., 1.]])
kernel = torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0).float()


def high_pass(img, device='cpu'):
    global kernel
    img_hp = torch.nn.functional.conv2d(img, kernel.to(device), stride=(1, 1), padding='same')
    return img_hp


class PanGan(GanInterface, ABC):
    def __init__(self, channels, device="cpu", name="PanGan"):
        super(PanGan, self).__init__(name=name, device=device)

        self.generator = PanGan.Generator(channels)
        self.spatial_discriminator = PanGan.Discriminator(1)
        self.spectral_discriminator = PanGan.Discriminator(channels)

        # Discriminator
        self.a = 0  # Label for Original
        self.b = 1  # Label for Fake

        # Generator
        self.c = 1  # Label for Fake
        self.d = 1  # Label for Fake

        self.alpha = .002
        self.beta = .001
        self.mu = 5

        self.best_losses = [np.inf, np.inf, np.inf]
        self.mse = torch.nn.MSELoss(reduction='mean')

        self.optimizer_gen = optim.Adam(self.generator.parameters(), lr=.001, weight_decay=.99)
        self.optimizer_spatial_disc = optim.Adam(self.spatial_discriminator.parameters(), lr=.001)
        self.optimizer_spectral_disc = optim.Adam(self.spectral_discriminator.parameters(), lr=.001)

    # ------------------------- Specific GAN Methods -----------------------------------
    class Generator(nn.Module):
        def init_weights(self, m):
            if isinstance(m, nn.Conv2d):
                torch.nn.init.trunc_normal_(m.weight, std=1e-3)
                m.bias.data.fill_(0.0)

        def __init__(self, in_channels):
            super(PanGan.Generator, self).__init__()

            # BxC+1xHxW ---> Bx64xHxW
            self.block_1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels + 1, out_channels=64, kernel_size=(9, 9), padding="same",
                          stride=(1, 1), bias=True, padding_mode="replicate"),
                nn.BatchNorm2d(64, eps=1e-5, momentum=.9),
                nn.ReLU()
            )
            self.block_1.apply(self.init_weights)

            # Bx64+C+1xHxW ---> Bx32xHxW
            self.block_2 = nn.Sequential(
                nn.Conv2d(in_channels=64 + in_channels + 1, out_channels=32, kernel_size=(5, 5), padding="same",
                          stride=(1, 1), bias=True, padding_mode="replicate"),
                nn.BatchNorm2d(32, eps=1e-5, momentum=.9),
                nn.ReLU()
            )
            self.block_2.apply(self.init_weights)

            # Bx32+64+C+1  ---> BxCxHxW
            self.block_3 = nn.Sequential(
                nn.Conv2d(in_channels=32 + 64 + in_channels + 1, out_channels=in_channels, kernel_size=(5, 5),
                          padding="same", stride=(1, 1), bias=True, padding_mode="replicate"),
                nn.Tanh()
            )
            self.block_3.apply(self.init_weights)

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
        pred_fake = self.spatial_discriminator(averaged)
        pred_real = self.spatial_discriminator(pan)
        return self.adv_loss(pred_fake, pred_real)

    def discriminator_spectral_loss(self, ms, generated):
        pred_fake = self.spectral_discriminator(generated)
        pred_real = self.spectral_discriminator(ms)
        return self.adv_loss(pred_fake, pred_real)

    def generator_loss(self, pan, ms, generated):

        averaged = torch.mean(generated, 1, keepdim=True)
        details_generated = high_pass(averaged, self.device)
        details_original = high_pass(pan, self.device)

        # Reconstruction Loss
        L_rec_spatial = self.rec_loss(details_original, details_generated)
        L_rec_spectral = self.rec_loss(generated, ms)

        # Adversarial Loss
        pred_fake_spec = self.spectral_discriminator(generated)
        pred_fake_spat = self.spatial_discriminator(averaged)

        if self.adv_loss.use_real_data:
            pred_real_spat = self.spatial_discriminator(pan.detach())
            pred_real_spec = self.spectral_discriminator(ms.detach())
        else:
            pred_real_spat = None
            pred_real_spec = None

        L_adv_spatial = self.adv_loss(pred_fake_spat, pred_real_spat, True)
        L_adv_spectral = self.adv_loss(pred_fake_spec, pred_real_spec, True)

        # Total Loss
        L_spatial = self.mu * L_rec_spatial + self.mu * L_adv_spatial
        L_spectral = 1 * L_rec_spectral + 1 * L_adv_spectral

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

            pan = pan.to(self.device)
            ms = ms.to(self.device)

            if len(pan.shape) != len(ms.shape):
                pan = torch.unsqueeze(pan, 0)

            # Generate Data for Discriminators Training
            with torch.no_grad():
                generated_HRMS = self.generate_output(pan, ms)

            # ------------------- Training Discriminators ----------------------------
            self.spatial_discriminator.train(True)
            self.spectral_discriminator.train(True)
            self.generator.train(False)
            self.spatial_discriminator.zero_grad()
            self.spectral_discriminator.zero_grad()
            self.optimizer_spatial_disc.zero_grad()
            self.optimizer_spectral_disc.zero_grad()

            # Spatial Discriminator
            loss_spatial = self.discriminator_spatial_loss(pan, generated_HRMS)
            loss_spatial.backward()
            self.optimizer_spatial_disc.step()
            loss = loss_spatial.item()
            torch.cuda.empty_cache()
            loss_d_spat_batch += loss

            # Spectral Discriminator
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
            self.optimizer_gen.zero_grad()

            # Compute prediction and loss
            generated = self.generate_output(pan, ms, evaluation=False)
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

                pan = pan.to(self.device)
                ms = ms.to(self.device)

                if len(pan.shape) != len(ms.shape):
                    pan = torch.unsqueeze(pan, 0)
                generated_HRMS = self.generate_output(pan, ms)

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
                    'tot_epochs': self.tot_epochs,
                    'metrics': [self.best_q, self.best_q_avg, self.best_sam, self.best_ergas]
                    }, f"{path}")

    def load_model(self, path, weights_only=False):
        trained_model = torch.load(f"{path}", map_location=torch.device(self.device))
        self.generator.load_state_dict(trained_model['gen_state_dict'])
        self.spatial_discriminator.load_state_dict(trained_model['spat_disc_state_dict'])
        self.spectral_discriminator.load_state_dict(trained_model['spec_disc_state_dict'])
        if weights_only:
            return
        self.optimizer_gen.load_state_dict(trained_model['gen_optimizer_state_dict'])
        self.optimizer_spatial_disc.load_state_dict(trained_model['spat_disc_optimizer_state_dict'])
        self.optimizer_spectral_disc.load_state_dict(trained_model['spec_disc_optimizer_state_dict'])
        self.tot_epochs = trained_model['tot_epochs']
        self.best_epoch = trained_model['best_epoch']
        self.best_losses = [trained_model['gen_best_loss'],
                            trained_model['spat_disc_best_loss'],
                            trained_model['spec_disc_best_loss']]
        try:
            self.best_q, self.best_q_avg, self.best_sam, self.best_ergas = trained_model['metrics']
        except KeyError:
            pass

    def set_optimizers_lr(self, lr):
        for g in self.optimizer_gen.param_groups:
            g['lr'] = lr
        for g in self.optimizer_spatial_disc.param_groups:
            g['lr'] = lr
        for g in self.optimizer_spectral_disc.param_groups:
            g['lr'] = lr

    def define_losses(self, rec_loss=None, adv_loss=None):
        self.rec_loss = rec_loss if rec_loss is not None else torch.nn.MSELoss(reduction='mean')
        self.adv_loss = adv_loss if adv_loss is not None else LsganLoss()


if __name__ == '__main__':
    from train_file import create_test_dict
    from constants import DATASET_DIR

    x = create_test_dict(f"{DATASET_DIR}/FR/W3/test_1_256.h5", "xx")
    print(x.keys())
