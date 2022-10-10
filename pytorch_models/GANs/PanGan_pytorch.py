import numpy as np
import torch
import torchsummary
from torch import nn
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader

from dataset.DatasetPytorch import DatasetPytorch


class Generator(nn.Module):
    def __init__(self, in_channels):
        super(Generator, self).__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels + 1, out_channels=64, kernel_size=(9, 9), padding="same", bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(64, eps=1e-5, momentum=.9)
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=64 + in_channels + 1, out_channels=32, kernel_size=(9, 9), padding="same", bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(32, eps=1e-5, momentum=.9)
        )

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
            super(Discriminator.ConvBlock, self).__init__()
            self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3),
                                   stride=(2, 2),
                                   padding=(2, 2), bias=True)
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
        super(Discriminator, self).__init__()
        self.strides2_stack = nn.Sequential(
            self.ConvBlock(channels, 16, batch_normalization=False),
            self.ConvBlock(16, 32),
            self.ConvBlock(32, 64),
            self.ConvBlock(64, 128),
            self.ConvBlock(128, 256)
        )
        self.last_conv2 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=(4, 4), stride=(1, 1), padding="valid",
                                    bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=.2)

    def forward(self, input_data):
        rs = self.strides2_stack(input_data)
        rs = self.last_conv2(rs)
        rs = self.lrelu(rs)
        return rs


class PanGan:
    def __init__(self, channels):
        self.generator = Generator(channels)
        self.spatial_discriminator = Discriminator(1)
        self.spectral_discriminator = Discriminator(channels)

        self.a = .2
        self.b = .8
        self.c = .9
        self.d = .9

        self.alpha = .002
        self.beta = .001
        self.mu = 5

        self.optimizer_gen = torch.optim.Adam(self.generator.parameters(), lr=.001)
        self.optimizer_spatial_disc = torch.optim.Adam(self.spatial_discriminator.parameters(), lr=.001)
        self.optimizer_spectral_disc = torch.optim.Adam(self.spectral_discriminator.parameters(), lr=.001)

    def discriminator_spatial_loss(self, pan, generated):
        averaged = torch.mean(generated, 1, keepdim=True)
        spatial_neg_loss = torch.mean(torch.square(self.spatial_discriminator(pan) - self.b))
        spatial_pos_loss = torch.mean(torch.square(self.spatial_discriminator(averaged) - self.a))
        return spatial_pos_loss + spatial_neg_loss

    def discriminator_spectral_loss(self, ms, generated):
        spectrum_neg_loss = torch.mean(torch.square(self.spectral_discriminator(ms) - self.b))
        spectrum_pos_loss = torch.mean(torch.square(self.spectral_discriminator(generated) - self.a))
        return spectrum_pos_loss + spectrum_neg_loss

    def generator_loss(self, pan, ms_lr, generated):
        # Spectral Loss
        downsampled = downsample(generated, (ms_lr.shape[2:]))
        L_spectral_base = torch.mean(torch.square(torch.linalg.matrix_norm((downsampled - ms_lr), 'fro')))
        L_adv1 = torch.mean(torch.square(self.spectral_discriminator(generated) - self.c))
        L_spectral = L_spectral_base + self.alpha * L_adv1

        # Spatial Loss
        averaged = torch.mean(generated, 1, keepdim=True)
        details_generated = high_pass(averaged)
        details_original = high_pass(pan)
        L_spatial_base = self.mu * torch.mean(
            torch.square(torch.linalg.matrix_norm((details_generated - details_original), 'fro')))
        L_adv2 = torch.mean(torch.square(self.spatial_discriminator(averaged) - self.d))
        L_spatial = L_spatial_base + self.beta * L_adv2

        return L_spatial + L_spectral

    def train(self, num_epochs, train_loader):
        self.spatial_discriminator.training = False
        self.spectral_discriminator.training = False
        self.generator.training = False

        for epoch in range(num_epochs):
            for n, (_, original_pans, original_ms, original_ms_lr) in enumerate(train_loader):

                # Generate Data for Discriminators Training
                generated_hrms = self.generator(original_ms, original_pans)

                # Train Discriminators
                self.spatial_discriminator.training = True
                self.spatial_discriminator.zero_grad()
                loss_spatial_discriminator = self.discriminator_spatial_loss(original_pans, generated_hrms)
                loss_spatial_discriminator.backward(retain_graph=True)
                self.optimizer_spatial_disc.step()
                self.spatial_discriminator.training = False

                self.spectral_discriminator.training = True
                self.spectral_discriminator.zero_grad()
                loss_spectral_discriminator = self.discriminator_spectral_loss(original_ms, generated_hrms)
                loss_spectral_discriminator.backward(retain_graph=True)
                self.optimizer_spectral_disc.step()
                self.spectral_discriminator.training = False

                # Train Generator
                self.generator.training = True
                self.generator.zero_grad()
                generated = self.generator(original_pans, original_ms)
                loss_generator = self.generator_loss(original_pans, original_ms_lr, generated)
                loss_generator.backward(retain_graph=True)
                self.optimizer_gen.step()
                self.generator.training = False

                # Show loss
                #if epoch % 10 == 0 and n == train_loader.batch_size - 1:
            print(f"Epoch: {epoch} Loss D. - Spatial: {loss_spatial_discriminator}")
            print(f"Epoch: {epoch} Loss D. - Spectral: {loss_spectral_discriminator}")
            print(f"Epoch: {epoch} Loss G.: {loss_generator}")


def downsample(img, new_shape):
    return interpolate(img, new_shape, mode='bilinear', antialias=True)


def high_pass(img):
    blur_kerel = np.zeros(shape=(1, img.shape[1], 3, 3), dtype=np.float32)
    value = np.array([[1., 1., 1.], [1., -8., 1.], [1., 1., 1.]])
    blur_kerel[:, :, :, :] = value
    img_hp = torch.nn.functional.conv2d(img, torch.from_numpy(blur_kerel), stride=(1, 1), padding='same')
    return img_hp


if __name__ == '__main__':
    train_dataloader = DataLoader(
        DatasetPytorch("C:/Users/pmans/Documenti/Progetti_local/Pycharm/GAN-PAN/datasets/train.h5"), batch_size=64,
        shuffle=True)
    val_dataloader = DataLoader(
        DatasetPytorch("C:/Users/pmans/Documenti/Progetti_local/Pycharm/GAN-PAN/datasets/valid.h5"), batch_size=64,
        shuffle=True)
    test_dataloader = DataLoader(
        DatasetPytorch("C:/Users/pmans/Documenti/Progetti_local/Pycharm/GAN-PAN/datasets/test.h5"), batch_size=64,
        shuffle=True)
    gt, pan, ms, ms_lr = train_dataloader.dataset[:]
    channels = gt.shape[1]

    model = PanGan(channels)
    model.train(20, train_loader=train_dataloader)
