from abc import ABC

import numpy as np
import torch
from torch import nn

from pytorch_models.GANs.GanInterface import GanInterface


class PanColorGan(GanInterface, ABC):
    def __init__(self, channels, device="cpu", name="PanColorGan"):
        super().__init__(device, name)
        self.best_losses = [np.inf, np.inf]
        self.generator = PanColorGan.Generator(channels)
        self.discriminator = PanColorGan.Discriminator(channels)
        self.mse = torch.nn.MSELoss(reduction='mean')
        self.gen_opt = torch.optim.Adam(self.generator.parameters())
        self.disc_opt = torch.optim.Adam(self.discriminator.parameters())

    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels, kernel, padding=0, stride=1, use_dropout=False):
            super().__init__()
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel, kernel),
                                  padding=(padding, padding), stride=(stride, stride), bias=True)
            self.bn = nn.BatchNorm2d(out_channels)
            self.lrelu = nn.LeakyReLU(.2)
            self.dropout = nn.Dropout2d()
            self.use_dropout = use_dropout

        def forward(self, input_tensor):
            out = self.conv(input_tensor)
            out = self.bn(out)
            out = self.lrelu(out)
            if self.use_dropout:
                out = self.dropout(out)
            return out

    class Generator(nn.Module):

        class UpConvBlock(nn.Module):
            def __init__(self, in_channels, out_channels, kernel, padding=0, stride=1, out_padding=0):
                super().__init__()
                self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                               kernel_size=(kernel, kernel), padding=(padding, padding),
                                               stride=(stride, stride), bias=True,
                                               output_padding=(out_padding, out_padding))
                self.bn = nn.BatchNorm2d(out_channels)
                self.lrelu = nn.LeakyReLU(.2)

            def forward(self, input_tensor):
                out = self.conv(input_tensor)
                out = self.bn(out)
                out = self.lrelu(out)
                return out

        class ResnetBlock(nn.Module):
            def __init__(self, channels, kernel, padding=0, use_dropout=False):
                super().__init__()
                self.conv1 = nn.Conv2d(channels, channels, kernel, padding=(padding, padding))
                self.lrelu = nn.LeakyReLU(.2, True)
                self.bn = nn.BatchNorm2d(channels)
                self.use_dropout = use_dropout
                self.dropout = nn.Dropout(.2)
                self.conv2 = nn.Conv2d(channels, channels, kernel, padding=(padding, padding))

            def forward(self, input_tensor):
                out = self.conv1(input_tensor)
                out = self.bn(out)
                out = self.lrelu(out)
                if self.use_dropout:
                    out = self.dropout(out)
                out = self.conv2(out)
                out = self.bn(out)
                return out + input_tensor

        def __init__(self, channels, name="Gen"):
            super().__init__()
            self._model_name = name
            self.channels = channels

            # Color Injection
            self.color1 = nn.Sequential(
                PanColorGan.ConvBlock(channels, 32, 3, padding=1),
                PanColorGan.ConvBlock(32, 32, 3, padding=1)
            )
            self.color2 = PanColorGan.ConvBlock(32, 64, 3, stride=2, padding=1)
            self.color3 = PanColorGan.ConvBlock(64, 128, 3, stride=2, padding=1)

            # Spatial Details Extraction
            self.model1 = nn.Sequential(
                PanColorGan.ConvBlock(1, 32, 3, padding=1),
                PanColorGan.ConvBlock(32, 32, 3, padding=1)
            )
            self.model2 = PanColorGan.ConvBlock(64, 64, 3, stride=2, padding=1)
            self.model3 = PanColorGan.ConvBlock(128, 128, 3, stride=2, padding=1)

            # Feature Transformation
            self.feature_transformation = nn.Sequential(
                PanColorGan.Generator.ResnetBlock(256, 3, padding=1),
                PanColorGan.Generator.ResnetBlock(256, 3, padding=1),
                PanColorGan.Generator.ResnetBlock(256, 3, padding=1),
                PanColorGan.Generator.ResnetBlock(256, 3, padding=1),
                PanColorGan.Generator.ResnetBlock(256, 3, padding=1),
                PanColorGan.Generator.ResnetBlock(256, 3, padding=1)
            )

            # Image Synthesis
            self.model4 = PanColorGan.ConvBlock(256, 128, 3, padding=1)
            self.model5 = PanColorGan.Generator.UpConvBlock(256, 128, 3, stride=2, padding=1, out_padding=1)

            self.model6 = PanColorGan.ConvBlock(128, 64, 3, padding=1)
            self.model7 = PanColorGan.Generator.UpConvBlock(128, 64, 3, stride=2, padding=1, out_padding=1)

            self.model8 = PanColorGan.ConvBlock(64, 32, 3, padding=1)
            self.model9 = nn.Sequential(
                PanColorGan.ConvBlock(64, 32, 3, padding=1),
                nn.Conv2d(32, channels, kernel_size=(3, 3), padding=(1, 1))
            )
            self.out_model = nn.Tanh()

        def forward(self, ms, pan):
            m1 = self.model1(pan)
            c1 = self.color1(ms)
            mc1 = torch.cat([m1, c1], 1)
            m2 = self.model2(mc1)
            c2 = self.color2(c1)
            mc2 = torch.cat([m2, c2], 1)
            m3 = self.model3(mc2)
            c3 = self.color3(c2)
            mc3 = torch.cat([m3, c3], 1)

            res = self.feature_transformation(mc3)

            m4 = self.model4(res)
            m34 = torch.cat([m3, m4], 1)
            m5 = self.model5(m34)
            m6 = self.model6(m5)
            m26 = torch.cat([m2, m6], 1)
            m7 = self.model7(m26)
            m8 = self.model8(m7)
            m18 = torch.cat([m1, m8], 1)
            m9 = self.model9(m18)
            out = self.out_model(m9)
            return out

    class Discriminator(nn.Module):
        def __init__(self, channels):
            super().__init__()
            in_channels = channels * 2 + 1
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=(4, 4), stride=(2, 2), padding=(2, 2)),
                nn.LeakyReLU(.2, True),

                PanColorGan.ConvBlock(32, 64, 4, stride=2, padding=2),
                PanColorGan.ConvBlock(64, 128, 4, stride=2, padding=2),
                PanColorGan.ConvBlock(128, 256, 4, stride=2, padding=2),
                PanColorGan.ConvBlock(256, 256, 4, stride=1, padding=2),

                nn.Conv2d(256, 1, kernel_size=(4, 4), stride=(1, 1), padding=(2, 2)),

                nn.Sigmoid()
            )

        def forward(self, input_tensor):
            return self.net(input_tensor)

    def loss_discriminator(self, ms, pan, gt, generated):
        fake_ab = torch.cat([ms, pan, generated], 1)
        real_ab = torch.cat([ms, pan, gt], 1)

        pred_fake = self.discriminator(fake_ab)
        pred_real = self.discriminator(real_ab)

        ones = torch.ones_like(pred_fake)
        zeros = torch.zeros_like(pred_fake)

        # Label 1 for fake, Label 0 for True
        loss_d_fake = self.mse(pred_fake - torch.mean(pred_real), ones)
        loss_d_real = self.mse(pred_real - torch.mean(pred_fake), zeros)

        return (loss_d_real + loss_d_fake) / 2

    def loss_generator(self, ms, pan, gt):
        generated = self.generator(ms, pan)

        fake_ab = torch.cat([ms, pan, generated], 1)
        real_ab = torch.cat([ms, pan, gt], 1)

        pred_fake = self.discriminator(fake_ab)
        pred_real = self.discriminator(real_ab)

        ones = torch.ones_like(pred_fake)
        zeros = torch.zeros_like(pred_fake)

        loss_g_real = self.mse(pred_real - torch.mean(pred_fake), zeros)
        loss_g_fake = self.mse(pred_fake - torch.mean(pred_real), ones)

        return (loss_g_real + loss_g_fake) / 2

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

            # Generate Data for Discriminators Training
            with torch.no_grad():
                generated_HRMS = self.generate_output(pan, ms=ms)

            # ------------------- Training Discriminator ----------------------------
            self.discriminator.train(True)
            self.generator.train(False)
            self.discriminator.zero_grad()

            # Compute prediction and loss
            loss_d = self.loss_discriminator(ms, pan, gt, generated_HRMS)

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
            loss_g = self.loss_generator(ms, pan, gt)

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

                generated = self.generate_output(pan, ms=ms)

                dloss = self.loss_discriminator(ms, pan, gt, generated)
                disc_loss += dloss.item()

                gloss = self.loss_generator(ms, pan, gt)
                gen_loss += gloss.item()

        return {"Gen loss": gen_loss / len(dataloader),
                "Disc loss": disc_loss / len(dataloader)
                }

    def save_checkpoint(self, path, curr_epoch):
        torch.save({'gen_state_dict': self.generator.state_dict(),
                    'disc_state_dict': self.discriminator.state_dict(),
                    'gen_optimizer_state_dict': self.gen_opt.state_dict(),
                    'disc_optimizer_state_dict': self.disc_opt.state_dict(),
                    'gen_best_loss': self.best_losses[0],
                    'disc_best_loss': self.best_losses[1]
                    }, f"{path}/checkpoint_{curr_epoch}.pth")

    def save_model(self, path):
        torch.save({
            'best_epoch': self.best_epoch,
            'gen_state_dict': self.generator.state_dict(),
            'disc_state_dict': self.discriminator.state_dict(),
            'gen_optimizer_state_dict': self.gen_opt.state_dict(),
            'disc_optimizer_state_dict': self.disc_opt.state_dict(),
            'gen_best_loss': self.best_losses[0],
            'disc_best_loss': self.best_losses[1],
            'tot_epochs': self.pretrained_epochs
        }, f"{path}/model.pth")

    def load_model(self, path):
        trained_model = torch.load(f"{path}/model.pth", map_location=torch.device(self.device))
        self.generator.load_state_dict(trained_model['gen_state_dict'])
        self.discriminator.load_state_dict(trained_model['disc_state_dict'])
        self.gen_opt.load_state_dict(trained_model['gen_optimizer_state_dict'])
        self.disc_opt.load_state_dict(trained_model['disc_optimizer_state_dict'])
        self.pretrained_epochs = trained_model['tot_epochs']
        self.best_epoch = trained_model['best_epoch']
        self.best_losses = [trained_model['gen_best_loss'], trained_model['disc_best_loss']]

    def generate_output(self, pan, **kwargs):
        ms = kwargs['ms']
        return self.generator(ms, pan)

    def set_optimizers_lr(self, lr):
        for g in self.gen_opt.param_groups:
            g['lr'] = lr
        for g in self.disc_opt.param_groups:
            g['lr'] = lr