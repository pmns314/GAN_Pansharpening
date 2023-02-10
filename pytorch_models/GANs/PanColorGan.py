from abc import ABC

import numpy as np
import pandas as pd
import torch
from torch import nn

from pytorch_models.GANs.GanInterface import GanInterface
from pytorch_models.adversarial_losses import RaganLoss
from quality_indexes_toolbox.indexes_evaluation import indexes_evaluation
from constants import *


class PanColorGan(GanInterface, ABC):
    def __init__(self, channels, device="cpu", name="PanColorGan", padding_mode="replicate"):
        super().__init__(device, name)
        self.best_losses = [np.inf, np.inf]
        self.generator = PanColorGan.Generator(channels, padding_mode)
        self.discriminator = PanColorGan.Discriminator(channels)
        self.mse = torch.nn.MSELoss(reduction='mean')
        self.mae = torch.nn.L1Loss(reduction='mean')
        self.lambda_factor = 1
        self.weight_gan = 1
        self.gen_opt = torch.optim.Adam(self.generator.parameters())
        self.disc_opt = torch.optim.Adam(self.discriminator.parameters())

    # ------------------------- Specific GAN Methods -----------------------------------
    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels, kernel, padding_mode="replicate", padding=0, stride=1,
                     use_dropout=False):
            super().__init__()
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel, kernel),
                                  padding=(padding, padding), stride=(stride, stride), padding_mode=padding_mode,
                                  bias=True)
            self.bn = nn.BatchNorm2d(out_channels, affine=True)
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
                self.bn = nn.BatchNorm2d(out_channels, affine=True)
                self.lrelu = nn.LeakyReLU(.2)

            def forward(self, input_tensor):
                out = self.conv(input_tensor)
                out = self.bn(out)
                out = self.lrelu(out)
                return out

        class ResnetBlock(nn.Module):
            def __init__(self, channels, kernel, padding_mode, padding=1, use_dropout=False):
                super().__init__()
                self.conv1 = nn.Conv2d(channels, channels, kernel, padding=(padding, padding),
                                       padding_mode=padding_mode)
                self.lrelu = nn.LeakyReLU(.2, True)
                self.bn = nn.BatchNorm2d(channels, affine=True)
                self.use_dropout = use_dropout
                self.dropout = nn.Dropout(.2)
                self.conv2 = nn.Conv2d(channels, channels, kernel, padding=(padding, padding),
                                       padding_mode=padding_mode)

            def forward(self, input_tensor):
                out = self.conv1(input_tensor)
                out = self.bn(out)
                out = self.lrelu(out)
                if self.use_dropout:
                    out = self.dropout(out)
                out = self.conv2(out)
                out = self.bn(out)
                return out + input_tensor

        def __init__(self, channels, padding_mode, name="Gen"):
            super().__init__()
            self._model_name = name
            self.channels = channels

            # Color Injection
            self.color1 = nn.Sequential(
                PanColorGan.ConvBlock(channels, 32, 3, padding=1, padding_mode=padding_mode),
                PanColorGan.ConvBlock(32, 32, 3, padding=1, padding_mode=padding_mode)
            )
            self.color2 = PanColorGan.ConvBlock(32, 64, 3, stride=2, padding=1, padding_mode=padding_mode)
            self.color3 = PanColorGan.ConvBlock(64, 128, 3, stride=2, padding=1, padding_mode=padding_mode)

            # Spatial Details Extraction
            self.model1 = nn.Sequential(
                PanColorGan.ConvBlock(1, 32, 3, padding=1, padding_mode=padding_mode),
                PanColorGan.ConvBlock(32, 32, 3, padding=1, padding_mode=padding_mode)
            )
            self.model2 = PanColorGan.ConvBlock(64, 64, 3, stride=2, padding=1, padding_mode=padding_mode)
            self.model3 = PanColorGan.ConvBlock(128, 128, 3, stride=2, padding=1, padding_mode=padding_mode)

            # Feature Transformation
            self.feature_transformation = nn.Sequential(
                PanColorGan.Generator.ResnetBlock(256, 3, padding=1, padding_mode=padding_mode),
                PanColorGan.Generator.ResnetBlock(256, 3, padding=1, padding_mode=padding_mode),
                PanColorGan.Generator.ResnetBlock(256, 3, padding=1, padding_mode=padding_mode),
                PanColorGan.Generator.ResnetBlock(256, 3, padding=1, padding_mode=padding_mode),
                PanColorGan.Generator.ResnetBlock(256, 3, padding=1, padding_mode=padding_mode),
                PanColorGan.Generator.ResnetBlock(256, 3, padding=1, padding_mode=padding_mode)
            )

            # Image Synthesis
            self.model4 = PanColorGan.ConvBlock(256, 128, 3, padding=1, padding_mode=padding_mode)
            self.model5 = PanColorGan.Generator.UpConvBlock(256, 128, 3, stride=2, padding=1, out_padding=1)

            self.model6 = PanColorGan.ConvBlock(128, 64, 3, padding=1, padding_mode=padding_mode)
            self.model7 = PanColorGan.Generator.UpConvBlock(128, 64, 3, stride=2, padding=1, out_padding=1)

            self.model8 = PanColorGan.ConvBlock(64, 32, 3, padding=1, padding_mode=padding_mode)
            self.model9 = nn.Sequential(
                PanColorGan.ConvBlock(64, 32, 3, padding=1, padding_mode=padding_mode),
                nn.Conv2d(32, channels, kernel_size=(3, 3), padding=(1, 1), padding_mode=padding_mode)
            )
            self.out_model = nn.Tanh()

        def forward(self, pan, ms):
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

            )
            self.act = nn.Sigmoid()
            self.apply_activation = True

        def forward(self, input_tensor):
            out = self.net(input_tensor)
            if self.apply_activation:
                out = self.act(out)
            return out

    def loss_discriminator(self, ms, pan, gt, generated):
        fake_ab = torch.cat([ms, pan, generated], 1)
        real_ab = torch.cat([ms, pan, gt], 1)

        pred_fake = self.discriminator(fake_ab)
        pred_real = self.discriminator(real_ab)

        return self.adv_loss(pred_fake, pred_real)

    def loss_generator(self, ms, pan, gt, generated):

        fake_ab = torch.cat([ms, pan, generated], 1)
        real_ab = torch.cat([ms, pan, gt], 1)

        pred_fake = self.discriminator(fake_ab)

        if self.adv_loss.use_real_data:
            pred_real = self.discriminator(real_ab)
        else:
            pred_real = None

        loss_adv = self.adv_loss(pred_fake, pred_real, True)
        loss_rec = self.rec_loss(generated, gt)

        a, b = loss_rec
        df = pd.DataFrame(columns=["Epochs", "Value"])
        df.loc[0] = [self.tot_epochs, a.detach().cpu().numpy()]
        df.to_csv(f"{self.output_path}/q_loss.csv", index=False, header=True if self.tot_epochs == 1 else False,
                  mode='a', sep=";")
        df = pd.DataFrame(columns=["Epochs", "Value"])
        df.loc[0] = [self.tot_epochs, b.detach().cpu().numpy()]
        df.to_csv(f"{self.output_path}/mae_loss.csv", index=False, header=True if self.tot_epochs == 1 else False,
                  mode='a', sep=";")

        loss_rec = a + b

        loss_g = loss_adv * self.weight_gan + loss_rec * self.lambda_factor
        return loss_g

    # ------------------------- Concrete Interface Methods -----------------------------
    def train_step(self, dataloader):
        self.train(True)

        loss_g_batch = 0
        loss_d_batch = 0
        for batch, data in enumerate(dataloader):
            pan, ms, ms_lr, gt = data

            # pan = pan.to(self.device)
            # gt = gt.to(self.device)
            # ms = ms.to(self.device)
            gt = ms_lr.to(self.device)

            # Downsample MS_LR
            ms_lr = nn.functional.interpolate(gt, scale_factor=1 / 4, mode='bicubic', align_corners=False)
            # Upsample MS_LR_LR
            ms = nn.functional.interpolate(ms_lr, scale_factor=4, mode='bicubic', align_corners=False)
            # Convert MS_LR to Grayscale
            pan = torch.mean(gt, 1, keepdim=True)

            # Generate Data for Discriminators Training
            with torch.no_grad():
                generated_HRMS = self.generate_output(pan, ms)

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
            generated = self.generate_output(pan, ms, evaluation=False)
            loss_g = self.loss_generator(ms, pan, gt, generated)

            # Backpropagation
            self.gen_opt.zero_grad()
            loss_g.backward()
            self.gen_opt.step()

            loss = loss_g.item()
            torch.cuda.empty_cache()

            loss_g_batch += loss

        try:
            self.rec_loss.reset()
        except:
            pass
        return {"Gen loss": loss_g_batch / len(dataloader),
                "Disc loss": loss_d_batch / len(dataloader)
                }

    def validation_step(self, dataloader, evaluate_indexes=False):
        self.eval()
        self.discriminator.eval()
        self.generator.eval()

        gen_loss = 0.0
        disc_loss = 0.0
        running_q2n = 0.0
        running_q = 0.0
        running_sam = 0.0
        running_ergas = 0.0
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                pan, ms, ms_lr, gt = data

                gt = gt.to(self.device)
                pan = pan.to(self.device)
                ms = ms.to(self.device)
                ms_lr.to(self.device)
                # Downsample MS_LR
                # ms_lr = nn.functional.interpolate(gt, scale_factor=1 / 4, mode='bicubic', align_corners=False)
                # Upsample MS_LR_LR
                # ms = nn.functional.interpolate(ms_lr, scale_factor=4, mode='bicubic', align_corners=False)
                # Convert MS_LR to Grayscale
                # pan = torch.mean(gt, 1, keepdim=True)

                generated = self.generate_output(pan, ms, evaluation=True)
                #dloss = self.loss_discriminator(ms, pan, gt, generated)
                #disc_loss += dloss.item()

                # generated = self.generate_output(ms_lr_gray, ms_lr_up)
                #gloss = self.loss_generator(ms, pan, gt, generated)
                #gen_loss += gloss.item()
                # Compute indexes

                # Downgrade Output and compare with MS_LR
                generated_dg = nn.functional.interpolate(generated, scale_factor=1 / 4, mode='bicubic', align_corners=False)

                if evaluate_indexes:
                    batch_q = batch_q2n = batch_ergas = batch_sam = 0.0
                    voutputs = torch.permute(generated_dg, (0, 2, 3, 1)).detach().cpu().numpy()
                    gt_all = torch.permute(ms_lr, (0, 2, 3, 1)).detach().cpu().numpy()
                    num_elem_batch = voutputs.shape[0]
                    for k in range(num_elem_batch):
                        gt = gt_all[k, :, :, :]
                        gen = voutputs[k, :, :, :]
                        indexes = indexes_evaluation(gt, gen, ratio, L, 8, 0, dim_cut,
                                                     th_values)
                        batch_q2n += indexes[0]
                        batch_q += indexes[1]
                        batch_ergas += indexes[2]
                        batch_sam += indexes[3]
                    running_q += batch_q / num_elem_batch
                    running_q2n += batch_q2n / num_elem_batch
                    running_sam += batch_sam / num_elem_batch
                    running_ergas += batch_ergas / num_elem_batch

            q2n_tot = running_q2n / len(dataloader)
            q_tot = running_q / len(dataloader)
            ergas_tot = running_ergas / len(dataloader)
            sam_tot = running_sam / len(dataloader)

        try:
            self.loss_fn.reset()
        except:
            pass
        return {"Gen loss": gen_loss / len(dataloader),
                "Disc loss": disc_loss / len(dataloader)
                }, [q2n_tot, q_tot, ergas_tot, sam_tot]

    def save_model(self, path):
        torch.save({
            'gen_state_dict': self.generator.state_dict(),
            'disc_state_dict': self.discriminator.state_dict(),
            'gen_optimizer_state_dict': self.gen_opt.state_dict(),
            'disc_optimizer_state_dict': self.disc_opt.state_dict(),
            'gen_best_loss': self.best_losses[0],
            'disc_best_loss': self.best_losses[1],
            'tot_epochs': self.tot_epochs,
            'best_epoch': self.best_epoch,
            'metrics': [self.best_q, self.best_q_avg, self.best_sam, self.best_ergas]
        }, f"{path}")

    def load_model(self, path, weights_only=False):
        trained_model = torch.load(f"{path}", map_location=torch.device(self.device))
        self.generator.load_state_dict(trained_model['gen_state_dict'])
        self.discriminator.load_state_dict(trained_model['disc_state_dict'])
        if weights_only:
            return
        self.gen_opt.load_state_dict(trained_model['gen_optimizer_state_dict'])
        self.disc_opt.load_state_dict(trained_model['disc_optimizer_state_dict'])
        self.tot_epochs = trained_model['tot_epochs']
        self.best_epoch = trained_model['best_epoch']
        self.best_losses = [trained_model['gen_best_loss'], trained_model['disc_best_loss']]
        try:
            self.best_q, self.best_q_avg, self.best_sam, self.best_ergas = trained_model['metrics']
        except KeyError:
            pass

    def set_optimizers_lr(self, lr):
        for g in self.gen_opt.param_groups:
            g['lr'] = lr
        for g in self.disc_opt.param_groups:
            g['lr'] = lr

    def define_losses(self, rec_loss=None, adv_loss=None):
        self.rec_loss = rec_loss if rec_loss is not None else torch.nn.L1Loss(reduction='mean')
        self.adv_loss = adv_loss if adv_loss is not None else RaganLoss()
        self.discriminator.apply_activation = self.adv_loss.apply_activation
