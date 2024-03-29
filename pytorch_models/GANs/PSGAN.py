from abc import ABC

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.nn import LeakyReLU

from pytorch_models.GANs.GanInterface import GanInterface
from pytorch_models.adversarial_losses import MinmaxLoss


class PSGAN(GanInterface, ABC):
    """ PSGAN Implementation"""

    def __init__(self, channels, device='cpu', name="PSGAN", pad_mode="replicate"):
        """ Constructor of the class

        Parameters
        ----------
        channels : int
            number of channels accepted as input
        device : str, optional
            the device onto which train the network (either cpu or a cuda visible device).
            Default is 'cpu'
        name : str, optional
            the name of the network. Default is 'PSGAN'
        pad_mode : str, optional
            padding mode. Default to "replicate"
        """
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
        """ PSGAN Generator"""

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
            self.ms_enc_1 = nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=(3, 3), padding='same',
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
        """ PSGAN Discriminator"""

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
            self.apply_activation = True

        def forward(self, inputs):
            out = self.backbone(inputs)
            out = self.out_conv(out)
            if self.apply_activation:
                out = self.sigmoid(out)
            return out

    def loss_generator(self, ms, gt, generated):
        """ Calculate loss of generator
        Parameters
        ----------
            ms : torch.Tensor
                multi spectral image
            gt : torch.Tensor
                target image
            generated : torch.Tensor
                fused image
        """

        pred_fake = self.discriminator(torch.cat([ms, generated], 1))

        if self.adv_loss.use_real_data:
            pred_true = self.discriminator(torch.cat([ms, gt], 1).detach())
        else:
            pred_true = None

        # From Formula
        gen_loss_GAN = self.adv_loss(pred_fake, pred_true, True)  # Inganna il discriminatore
        gen_loss_L1 = self.rec_loss(gt, generated)
        a, b = gen_loss_L1
        df = pd.DataFrame(columns=["Epochs", "Value"])
        df.loc[0] = [self.tot_epochs, a.detach().cpu().numpy()]
        df.to_csv(f"{self.output_path}/q_loss.csv", index=False, header=True if self.tot_epochs == 1 else False,
                  mode='a', sep=";")
        df = pd.DataFrame(columns=["Epochs", "Value"])
        df.loc[0] = [self.tot_epochs, b.detach().cpu().numpy()]
        df.to_csv(f"{self.output_path}/mae_loss.csv", index=False, header=True if self.tot_epochs == 1 else False,
                  mode='a', sep=";")

        gen_loss_L1 = a + b

        gen_loss = self.alpha * gen_loss_GAN + self.beta * gen_loss_L1

        return gen_loss

    def loss_discriminator(self, ms, gt, generated):
        """ Calculate loss of discriminator

        Parameters
        ----------
            ms : torch.Tensor
                multi spectral image
            gt : torch.Tensor
                target image
            generated : torch.Tensor
                fused image
        """
        pred_fake = self.discriminator(torch.cat([ms, generated], 1).detach())
        pred_real = self.discriminator(torch.cat([ms, gt], 1))

        return self.adv_loss(pred_fake, pred_real)

    # -------------------------------- Interface Methods ------------------------------
    def train_step(self, dataloader):
        """ Defines the operations to be carried out during the training step

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            the dataloader that loads the training data
        """
        self.train(True)

        loss_g_batch = 0
        loss_d_batch = 0
        for batch, data in enumerate(dataloader):
            pan, ms, ms_lr, gt = data

            gt = gt.to(self.device)
            pan = pan.to(self.device)
            ms = ms.to(self.device)
            if self.use_ms_lr is False:
                multi_spectral = ms
            else:
                multi_spectral = ms_lr.to(self.device)

            if len(pan.shape) != len(ms.shape):
                pan = torch.unsqueeze(pan, 0)

            # Generate Data for Discriminators Training
            with torch.no_grad():
                generated_HRMS = self.generate_output(pan, multi_spectral)

            # ------------------- Training Discriminator ----------------------------
            self.discriminator.train(True)
            self.generator.train(False)
            self.discriminator.zero_grad()
            self.disc_opt.zero_grad()

            # Compute prediction and loss
            loss_d = self.loss_discriminator(ms, gt, generated_HRMS)

            # Backpropagation
            loss_d.backward()
            self.disc_opt.step()

            loss = loss_d.item()
            torch.cuda.empty_cache()

            loss_d_batch += loss

            # ------------------- Training Generator ----------------------------
            self.discriminator.train(False)
            self.generator.train(True)
            self.generator.zero_grad()
            self.gen_opt.zero_grad()

            # Compute prediction and loss
            generated = self.generate_output(pan, multi_spectral, evaluation=False)
            loss_g = self.loss_generator(ms, gt, generated)

            # Backpropagation
            loss_g.backward()
            self.gen_opt.step()

            loss = loss_g.item()
            torch.cuda.empty_cache()

            loss_g_batch += loss

        self.rec_loss.reset()

        return {"Gen loss": loss_g_batch / len(dataloader),
                "Disc loss": loss_d_batch / len(dataloader)
                }

    def validation_step(self, dataloader):
        """ Defines the operations to be carried out during the validation step

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            the dataloader that loads the validation data
        """
        self.eval()
        self.discriminator.eval()
        self.generator.eval()

        gen_loss = 0.0
        disc_loss = 0.0

        with torch.no_grad():
            for i, data in enumerate(dataloader):
                pan, ms, ms_lr, gt = data

                gt = gt.to(self.device)
                pan = pan.to(self.device)
                ms = ms.to(self.device)

                if self.use_ms_lr is False:
                    multi_spectral = ms
                else:
                    multi_spectral = ms_lr.to(self.device)

                if len(pan.shape) == 3:
                    pan = torch.unsqueeze(pan, 0)

                generated = self.generate_output(pan, multi_spectral)

                dloss = self.loss_discriminator(ms, gt, generated)
                disc_loss += dloss.item()

                gloss = self.loss_generator(ms, gt, generated)
                gen_loss += gloss.item()

            self.loss_fn.reset()

        return {"Gen loss": gen_loss / len(dataloader),
                "Disc loss": disc_loss / len(dataloader)
                }

    def save_model(self, path):
        """ Saves the model as a .pth file

        Parameters
        ----------

        path : str
            the path where the model has to be saved into
        """
        torch.save({
            'gen_state_dict': self.generator.state_dict(),
            'disc_state_dict': self.discriminator.state_dict(),
            'gen_optimizer_state_dict': self.gen_opt.state_dict(),
            'disc_optimizer_state_dict': self.disc_opt.state_dict(),
            'gen_best_loss': self.best_losses[0],
            'disc_best_loss': self.best_losses[1],
            'best_epoch': self.best_epoch,
            'tot_epochs': self.tot_epochs,
            'metrics': [self.best_q, self.best_q_avg, self.best_sam, self.best_ergas]
        }, f"{path}")

    def load_model(self, path, weights_only=False):
        """ Loads the network model

        Parameters
        ----------

        path : str
            the path of the model
        weights_only : bool, optional
            True if only the weights of the generator must be loaded, False otherwise (default is False)

        """
        trained_model = torch.load(f"{path}", map_location=torch.device(self.device))
        self.generator.load_state_dict(trained_model['gen_state_dict'])
        self.discriminator.load_state_dict(trained_model['disc_state_dict'])
        if weights_only:
            return
        self.gen_opt.load_state_dict(trained_model['gen_optimizer_state_dict'])
        self.disc_opt.load_state_dict(trained_model['disc_optimizer_state_dict'])
        self.best_losses = [trained_model['gen_best_loss'], trained_model['disc_best_loss']]
        self.best_epoch = trained_model['best_epoch']
        self.tot_epochs = trained_model['tot_epochs']
        try:
            self.best_q, self.best_q_avg, self.best_sam, self.best_ergas = trained_model['metrics']
        except KeyError:
            pass

    def set_optimizers_lr(self, lr):
        """ Sets the learning rate of the optimizers

        Parameter
        ---------
        lr : int
            the new learning rate of the optimizers
        """
        for g in self.gen_opt.param_groups:
            g['lr'] = lr
        for g in self.disc_opt.param_groups:
            g['lr'] = lr

    def define_losses(self, rec_loss=None, adv_loss=None):
        """ Set adversarial and reconstruction losses """
        self.rec_loss = rec_loss if rec_loss is not None else torch.nn.L1Loss(reduction='mean')
        self.adv_loss = adv_loss if adv_loss is not None else MinmaxLoss()
        self.discriminator.apply_activation = self.adv_loss.apply_activation
