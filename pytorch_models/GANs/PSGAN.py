import os
import shutil

import numpy as np
import torch
from torch import nn, optim
from torch.nn import LeakyReLU
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset.DatasetPytorch import DatasetPytorch

EPS = 1e-12
TO_SAVE = [1, 2, 3, 5, 8,
           10, 20, 30, 50, 80,
           100, 200, 300, 500, 800,
           1000, 2000, 3000, 5000, 8000, 10000]


class PSGAN(nn.Module):
    def __init__(self, channels, name="PSGAN"):
        super(PSGAN, self).__init__()
        self._model_name = name
        self.channels = channels
        self.alpha = 1
        self.beta = 100
        self.generator = self.Generator(channels)
        self.discriminator = self.Discriminator(channels)

        self.gen_opt = None
        self.disc_opt = None

    @property
    def name(self):
        return self._model_name

    class Generator(nn.Module):
        def __init__(self, channels, name="Gen"):
            super().__init__()
            self._model_name = name
            self.channels = channels

            self.pan_enc_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding='same', bias=True)
            self.pan_enc_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding='same', bias=True)
            self.pan_enc_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(2, 2),
                                       padding=(1, 1), bias=True)

            self.ms_enc_1 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=(3, 3), padding='same', bias=True)
            self.ms_enc_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding='same', bias=True)
            self.ms_enc_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(2, 2),
                                      padding=(1, 1), bias=True)

            self.enc = nn.Sequential(
                LeakyReLU(negative_slope=.2),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding='same', bias=True),
                LeakyReLU(negative_slope=.2),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding='same', bias=True)
            )

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

            self.common = nn.Sequential(
                LeakyReLU(negative_slope=.2),
                nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), padding=(1, 1), bias=True),
                LeakyReLU(negative_slope=.2),
                nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(2, 2), stride=(2, 2),
                                   padding=(0, 0), bias=True)
            )
            # ?
            self.final_part = nn.Sequential(
                LeakyReLU(negative_slope=.2),
                nn.Conv2d(in_channels=192, out_channels=128, kernel_size=(3, 3), padding='same',
                          bias=True),
                LeakyReLU(negative_slope=.2),
                nn.Conv2d(in_channels=128, out_channels=channels, kernel_size=(3, 3), padding='same',
                          bias=True)
            )

            self.relu = nn.ReLU(inplace=True)
            self.lrelu = LeakyReLU(negative_slope=.2)

        def forward(self, ms, pan):
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
        def __init__(self, channels, name="Disc"):
            super().__init__()
            self._model_name = name
            self.channels = channels

            self.backbone = nn.Sequential(
                nn.Conv2d(in_channels=channels * 2, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=(2, 2),
                          bias=True),
                LeakyReLU(negative_slope=.2),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(2, 2),
                          bias=True),
                LeakyReLU(negative_slope=.2),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=(2, 2),
                          bias=True),
                LeakyReLU(negative_slope=.2),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2),
                          bias=True),
                LeakyReLU(negative_slope=.2),
            )

            self.out_conv = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=(3, 3), stride=(1, 1),
                                      padding='same',
                                      bias=True)
            self.sigmoid = nn.Sigmoid()

        def forward(self, generated, target):
            inputs = torch.cat([generated, target], 1)
            out = self.backbone(inputs)
            out = self.out_conv(out)
            out = self.sigmoid(out)
            return out

    def set_optimizers(self, lr):
        self.gen_opt = optim.Adam(self.generator.parameters(), lr=lr)
        self.disc_opt = optim.Adam(self.discriminator.parameters(), lr=lr)

    def loss_generator(self, ms, pan, gt, *args):

        outputs = self.generator(ms, pan)
        predict_fake = self.discriminator(ms, outputs)
        # From Code
        # gen_loss_GAN = tf.reduce_mean(-tf.math.log(predict_fake + EPS))
        # gen_loss_L1 = tf.reduce_mean(tf.math.abs(gt - outputs))
        # gen_loss = gen_loss_GAN * self.alpha + gen_loss_L1 * self.beta

        # From Formula
        gen_loss_GAN = torch.mean(torch.log(predict_fake + EPS))
        gen_loss_L1 = torch.mean(torch.abs(gt - outputs))
        gen_loss = - self.alpha * gen_loss_GAN + self.beta * gen_loss_L1

        return gen_loss

    def loss_discriminator(self, ms, gt, output):

        predict_fake = self.discriminator(ms, output)
        predict_real = self.discriminator(ms, gt)

        # From Formula
        # mean[ 1 - log(fake) + log(real) ]
        return torch.mean(
            1 - torch.log(predict_fake + EPS) + torch.log(predict_real + EPS)
        )

        # From Code
        # return tf.reduce_mean(
        #     -(
        #             tf.math.log(predict_real + EPS) + tf.math.log(1 - predict_fake + EPS)
        #     )
        # )

    def train_loop(self, dataloader, device='cpu'):
        size = len(dataloader.dataset)
        batch_size = dataloader.batch_size
        self.train(True)

        loss_g_batch = 0
        loss_d_batch = 0
        for batch, data in enumerate(dataloader):
            pan, ms, ms_lr, gt = data

            gt = gt.to(device)
            pan = pan.to(device)
            ms = ms.to(device)

            # Generate Data for Discriminators Training
            generated_HRMS = self.generator(ms, pan)

            # ------------------- Training Discriminator ----------------------------
            self.discriminator.train(True)
            self.generator.train(False)

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

            # Compute prediction and loss
            loss_g = self.loss_generator(ms, pan, gt)

            # Backpropagation
            self.gen_opt.zero_grad()
            loss_g.backward()
            self.gen_opt.step()

            loss = loss_g.item()
            torch.cuda.empty_cache()

            loss_g_batch += loss

        return loss_d_batch / len(dataloader), loss_g_batch / len(dataloader)

    def validation_loop(self, dataloader, device='cpu'):
        self.eval()
        self.discriminator.eval()
        self.generator.eval()

        gen_loss = 0.0
        disc_loss = 0.0
        i = 0
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                pan, ms, ms_lr, gt = data

                gt = gt.to(device)
                pan = pan.to(device)
                ms = ms.to(device)

                generated = self.generator(ms, pan)

                dloss = self.loss_discriminator(ms, gt, generated)
                disc_loss += dloss.item()

                gloss = self.loss_generator(ms, pan, gt)
                gen_loss += gloss.item()

        return disc_loss / len(dataloader), gen_loss / len(dataloader)

    def test_loop(self, dataloader, device='cpu'):
        self.eval()
        self.discriminator.eval()
        self.generator.eval()

        gen_loss = 0.0
        disc_loss = 0.0
        i = 0
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                pan, ms, ms_lr, gt = data

                gt = gt.to(device)
                pan = pan.to(device)
                ms = ms.to(device)

                generated = self(pan, ms)

                dloss = self.loss_discriminator(ms, gt, generated)
                disc_loss += dloss.item()

                gloss = self.loss_generator(ms, pan, gt)
                gen_loss += gloss.item()

        print(f"Evaluation on Test Set: \n "
              f"\t Avg disc loss: {disc_loss / len(dataloader):>8f} \n"
              f"\t Avg gen loss: {gen_loss / len(dataloader):>8f} \n")

    def my_training(self, epochs,
                    best_vloss_d, best_vloss_g,
                    output_path, chk_path,
                    train_dataloader, val_dataloader,
                    pretrained_epochs=0, device='cpu'):
        # TensorBoard
        writer = SummaryWriter(output_path + "log/")
        # Early stopping
        patience = 30
        triggertimes = 0

        # Reduce Learning Rate on Plateaux
        # scheduler_d = ReduceLROnPlateau(self.disc_opt, 'min', patience=10, verbose=True)
        # scheduler_g = ReduceLROnPlateau(self.gen_opt, 'min', patience=10, verbose=True)

        pretrained_epochs = pretrained_epochs + 1
        epoch = 0

        print(f"Training started for {output_path} at epoch {pretrained_epochs}")
        for epoch in range(epochs):
            disc_loss, gen_loss = self.train_loop(train_dataloader, device)
            if val_dataloader is not None:
                curr_loss_d, curr_loss_g = self.validation_loop(val_dataloader, device)
                print(f'Epoch {pretrained_epochs + epoch}\t'
                      f'\t Disc: train {disc_loss :.2f}\t valid {curr_loss_d:.2f}\n'
                      f'\t Gen: train {gen_loss :.2f}\t valid {curr_loss_g:.2f}\n')
                writer.add_scalars("Disc_Loss", {"train": disc_loss, "validation": curr_loss_d},
                                   pretrained_epochs + epoch)
                writer.add_scalars("Gen_Loss", {"train": gen_loss, "validation": curr_loss_g},
                                   pretrained_epochs + epoch)
            else:
                print(f'Epoch {pretrained_epochs + epoch}\t'
                      f'\t Disc: {disc_loss :.2f}'
                      f'\t Gen: {gen_loss :.2f}')
                curr_loss_d = disc_loss
                curr_loss_g = gen_loss
                writer.add_scalar("Disc_loss/Train", disc_loss, pretrained_epochs + epoch)
                writer.add_scalar("Gen_loss/Train", gen_loss, pretrained_epochs + epoch)

            # Save Checkpoints
            if pretrained_epochs + epoch in TO_SAVE:
                torch.save({'gen_state_dict': self.generator.state_dict(),
                            'disc_state_dict': self.discriminator.state_dict(),
                            'gen_optimizer_state_dict': self.gen_opt.state_dict(),
                            'disc_optimizer_state_dict': self.disc_opt.state_dict(),
                            'gen_best_loss': best_vloss_g,
                            'disc_best_loss': best_vloss_d
                            }, f"{chk_path}/checkpoint_{pretrained_epochs + epoch}.pth")

            if curr_loss_g < best_vloss_g:
                best_vloss_g = curr_loss_g
                torch.save({
                    'best_epoch': pretrained_epochs + epoch,
                    'gen_state_dict': self.generator.state_dict(),
                    'disc_state_dict': self.discriminator.state_dict(),
                    'gen_optimizer_state_dict': self.gen_opt.state_dict(),
                    'disc_optimizer_state_dict': self.disc_opt.state_dict(),
                    'gen_best_loss': best_vloss_g,
                    'disc_best_loss': best_vloss_d
                }, output_path + "/model.pth")
                triggertimes = 0
            else:
                triggertimes += 1
                if triggertimes >= patience:
                    print("Early Stopping!")
                    break

            if curr_loss_d < best_vloss_d:
                best_vloss_d = curr_loss_d

            # scheduler_d.step(best_vloss_d)
            # scheduler_g.step(best_vloss_g)

        m = torch.load(output_path + "/model.pth")
        m['tot_epochs'] = pretrained_epochs + epoch
        torch.save(m, output_path + "/model.pth")
        writer.flush()
        print(f"Training Completed at epoch {pretrained_epochs + epoch}. Saved in {output_path} folder")


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    satellite = "W3"
    output_path = 'pytorch/'
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    train_data = DatasetPytorch("../../datasets/" + satellite + "/train.h5")
    train_dataloader = DataLoader(train_data, batch_size=64,
                                  shuffle=True)
    val_dataloader = DataLoader(DatasetPytorch("../../datasets/" + satellite + "/val.h5"), batch_size=64,
                                shuffle=True)
    test_dataloader = DataLoader(DatasetPytorch("../../datasets/" + satellite + "/test.h5"), batch_size=64,
                                 shuffle=True)

    model = PSGAN(train_data.channels)

    from torchsummary import summary

    g = model.generator

    model.to(device)

    model.my_training(output_path, 500)
    model.test_loop(test_dataloader)
