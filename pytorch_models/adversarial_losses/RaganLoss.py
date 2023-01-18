import pandas as pd
import torch
import torch.nn as nn


class RaganLoss(nn.Module):
    def __init__(self):
        super(RaganLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="mean")
        self.ones = None
        self.use_real_data = True
        self.fake_label = 1  # Label for Fake Data
        self.real_label = 0  # Label for Real Data
        self.head = True

    def ragan_loss(self, x1, x2, log=False):

        # L_RaGAN(x1, x2) = loss( D(x1) - mean( D(x2) )
        x2_mean = torch.mean(x2)
        x1_mean = torch.mean(x1)
        loss_fake = self.mse(x1 - x2_mean, self.ones * self.fake_label)
        loss_real = self.mse(x2 - x1_mean, self.ones * self.real_label)
        tot_loss = (loss_real + loss_fake) / 2

        if log:
            df = pd.DataFrame(columns=["loss_fake", "loss_real", "fake_mean", "real_mean", "tot"])
            df.loc[0] = [
                loss_fake.cpu().detach().numpy(),
                loss_real.cpu().detach().numpy(),
                x2_mean.cpu().detach().numpy(),
                x1_mean.cpu().detach().numpy(),
                tot_loss.cpu().detach().numpy(),
                         ]
            df.to_csv("loss_ragan.csv", index=False, header=self.head,
                      mode='a', sep=";")
            self.head = False
        return tot_loss

    def forward(self, fake, real, is_generator=False):
        if self.ones is None or self.ones.shape != fake.shape:
            self.ones = torch.ones_like(fake)

        if is_generator:
            return self.ragan_loss(fake, real)
        else:
            return self.ragan_loss(real, fake, True)
