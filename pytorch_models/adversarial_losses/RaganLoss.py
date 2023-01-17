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

    def ragan_loss(self, x1, x2):

        # L_RaGAN(x1, x2) = loss( D(x1) - mean( D(x2) )
        loss_fake = self.mse(x1 - torch.mean(x2), self.ones * self.fake_label)
        # D : Fake - mean(real) , Fake_label
        loss_real = self.mse(x2 - torch.mean(x1), self.ones * self.real_label)

        return (loss_real + loss_fake) / 2

    def forward(self, fake, real, is_generator=False):
        if self.ones is None or self.ones.shape != fake.shape:
            self.ones = torch.ones_like(fake)

        if is_generator:
            return self.ragan_loss(fake, real)
        else:
            return self.ragan_loss(real, fake)
