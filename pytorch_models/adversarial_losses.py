import torch.nn as nn
import torch


class Ragan_Loss(nn.Module):
    def __init__(self):
        super(Ragan_Loss, self).__init__()
        self.mse = nn.MSELoss(reduction="mean")
        # Label 1 for fake, Label 0 for real
        self.ones = None
        self.zeros = None

    def forward(self, true, pred, discriminator):
        pred_fake = discriminator(pred)
        pred_real = discriminator(true)

        if self.ones is None:
            self.ones = torch.ones_like(pred_fake)
            self.zeros = torch.zeros_like(pred_fake)

        # L_RaGAN(x1, x2) = loss( D(x1) - mean( D(x2) )
        loss_fake = self.mse(pred_fake - torch.mean(pred_real), self.zeros)
        loss_real = self.mse(pred_real - torch.mean(pred_fake), self.ones)

        return (loss_real + loss_fake) / 2


class LSGAN_loss(nn.Module):
    def __init__(self):
        super(LSGAN_loss, self).__init__()
        self.mse = nn.MSELoss(reduction="mean")
        # Label 1 for fake, Label 0 for real
        self.ones = None
        self.zeros = None

    def forward(self, true, pred, discriminator):
        pred_fake = discriminator(pred)
        pred_real = discriminator(true)

        if self.ones is None:
            self.ones = torch.ones_like(pred_fake)
            self.zeros = torch.zeros_like(pred_fake)

        loss_real = self.mse(pred_real, self.ones)
        loss_fake = self.mse(pred_fake, self.zeros)

        return loss_real + loss_fake


class PSGAN_loss(nn.Module):
    def __init__(self):
        super(PSGAN_loss, self).__init__()
        self.mse = nn.MSELoss(reduction="mean")
        # Label 1 for fake, Label 0 for real
        self.EPS = 1e-12

    def forward(self, true, pred, discriminator):
        pred_fake = discriminator(pred)
        pred_real = discriminator(true)

        return torch.mean(
            -(
                    torch.log(pred_real + self.EPS) + torch.log(1 - pred_fake + self.EPS)
            )
        )
