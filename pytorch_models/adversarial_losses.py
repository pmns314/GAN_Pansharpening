import torch.nn as nn
import torch


class Ragan_Loss(nn.Module):
    def __init__(self):
        super(Ragan_Loss, self).__init__()
        self.mse = nn.MSELoss(reduction="mean")
        # Label 1 for fake, Label 0 for real
        self.ones = None
        self.zeros = None

    def forward(self, true, pred):
        if self.ones is None:
            self.ones = torch.ones_like(true)
            self.zeros = torch.zeros_like(true)

        # L_RaGAN(x1, x2) = loss( D(x1) - mean( D(x2) )
        loss_fake = self.mse(pred - torch.mean(true), self.zeros)
        loss_real = self.mse(true - torch.mean(pred), self.ones)

        return (loss_real + loss_fake) / 2


class LSGAN_loss(nn.Module):
    def __init__(self):
        super(LSGAN_loss, self).__init__()
        self.mse = nn.MSELoss(reduction="mean")
        # Label 1 for fake, Label 0 for real
        self.ones = None
        self.zeros = None

    def forward(self, true, pred):
        if self.ones is None:
            self.ones = torch.ones_like(true)
            self.zeros = torch.zeros_like(pred)

        loss_real = self.mse(true, self.ones)
        loss_fake = self.mse(pred, self.zeros)

        return loss_real + loss_fake


class PSGAN_loss(nn.Module):
    def __init__(self):
        super(PSGAN_loss, self).__init__()
        self.mse = nn.MSELoss(reduction="mean")
        # Label 1 for fake, Label 0 for real
        self.EPS = 1e-12

    def forward(self, true, pred):
        return torch.mean(
            -(
                    torch.log(true + self.EPS) + torch.log(1 - pred + self.EPS)
            )
        )
