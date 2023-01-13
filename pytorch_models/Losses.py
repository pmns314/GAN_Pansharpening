from enum import Enum
import torch
from pytorch_msssim import ssim, ms_ssim


class CharbonnierLoss(torch.nn.Module):
    def __init__(self):
        super(CharbonnierLoss, self).__init__()

    def forward(self, y_true, y_pred):
        epsilon = 1e-6
        x = y_true - y_pred

        # loss = sqrt(x**2 + eps**2)
        loss = torch.sqrt(torch.square(x) + epsilon)
        # Mean over batch
        loss = torch.mean(torch.mean(loss, [1, 2, 3]))
        return loss


class FrobeniusLoss(torch.nn.Module):
    def __init__(self):
        super(FrobeniusLoss, self).__init__()

    def forward(self, y_true, y_pred):
        tensor = y_pred - y_true
        norm = torch.norm(tensor, p="fro")
        return torch.mean(torch.square(norm))


class SSIMLoss(torch.nn.Module):
    def __init__(self, Qblocks_size=32):
        super(SSIMLoss, self).__init__()
        self.Qblocks_size = Qblocks_size

    def forward(self, y_true, y_pred):
        y_pred = y_pred * 2048.0
        y_true = y_true * 2048.0

        return -ssim(y_pred, y_true)


class MSSSIMLoss(torch.nn.Module):
    def __init__(self, Qblocks_size=32):
        super(MSSSIMLoss, self).__init__()
        self.Qblocks_size = Qblocks_size

    def forward(self, y_true, y_pred):
        y_pred = y_pred * 2048.0
        y_true = y_true * 2048.0

        return -ms_ssim(y_pred, y_true)


class Losses(Enum):
    MAE = torch.nn.L1Loss
    MSE = torch.nn.MSELoss
    CHARBONNIER = CharbonnierLoss
    FROBENIUS = FrobeniusLoss
    SSIM = SSIMLoss
    MSSSIM = MSSSIMLoss
