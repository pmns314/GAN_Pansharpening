from enum import Enum

import torch
from torchmetrics import UniversalImageQualityIndex as q
from torchmetrics.functional import error_relative_global_dimensionless_synthesis as ergas
from torchmetrics.functional import spectral_angle_mapper as sam
from torchmetrics.functional import structural_similarity_index_measure as ssim


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
    def __init__(self):
        super(SSIMLoss, self).__init__()

    def forward(self, y_true, y_pred):
        y_pred = y_pred * 2048.0
        y_true = y_true * 2048.0

        return 1 - ssim(y_pred, y_true)


class SAMLoss(torch.nn.Module):
    def __init__(self):
        super(SAMLoss, self).__init__()

    def forward(self, y_true, y_pred):
        y_pred = y_pred * 2048.0
        y_true = y_true * 2048.0

        return sam(y_pred, y_true)


class ERGASLoss(torch.nn.Module):
    def __init__(self):
        super(ERGASLoss, self).__init__()

    def forward(self, y_true, y_pred):
        y_pred = y_pred * 2048.0
        y_true = y_true * 2048.0

        return ergas(y_pred, y_true)


class QLoss(torch.nn.Module):
    def __init__(self):
        super(QLoss, self).__init__()
        self.q = q()
        self.mae = torch.nn.L1Loss()

    def reset(self):
        self.q.reset()

    def forward(self, y_true, y_pred):
        y_pred = y_pred * 2048.0
        y_true = y_true * 2048.0

        return (1 - self.q(y_pred, y_true)) + 10 * self.mae(y_pred, y_true)


class Losses(Enum):
    MAE = torch.nn.L1Loss
    MSE = torch.nn.MSELoss
    CHARBONNIER = CharbonnierLoss
    FROBENIUS = FrobeniusLoss
    SSIM = SSIMLoss
    SAM = SAMLoss
    ERGAS = ERGASLoss
    Q = QLoss
