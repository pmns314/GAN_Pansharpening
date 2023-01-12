from enum import Enum
import torch


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


class Losses(Enum):
    MAE = torch.nn.L1Loss
    MSE = torch.nn.MSELoss
    CHARBONNIER = CharbonnierLoss
    FROBENIUS = FrobeniusLoss
