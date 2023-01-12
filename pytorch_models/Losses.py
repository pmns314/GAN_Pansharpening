from enum import Enum
import torch

from quality_indexes_toolbox.q2n import q2n


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


class Q2nLoss(torch.nn.Module):
    def __init__(self, Qblocks_size=32):
        super(Q2nLoss, self).__init__()
        self.Qblocks_size = Qblocks_size

    def forward(self, y_true, y_pred):
        Q2n_index = q2n(y_true, y_pred, self.Qblocks_size, self.Qblocks_size)
        return -Q2n_index


class Losses(Enum):
    MAE = torch.nn.L1Loss
    MSE = torch.nn.MSELoss
    CHARBONNIER = CharbonnierLoss
    FROBENIUS = FrobeniusLoss
    Q2NLOSS = Q2nLoss
