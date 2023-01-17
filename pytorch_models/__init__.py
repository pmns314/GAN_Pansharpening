from enum import Enum

import torch

from pytorch_models.CNNs import CNNS
from pytorch_models.GANs import GANS
from pytorch_models.Losses import Losses
from pytorch_models.adversarial_losses import AdvLosses


class Optimizers(Enum):
    ADAM = torch.optim.Adam
    SGD = torch.optim.SGD
    RMSPROP = torch.optim.RMSprop
    ADADELTA = torch.optim.Adadelta
    ADAGRAD = torch.optim.Adagrad
