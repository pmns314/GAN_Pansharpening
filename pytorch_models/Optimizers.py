""" Optimizers Module """

from enum import Enum
import torch


class Optimizers(Enum):
    ADAM = torch.optim.Adam
    SGD = torch.optim.SGD
    RMSPROP = torch.optim.RMSprop
    ADADELTA = torch.optim.Adadelta
    ADAGRAD = torch.optim.Adagrad