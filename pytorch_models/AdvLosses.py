from enum import Enum

from pytorch_models.adversarial_losses import *


class AdvLosses(Enum):
    MINMAX = MinmaxLoss
    RAGAN = RaganLoss
    LSGAN = LsganLoss
