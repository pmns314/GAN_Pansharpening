from enum import Enum

from .MinmaxLoss import MinmaxLoss
from .RaganLoss import RaganLoss
from .LsganLoss import LsganLoss


class AdvLosses(Enum):
    """ Adversarial losses implemented in the framework"""
    MINMAX = MinmaxLoss
    RAGAN = RaganLoss
    LSGAN = LsganLoss
