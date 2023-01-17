from enum import Enum

from .MinmaxLoss import MinmaxLoss
from .RaganLoss import RaganLoss
from .LsganLoss import LsganLoss




class AdvLosses(Enum):
    MINMAX = MinmaxLoss
    RAGAN = RaganLoss
    LSGAN = LsganLoss
