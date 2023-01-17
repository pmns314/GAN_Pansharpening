from enum import Enum

from .PSGAN import PSGAN
from .FUPSGAN import FUPSGAN
from .STPSGAN import STPSGAN
from .PanGan import PanGan
from .PanColorGan import PanColorGan



class GANS(Enum):
    """ GAN networks implemented in the framework"""
    PSGAN = PSGAN
    FUPSGAN = FUPSGAN
    STPSGAN = STPSGAN
    PANGAN = PanGan
    PANCOLORGAN = PanColorGan

