from enum import Enum
from pytorch_models.GANs import *


class GANS(Enum):
    """ GAN networks implemented in the framework"""
    PSGAN = PSGAN
    FUPSGAN = FUPSGAN
    STPSGAN = STPSGAN
    PANGAN = PanGan
    PANCOLORGAN = PanColorGan
