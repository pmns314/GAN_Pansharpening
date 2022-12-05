from enum import Enum
from pytorch_models.GANs import *


class GANS(Enum):
    PSGAN = PSGAN
    FUPSGAN = FUPSGAN
    STPSGAN = STPSGAN
    PANGAN = PanGan
    PANCOLORGAN = PanColorGan
