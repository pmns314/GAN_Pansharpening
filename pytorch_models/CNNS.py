from enum import Enum
from pytorch_models.CNNs import *


class CNNS(Enum):
    """ Cnn networks implemented in the framework"""
    APNN = APNN
    BDPN = BDPN
    DICNN = DiCNN
    DRPNN = DRPNN
    FUSIONNET = FusionNet
    MSDCNN = MSDCNN
    PANNET = PanNet
    PNN = PNN
