from enum import Enum

from .APNN import APNN
from .BDPN import BDPN
from .DRPNN import DRPNN
from .DiCNN import DiCNN
from .FusionNet import FusionNet
from .MSDCNN import MSDCNN
from .PNN import PNN
from .PanNet import PanNet


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
