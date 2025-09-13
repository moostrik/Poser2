from enum import IntEnum
from typing import Callable
import numpy as np
from dataclasses import dataclass, field

from modules.WS.WSDraw import WSDrawSettings
from modules.WS.WSDataManager import WSDataSettings

class CompMode(IntEnum):
    NONE = 0
    TEST = 1
    CALIBRATE = 2
    VISUALS = 3

@dataclass
class WSSettings:

    draw_settings: WSDrawSettings
    data_settings: WSDataSettings

    mode: CompMode = CompMode.VISUALS
