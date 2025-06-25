from enum import Enum, IntEnum, auto
from typing import Callable
import numpy as np

class CompMode(IntEnum):
    NONE = auto()
    TEST = auto()
    CALIBRATE = auto()
    VISUALS = auto()

ANGLE =         3.0
RESOLUTION =    4000
RATE =          30

# IMG_TYPE =np.uint8
# IMG_MP = 255.0
IMG_TYPE = np.float32
IMG_MP = 1.0


class AvOutput():
    def __init__(self, resolution: int, angle: float = ANGLE) -> None:
        self.resolution: int = resolution
        self.img: np.ndarray = np.zeros((1, resolution, 3), dtype=IMG_TYPE)
        self.angle: float = angle

AvOutputCallback = Callable[[AvOutput], None]