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


class AvOutput():
    def __init__(self, resolution: int, angle: float = ANGLE) -> None:
        self.resolution: int = resolution
        self.img: np.ndarray = np.zeros((1, resolution, 3), dtype=np.float16)
        self.angle: float = angle

AvOutputCallback = Callable[[AvOutput], None]