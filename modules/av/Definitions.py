from enum import Enum, IntEnum, auto
from typing import Callable
import numpy as np
from dataclasses import dataclass

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
        self.test: np.ndarray = np.zeros((1, resolution, 4), dtype=IMG_TYPE)
        self.angle: float = angle

AvOutputCallback = Callable[[AvOutput], None]


@dataclass
class CompSettings():
    interval: float = 0.1               # seconds between updates
    num_players: int = 1                # number of people to track and display

    smoothness: float = 0.5             # between 0 and 1, higher is smoother
    responsiveness: float = 0.5         # between 0 and 1, higher is more responsive

    void_width: float =  0.05           # in normalized world width (0..1)
    void_edge: float = 0.01             # in normalized world width (0..1)
    use_void: bool = True

    pattern_width: float = 0.2          # in normalized world width (0..1)
    pattern_edge: float = 0.2           # in normalized world width (0..1)

    line_sharpness: float = 1.5         # higher is sharper
    line_speed: float = 1.5             # higher is faster
    line_width: float = 0.1             # in normalized world width (0..1)
    line_amount: float = 20.0            # number of lines
