from enum import Enum, IntEnum, auto
from typing import Callable
import numpy as np
from dataclasses import dataclass, field

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


@dataclass
class WSOutput:
    resolution: int
    angle: float = ANGLE
    light_img: np.ndarray = field(init=False)
    sound_img: np.ndarray = field(init=False)
    infos_img: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.light_img = np.zeros((1, self.resolution, 3), dtype=IMG_TYPE)
        self.sound_img = np.zeros((1, self.resolution, 3), dtype=IMG_TYPE)
        self.infos_img = np.zeros((1, self.resolution, 4), dtype=IMG_TYPE)

    @property
    def light_0(self) -> np.ndarray:
        return self.light_img[0, :, 0]
    @light_0.setter
    def light_0(self, value: np.ndarray | float) -> None:
        self.light_img[0, :, 0] = value

    @property
    def light_1(self) -> np.ndarray:
        return self.light_img[0, :, 1]
    @light_1.setter
    def light_1(self, value: np.ndarray | float) -> None:
        self.light_img[0, :, 1] = value

    @property
    def light_2(self) -> np.ndarray:
        return self.light_img[0, :, 2]
    @light_2.setter
    def light_2(self, value: np.ndarray | float) -> None:
        self.light_img[0, :, 2] = value

    @property
    def infos_0(self) -> np.ndarray:
        return self.infos_img[0, :, 0]
    @infos_0.setter
    def infos_0(self, value: np.ndarray | float) -> None:
        self.infos_img[0, :, 0] = value

    @property
    def infos_1(self) -> np.ndarray:
        return self.infos_img[0, :, 1]
    @infos_1.setter
    def infos_1(self, value: np.ndarray | float) -> None:
        self.infos_img[0, :, 1] = value

    @property
    def infos_2(self) -> np.ndarray:
        return self.infos_img[0, :, 2]
    @infos_2.setter
    def infos_2(self, value: np.ndarray | float) -> None:
        self.infos_img[0, :, 2] = value

    @property
    def infos_3(self) -> np.ndarray:
        return self.infos_img[0, :, 3]
    @infos_3.setter
    def infos_3(self, value: np.ndarray | float) -> None:
        self.infos_img[0, :, 3] = value

WSOutputCallback = Callable[[WSOutput], None]


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
