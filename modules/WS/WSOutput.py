import numpy as np
from typing import Callable
from dataclasses import dataclass, field


WS_IMG_TYPE = np.float32

@dataclass
class WSOutput:
    resolution: int
    light_img: np.ndarray = field(init=False)
    sound_img: np.ndarray = field(init=False)
    infos_img: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.light_img = np.zeros((1, self.resolution, 3), dtype=WS_IMG_TYPE)
        self.sound_img = np.zeros((1, self.resolution, 3), dtype=WS_IMG_TYPE)
        self.infos_img = np.zeros((1, self.resolution, 4), dtype=WS_IMG_TYPE)

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
