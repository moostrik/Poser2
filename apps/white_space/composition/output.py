import numpy as np
from typing import Callable
from dataclasses import dataclass, field

BUFFER_DTYPE = np.float32


@dataclass
class CompositionOutput:
    """LED strip output — white and blue channels sent to the installation over UDP."""
    resolution:  int
    target_rpm: float = 0.0
    playhead: float = 0.0
    light_img: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        # Shape (1, R, 3): channel 0 = white, channel 1 = blue, channel 2 = reserved
        self.light_img = np.zeros((1, self.resolution, 3), dtype=BUFFER_DTYPE)

    @property
    def white(self) -> np.ndarray:
        """White channel."""
        return self.light_img[0, :, 0]

    @white.setter
    def white(self, value: np.ndarray | float) -> None:
        self.light_img[0, :, 0] = value

    @property
    def blue(self) -> np.ndarray:
        """Blue channel."""
        return self.light_img[0, :, 1]

    @blue.setter
    def blue(self, value: np.ndarray | float) -> None:
        self.light_img[0, :, 1] = value


CompositionOutputCallback = Callable[[CompositionOutput], None]
