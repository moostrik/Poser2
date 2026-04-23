import numpy as np
from typing import Callable
from dataclasses import dataclass, field

COMP_DTYPE = np.float32


@dataclass
class CompositionOutput:
    """LED strip output — white and blue channels sent to the installation over UDP."""
    resolution: int
    light_img: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        # Shape (1, R, 3): channel 0 = white, channel 1 = blue, channel 2 = reserved
        self.light_img = np.zeros((1, self.resolution, 3), dtype=COMP_DTYPE)

    @property
    def light_0(self) -> np.ndarray:
        """White channel."""
        return self.light_img[0, :, 0]

    @light_0.setter
    def light_0(self, value: np.ndarray | float) -> None:
        self.light_img[0, :, 0] = value

    @property
    def light_1(self) -> np.ndarray:
        """Blue channel."""
        return self.light_img[0, :, 1]

    @light_1.setter
    def light_1(self, value: np.ndarray | float) -> None:
        self.light_img[0, :, 1] = value


@dataclass
class CompositionDebug:
    """Intermediate composition channels for on-screen visualisation only — never sent over UDP.

    Channel layout (matches WS_Lines.frag RGBA access):
      0 (.r) = white_l  — left-side wave pattern (rendered as orange bars)
      1 (.g) = white_r  — right-side wave pattern (rendered as cyan bars)
      2 (.b) = blue     — blue wave pattern (rendered as blue bars)
      3 (.a) = void     — void field (rendered as grey overlay)
    """
    resolution: int
    debug_img: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.debug_img = np.zeros((1, self.resolution, 4), dtype=COMP_DTYPE)


CompositionOutputCallback = Callable[[CompositionOutput], None]
