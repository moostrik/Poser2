from enum import Enum
from typing import Callable

import numpy as np


class GenState(Enum):
    NONE        = 0
    IDLE        = 1     # generation finished (image: depth of the last frame, the bloom mask) or failed
    CAM_DEPTH   = 2     # depth of the capture, starts the flash out
    PROMPT      = 3
    CN_DEPTH    = 4
    CN_POSE     = 5
    CN_DETAIL   = 6
    DIFFUSION   = 7     # step previews, then the finished portrait (final=True)
    DIFF_LOOP   = 8     # a grin frame as it is made (for saving; the render plays them from the GrinPlayer)
    DIF_DEPTH   = 9


class GenMessage():
    def __init__(self, state: GenState, progress: float = 0, image: np.ndarray | None = None, prompt: str = '', negative: str = '', final: bool = False) -> None:
        self.state: GenState = state
        self.progress: float = progress
        self.image: np.ndarray | None = image
        self.prompt: str = prompt
        self.negative: str = negative
        self.final: bool = final


MessageCallback = Callable[[GenMessage], None]
