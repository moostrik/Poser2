"""Playhead (low) — slow-speed light that lights the first white pixel (the FRONT white lamps).

See ``layers/low/__init__.py`` for the full slow-speed pixel→lamp mapping (front/back white,
left/right blue). Imported as ``PlayheadLow`` to distinguish it from the high-speed ``PlayheadHigh``
and the motor/content ``Playhead`` (the NCO in ``light/playhead.py``).
"""

import numpy as np

from modules.settings import Field

from .._base_layer import BaseLayer, LayerSettings
from ...frame import Frame


class PlayheadSettings(LayerSettings):
    level: Field[float] = Field(1.0, min=0.0, max=1.0, step=0.01, description="Front white lamp brightness")


class Playhead(BaseLayer):
    """Slow-speed light. Lights the first white pixel (the front white lamps); see the module
    docstring for the full slow-speed pixel→lamp mapping (front/back white, left/right blue)."""

    def __init__(self, resolution: int, config: PlayheadSettings, board) -> None:
        super().__init__(resolution, config, board)
        self._config = config

    def _draw(self, frame: Frame, white: np.ndarray, blue: np.ndarray) -> None:
        # First white pixel = front white lamps. (Back = white[R//2], blue L/R = blue[0]/blue[R//2].)
        white[0] += self._config.level
