"""Playhead (low) — slow-speed light that lights the first white pixel (the FRONT white lamps).

See ``layers/low/__init__.py`` for the full slow-speed pixel→lamp mapping (front/back white,
left/right blue). Distinct from the high-regime ``PlayheadHigh`` and the motor/content ``Playhead``
(the NCO in ``light/playhead.py``).
"""

import numpy as np

from modules.settings import Field

from .._base_layer import LowLayer, LayerSettings
from ...frame import Frame


class PlayheadLowSettings(LayerSettings):
    level: Field[float] = Field(1.0, min=0.0, max=1.0, step=0.01, description="Front white lamp brightness")


class PlayheadLow(LowLayer):
    """Slow-speed light. Lights the first white pixel (the front white lamps); see the module
    docstring for the full slow-speed pixel→lamp mapping (front/back white, left/right blue)."""

    def __init__(self, resolution: int, config: PlayheadLowSettings, board) -> None:
        super().__init__(resolution, config, board)
        self._config = config

    def _draw(self, frame: Frame, white: np.ndarray, blue: np.ndarray) -> None:
        self._add_lamps(white, blue, front_white=self._config.level)
