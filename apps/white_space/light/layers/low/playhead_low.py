"""Playhead (low) — the slow-speed searchlight: the FRONT white bar light at a constant level.

Distinct from the high-regime ``PlayheadHigh`` and the motor/content ``Playhead`` (the NCO in
``light/playhead.py``).
"""

import numpy as np

from modules.settings import Field

from .._base_layer import LowLayer, LayerSettings
from ...frame import Frame


class PlayheadLowSettings(LayerSettings):
    level: Field[float] = Field(1.0, min=0.0, max=1.0, step=0.01, description="Front white lamp brightness")


class PlayheadLow(LowLayer):
    """Slow-speed searchlight: the front white bar light at ``level``."""

    def __init__(self, resolution: int, config: PlayheadLowSettings, board) -> None:
        super().__init__(resolution, config, board)
        self._config = config

    def _draw(self, frame: Frame, bar_lights: np.ndarray) -> None:
        self._add_bar_lights(bar_lights, front_white=self._config.level)
