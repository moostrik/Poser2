"""Playhead (high) — visualises the content playhead as a bright marker on the pixel ring.

Draws a marker at the strip position of ``frame.playhead`` (the continuous content playhead,
radians [-π, π); NaN → nothing drawn). Imported as ``PlayheadHigh`` to distinguish it from the
low-speed ``PlayheadLow`` and the motor/content ``Playhead`` (the NCO in ``light/playhead.py``).
"""

import math

import numpy as np

from modules.settings import Field

from .._base_layer import BaseLayer, LayerSettings
from .._utilities import angle_to_strip_position
from ...frame import Frame


class PlayheadSettings(LayerSettings):
    level: Field[float] = Field(1.0,  min=0.0, max=1.0,  step=0.01,  description="Marker brightness")
    width: Field[float] = Field(0.01, min=0.0, max=0.5,  step=0.005, description="Marker width (fraction of the ring)")


class Playhead(BaseLayer):
    """A bright marker at the playhead's position on the pixel ring (visualises the content playhead)."""

    def __init__(self, resolution: int, config: PlayheadSettings, board) -> None:
        super().__init__(resolution, config, board)
        self._config = config

    def _draw(self, frame: Frame, white: np.ndarray, blue: np.ndarray) -> None:
        ph = frame.playhead
        if math.isnan(ph):
            return
        P = self._config
        center = int(angle_to_strip_position(ph) * self.resolution)   # [0, R)
        half   = max(1, int(P.width * self.resolution / 2))
        idx    = np.arange(center - half, center + half + 1) % self.resolution
        white[idx] += P.level
