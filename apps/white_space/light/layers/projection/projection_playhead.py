"""Playhead (high) — visualises the content playhead as a bright marker on the pixel ring.

Draws a marker at the strip position of ``frame.playhead`` (the continuous content playhead,
radians [-π, π); NaN → nothing drawn). Distinct from the beam-mode ``BeamPlayhead`` and the motor/content ``Playhead``
(the NCO in ``light/playhead.py``).
"""

import math

import numpy as np

from modules.settings import Field

from .._base_layer import ProjectionLayer, LayerSettings
from .._utilities import angle_to_strip_position
from ...frame import Frame


class ProjectionPlayheadSettings(LayerSettings):
    level: Field[float] = Field(1.0, min=0.0, max=1.0,  step=0.01, description="Marker brightness")
    width: Field[float] = Field(3.6, min=0.1, max=36.0, step=0.1,  description="Marker width (deg)")


class ProjectionPlayhead(ProjectionLayer):
    """A bright marker at the playhead's position on the pixel ring (visualises the content playhead)."""

    def __init__(self, resolution: int, config: ProjectionPlayheadSettings, board) -> None:
        super().__init__(resolution, config, board)
        self._config = config

    def _draw(self, frame: Frame, white: np.ndarray, blue: np.ndarray) -> None:
        ph = frame.playhead
        if math.isnan(ph):
            return
        P = self._config
        center = int(angle_to_strip_position(ph) * self.resolution)   # [0, R)
        w      = max(1, round(P.width / 360.0 * self.resolution))     # deg → pixel count
        start  = center - w // 2
        idx    = np.arange(start, start + w) % self.resolution
        white[idx] += P.level
