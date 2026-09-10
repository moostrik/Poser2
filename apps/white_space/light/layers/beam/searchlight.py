"""Playhead (low) — the beam-mode searchlight: the FRONT white beam light at a constant level.

Distinct from the projection-mode ``ProjectionPlayhead`` and the motor/content ``Playhead`` (the NCO in
``light/playhead.py``).
"""

import numpy as np

from modules.settings import Field

from .._base_layer import BeamLayer, LayerSettings
from ...frame import Frame


class SearchlightSettings(LayerSettings):
    level: Field[float] = Field(1.0, min=0.0, max=1.0, step=0.01, description="Front white lamp brightness")


class Searchlight(BeamLayer):
    """Slow-speed searchlight: the front white beam light at ``level``."""

    def __init__(self, resolution: int, config: SearchlightSettings, board) -> None:
        super().__init__(resolution, config, board)
        self._config = config

    def _draw(self, frame: Frame, beam_lights: np.ndarray) -> None:
        self._add_beam_lights(beam_lights, front_white=self._config.level)
