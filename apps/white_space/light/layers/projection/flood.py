"""Flood — the whole projection constant white: the END's wall of light.

Deliberately the dumbest layer in the pool: its one dynamic, the cross-to-full, is a mix
weight set by the states, never behavior inside the layer. The wall's ending belongs to
``beam_wind_down``. Stateless.
"""

import numpy as np

from modules.settings import Field

from .._base_layer import ProjectionLayer, LayerSettings
from ...frame import Frame


class FloodSettings(LayerSettings):
    level: Field[float] = Field(1.0, min=0.0, max=1.0, step=0.01, description="White level over the whole projection")


class Flood(ProjectionLayer):
    """``white[:] += level``; see the module docstring."""

    def __init__(self, resolution: int, config: FloodSettings, board) -> None:
        super().__init__(resolution, config, board)
        self._config = config

    def _draw(self, frame: Frame, white: np.ndarray, blue: np.ndarray) -> None:
        white += self._config.level
