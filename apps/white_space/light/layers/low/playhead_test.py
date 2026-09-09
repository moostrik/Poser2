"""PlayheadTest — the lamp regime's direct test tool: four settings levels drive the four
bar lights (front/back white, left/right blue) by name. Stateless; a debug layer
(``playhead_test``), never in a state's mix — and the immediate hardware check for the lamp
wiring itself.
"""

import numpy as np

from modules.settings import Field

from .._base_layer import LowLayer, LayerSettings
from ...frame import Frame


class PlayheadTestSettings(LayerSettings):
    front_white: Field[float] = Field(0.0, min=0.0, max=1.0, step=0.01, description="Front white lamp level")
    back_white:  Field[float] = Field(0.0, min=0.0, max=1.0, step=0.01, description="Back white lamp level")
    left_blue:   Field[float] = Field(0.0, min=0.0, max=1.0, step=0.01, description="Left blue lamp level")
    right_blue:  Field[float] = Field(0.0, min=0.0, max=1.0, step=0.01, description="Right blue lamp level")


class PlayheadTest(LowLayer):
    """Direct levels for the four bar lights; see the module docstring."""

    def __init__(self, resolution: int, config: PlayheadTestSettings, board) -> None:
        super().__init__(resolution, config, board)
        self._config = config

    def _draw(self, frame: Frame, bar_lights: np.ndarray) -> None:
        P = self._config
        self._add_bar_lights(bar_lights, front_white=P.front_white, back_white=P.back_white,
                             left_blue=P.left_blue, right_blue=P.right_blue)
