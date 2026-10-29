"""BeamTest — beam mode's direct test tool: four settings levels drive the four
beam lights (front/back white, left/right blue) by name. Stateless; a debug layer
(``beam_test``), never in a state's mix — and the immediate hardware check for the lamp
wiring itself.
"""

import numpy as np

from modules.settings import Field

from .._base_layer import BeamLayer, LayerSettings
from ...frame import LightFrame


class BeamTestSettings(LayerSettings):
    front_white: Field[float] = Field(0.0, min=0.0, max=1.0, step=0.01, description="Front white lamp level")
    back_white:  Field[float] = Field(0.0, min=0.0, max=1.0, step=0.01, description="Back white lamp level")
    left_blue:   Field[float] = Field(0.0, min=0.0, max=1.0, step=0.01, description="Left blue lamp level")
    right_blue:  Field[float] = Field(0.0, min=0.0, max=1.0, step=0.01, description="Right blue lamp level")


class BeamTest(BeamLayer):
    """Direct levels for the four beam lights; see the module docstring."""

    def __init__(self, resolution: int, settings: BeamTestSettings, board) -> None:
        super().__init__(resolution, settings, board)
        self._settings: BeamTestSettings = settings

    def _draw(self, frame: LightFrame, beam_lights: np.ndarray) -> None:
        P = self._settings
        self._add_beam_lights(beam_lights, front_white=P.front_white, back_white=P.back_white,
                             left_blue=P.left_blue, right_blue=P.right_blue)
