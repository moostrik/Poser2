"""BarLightSimulationLayer — the image the installation projects while the fixture is in slot
mode: the four bar lights as beams on the walls, unrolled over 360° like the ring.

The sibling of ``LightSimulationLayer`` (the ring): both read the composition output from
the board and push a Frame-shaped image through the same shader, so the two regimes share
one look. This one builds its image from the frame's explicit ``bar_lights``
and its playhead heading (``bar_light_projection``). The render draws whichever of the two
layers matches the fixture's readout mode for the frame.
"""

import math

import numpy as np
from OpenGL.GL import * # type: ignore

from modules.gl import Fbo, Texture, Image
from modules.board import HasCompositionOutput
from modules.render.layers.LayerBase import LayerBase
from modules.utils import HotReloadMethods

from apps.white_space.light import BUFFER_DTYPE
from apps.white_space.render.shaders.light_simulation import LightSimulation

from .bar_light_projection import project_bar_lights
from ...settings import BarLightSimSettings


class BarLightSimulationLayer(LayerBase):

    def __init__(self, board: HasCompositionOutput, config: BarLightSimSettings) -> None:
        self.board: HasCompositionOutput = board
        self._config: BarLightSimSettings = config
        self.fbo_angles: Fbo = Fbo()
        self.image: Image = Image()
        self._shader: LightSimulation = LightSimulation()
        self._projection: np.ndarray | None = None   # the Frame-shaped (1, R, 3) image, render-thread local
        self._heading: float = 0.0                   # last finite playhead — the bar stays where it stopped

        self.hot_reloader = HotReloadMethods(self.__class__, True, True)

    @property
    def texture(self) -> Texture:
        return self.fbo_angles

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self.fbo_angles.allocate(width, height, internal_format)
        self._shader.allocate()

    def deallocate(self) -> None:
        self.fbo_angles.deallocate()
        self.image.deallocate()
        self._shader.deallocate()

    def update(self) -> None:
        output = self.board.get_composition_output()
        if output is None:
            return

        if not math.isnan(output.playhead):
            self._heading = output.playhead

        if self._projection is None or self._projection.shape != output.light_img.shape:
            self._projection = np.zeros(output.light_img.shape, dtype=BUFFER_DTYPE)
        project_bar_lights(output.bar_lights, self._heading, math.radians(self._config.beam),
                           math.radians(self._config.blur), self._projection)

        self.image.set_image(self._projection)
        self.image.update()

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.fbo_angles.begin()
        self._shader.use(self.fbo_angles, self.image.texture, self.fbo_angles.width)
        self.fbo_angles.end()
