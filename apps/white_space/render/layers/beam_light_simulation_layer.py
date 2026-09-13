"""BeamLightSimulationLayer — the image the installation projects while the fixture is in beam
mode: the four beam lights on the walls, unrolled over 360° like the projection.

The sibling of ``LightSimulationLayer`` (the projection): both read the composition output from
the board and push a Frame-shaped image through the same shader, so the two modes share
one look. This one builds its image from the frame's explicit ``beam_lights``
and its playhead heading (``beam_light_projection``). The render draws whichever of the two
layers matches the fixture's readout mode for the frame.

Recent flashes from the board are drawn over it, fading out over ``flash_seconds``, so a flash that
lasts a tick or two on the fixture stays readable on screen; the steady lines are drawn as they are.
"""

import math
from time import monotonic
from typing import Protocol

import numpy as np
from OpenGL.GL import * # type: ignore

from modules.gl import Fbo, Texture, Image
from modules.board import HasCompositionOutput, HasFlashes
from modules.render.layers.LayerBase import LayerBase
from modules.utils import HotReloadMethods

from apps.white_space.light import BUFFER_DTYPE
from apps.white_space.render.shaders.light_simulation import LightSimulation

from .beam_light_projection import project_beam_lights, paint_flashes
from ...settings import BeamLightSimSettings


class BeamLightSimulationBoard(HasCompositionOutput, HasFlashes, Protocol):
    ...


class BeamLightSimulationLayer(LayerBase):

    def __init__(self, board: BeamLightSimulationBoard, config: BeamLightSimSettings) -> None:
        self.board: BeamLightSimulationBoard = board
        self._config: BeamLightSimSettings = config
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
        width, blur = math.radians(self._config.width), math.radians(self._config.blur)
        project_beam_lights(output.beam_lights, self._heading, width, blur, self._projection)
        paint_flashes(self.board.get_flashes(), monotonic(), self._config.flash_seconds, width, blur,
                      self._projection)

        self.image.set_image(self._projection)
        self.image.update()

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.fbo_angles.begin()
        self._shader.use(self.fbo_angles, self.image.texture, self.fbo_angles.width)
        self.fbo_angles.end()
