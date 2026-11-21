"""WhiteNoise: uncorrelated noise per pixel, three channels, new on every draw."""

from random import random

from OpenGL.GL import glUniform1f, glUseProgram
from ... import Shader, draw_quad

import logging
logger = logging.getLogger(__name__)


class WhiteNoise(Shader):
    """A hash of each pixel's position and a random seed per draw, in RGB."""

    def use(self) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("WhiteNoise shader not allocated or shader program missing.")
            return

        glUseProgram(self.shader_program)
        glUniform1f(self.get_uniform_loc("random"), random())
        draw_quad()
