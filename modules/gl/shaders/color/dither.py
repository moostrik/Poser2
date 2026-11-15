"""Dither: a color rounded to an output's bit depth with a threshold map, so smooth gradients do not band."""

from enum import IntEnum

from OpenGL.GL import GL_NEAREST, GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUseProgram
from ... import Shader, draw_quad, Texture
from ...blue_noise import blue_noise
from ...image import Image

import logging
logger = logging.getLogger(__name__)


class DitherPattern(IntEnum):
    """The threshold map."""
    ORDERED    = 0      # an 8 x 8 Bayer matrix
    BLUE_NOISE = 1      # a 64 x 64 void-and-cluster map (`blue_noise`)


class Dither(Shader):
    """`src` rounded to `bits` per channel, each pixel's threshold from `pattern`, tiled over the target."""

    def __init__(self) -> None:
        super().__init__()
        self._blue_noise: Image = Image(interpolation=GL_NEAREST)

    def allocate(self) -> None:
        super().allocate()
        self._blue_noise.set_image(blue_noise(64))
        self._blue_noise.update()

    def deallocate(self) -> None:
        self._blue_noise.deallocate()
        super().deallocate()

    def use(self, src: Texture, bits: int = 8, pattern: DitherPattern = DitherPattern.BLUE_NOISE) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("Dither shader not allocated or shader program missing.")
            return
        if not src.allocated or not self._blue_noise.allocated:
            return
        glUseProgram(self.shader_program)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, src.tex_id)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self._blue_noise.tex_id)
        glUniform1i(self.get_uniform_loc("src"), 0)
        glUniform1i(self.get_uniform_loc("blueNoise"), 1)
        glUniform1f(self.get_uniform_loc("levels"), float((1 << bits) - 1))
        glUniform1i(self.get_uniform_loc("pattern"), pattern)
        draw_quad()
