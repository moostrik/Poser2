"""LumaKey: a mask from each pixel's luminance."""

from OpenGL.GL import glUniform1f, glUniform1i, glUseProgram
from ... import Shader, draw_quad, Texture

import logging
logger = logging.getLogger(__name__)


class LumaKey(Shader):
    """`src` with alpha 0 where its Rec. 709 luminance is below `threshold`, 1 above `threshold` + `softness`, smooth
    between; `invert` keys out the light instead; the color kept."""

    def use(self, src: Texture, threshold: float = 0.1, softness: float = 0.1, invert: bool = False) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("LumaKey shader not allocated or shader program missing.")
            return
        if not src.allocated:
            return
        glUseProgram(self.shader_program)
        src.bind(0)
        glUniform1f(self.get_uniform_loc("threshold"), threshold)
        glUniform1f(self.get_uniform_loc("softness"), max(softness, 1e-4))
        glUniform1i(self.get_uniform_loc("invert"), int(invert))
        draw_quad()
