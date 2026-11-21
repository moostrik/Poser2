"""ChromaKey: a mask from how far each pixel's chroma is from a key color's."""

from OpenGL.GL import glUniform1f, glUniform3f, glUseProgram
from modules.utils import Color
from ... import Shader, draw_quad, Texture

import logging
logger = logging.getLogger(__name__)


class ChromaKey(Shader):
    """`src` with alpha 0 where its CbCr (BT.709) lies within `threshold` of `key`'s, 1 beyond `threshold` +
    `softness`, smooth between; the color kept."""

    def use(self, src: Texture, key: Color, threshold: float = 0.1, softness: float = 0.1) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("ChromaKey shader not allocated or shader program missing.")
            return
        if not src.allocated:
            return
        glUseProgram(self.shader_program)
        src.bind(0)
        glUniform3f(self.get_uniform_loc("key"), key.r, key.g, key.b)
        glUniform1f(self.get_uniform_loc("threshold"), threshold)
        glUniform1f(self.get_uniform_loc("softness"), max(softness, 1e-4))
        draw_quad()
