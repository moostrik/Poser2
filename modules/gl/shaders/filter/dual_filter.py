"""DualFilter: the dual-filter blur's passes (Bjørge 2015): half size down, double size up, five and eight taps."""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUseProgram
from ... import Shader, draw_quad, Texture

import logging
logger = logging.getLogger(__name__)


class DualFilter(Shader):
    """One pass of the dual filter into a target half (`down`) or double (`up`) `src`'s size; a wide, cheap blur over
    a chain of them. `down` with a `threshold` keeps only what is brighter, the bloom's bright pass."""

    def down(self, src: Texture, threshold: float = -1.0) -> None:
        """`src` at half size: its center and four diagonal neighbors; above `threshold` alone when 0 or more."""
        self._use(src, False, threshold)

    def up(self, src: Texture) -> None:
        """`src` at double size: eight neighbors around the center, the diagonals at half weight."""
        self._use(src, True, -1.0)

    def _use(self, src: Texture, upsample: bool, threshold: float) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("DualFilter shader not allocated or shader program missing.")
            return
        if not src.allocated:
            return
        glUseProgram(self.shader_program)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, src.tex_id)
        glUniform1i(self.get_uniform_loc("upsample"), int(upsample))
        glUniform1f(self.get_uniform_loc("threshold"), threshold)
        draw_quad()
