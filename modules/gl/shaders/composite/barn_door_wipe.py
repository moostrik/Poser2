"""BarnDoorWipe: a wipe from `src` to `dst` that opens from the middle."""

from OpenGL.GL import glUniform1f, glUseProgram
from ... import Shader, draw_quad, Texture

import logging
logger = logging.getLogger(__name__)


class BarnDoorWipe(Shader):
    """`dst` in a vertical band from the middle, `src` outside it; the band is the whole width at `fade` 1."""

    def use(self, src: Texture, dst: Texture, fade: float) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("BarnDoorWipe shader not allocated or shader program missing.")
            return
        if not src.allocated or not dst.allocated:
            logger.warning("BarnDoorWipe shader: input textures not allocated.")
            return

        glUseProgram(self.shader_program)

        src.bind(0)
        dst.bind(1)

        glUniform1f(self.get_uniform_loc("fade"), fade)

        draw_quad()
