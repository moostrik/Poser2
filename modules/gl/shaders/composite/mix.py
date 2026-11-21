"""Mix: two images mixed linearly, GLSL's `mix`."""

from OpenGL.GL import glUniform1f, glUseProgram
from ... import Shader, draw_quad, Texture

import logging
logger = logging.getLogger(__name__)


class Mix(Shader):
    """`src` at `amount` 0, `dst` at 1, linear between."""

    def use(self, src: Texture, dst: Texture, amount: float) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("Mix shader not allocated or shader program missing.")
            return
        if not src.allocated or not dst.allocated:
            logger.warning("Mix shader: input textures not allocated.")
            return

        glUseProgram(self.shader_program)
        src.bind(0)
        dst.bind(1)
        glUniform1f(self.get_uniform_loc("amount"), amount)
        draw_quad()
