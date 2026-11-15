"""Mix: two images mixed linearly, GLSL's `mix`."""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUseProgram
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
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, src.tex_id)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, dst.tex_id)
        glUniform1i(self.get_uniform_loc("src"), 0)
        glUniform1i(self.get_uniform_loc("dst"), 1)
        glUniform1f(self.get_uniform_loc("amount"), amount)
        draw_quad()
