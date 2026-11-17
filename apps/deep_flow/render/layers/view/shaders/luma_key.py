"""LumaKey: an image's darkness made transparency, its alpha from its brightness."""

import logging

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUseProgram

from modules.gl import Shader, draw_quad, Texture

logger = logging.getLogger(__name__)


class LumaKey(Shader):
    """`src` with its alpha times 1 − `key` + `key` × its brightest channel; RGB kept."""

    def use(self, src: Texture, key: float) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("LumaKey shader not allocated or shader program missing.")
            return
        if not src.allocated:
            logger.warning("LumaKey shader: input texture not allocated.")
            return
        glUseProgram(self.shader_program)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, src.tex_id)
        glUniform1i(self.get_uniform_loc("src"), 0)
        glUniform1f(self.get_uniform_loc("key"), key)
        draw_quad()
