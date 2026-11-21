"""Larger Amount shader: per pixel, the one of two textures with the larger amount."""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE2, GL_TEXTURE3, GL_TEXTURE_2D, glActiveTexture, glBindTexture, \
    glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class LargerAmount(Shader):
    """Writes, per pixel, `held` or `source`, whichever has the larger amount, as it is: the size of the red of
    `held_amount` and `source_amount`, read by uv, so the amounts may have another size."""

    def use(self, held: Texture, source: Texture, held_amount: Texture, source_amount: Texture) -> None:
        if not self.allocated or not self.shader_program:
            return
        if not all(texture.allocated for texture in (held, source, held_amount, source_amount)):
            return

        glUseProgram(self.shader_program)
        for unit, texture in enumerate((held, source, held_amount, source_amount)):
            glActiveTexture((GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE2, GL_TEXTURE3)[unit])
            glBindTexture(GL_TEXTURE_2D, texture.tex_id)

        draw_quad()
