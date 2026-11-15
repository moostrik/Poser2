"""AddMix: the foreground added to the base, then mixed toward the foreground, both by one blend."""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class AddMix(Shader):
    """`foreground` added to `base` at `blend` to the power 0.3, the sum mixed toward `foreground` at `blend` to the
    power 1.5: the add comes early, the foreground late; the base's alpha is kept."""

    def use(self, base: Texture, foreground: Texture, blend: float) -> None:
        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, base.tex_id)
        glUniform1i(self.get_uniform_loc("dst"), 0)

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, foreground.tex_id)
        glUniform1i(self.get_uniform_loc("src"), 1)

        glUniform1f(self.get_uniform_loc("blend"), blend)

        draw_quad()
