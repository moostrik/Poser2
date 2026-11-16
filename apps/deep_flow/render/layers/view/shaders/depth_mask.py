from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class DepthMask(Shader):
    def use(self, depth: Texture, near: float, far: float) -> None:
        """The player cut from `depth` (metres, 0 for none): the band from `near` to `far`."""
        if not self.allocated or not self.shader_program or not depth.allocated:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, depth.tex_id)

        glUniform1i(self.get_uniform_loc("depth"), 0)
        glUniform1f(self.get_uniform_loc("nearDepth"), near)
        glUniform1f(self.get_uniform_loc("farDepth"), far)

        draw_quad()
