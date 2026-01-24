from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture


class Tint(Shader):
    """Draw a texture multiplied by a tint color. Modern GL replacement for glColor4f()."""

    def use(self, tex: Texture, r: float = 1.0, g: float = 1.0, b: float = 1.0, a: float = 1.0) -> None:
        if not self.allocated or not self.shader_program:
            print("Tint shader not allocated or shader program missing.")
            return
        if not tex.allocated:
            print("Tint shader: input texture not allocated.")
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex.tex_id)

        glUniform1i(self.get_uniform_loc("tex"), 0)
        glUniform4f(self.get_uniform_loc("tint"), r, g, b, a)

        draw_quad()

        glBindTexture(GL_TEXTURE_2D, 0)
        glUseProgram(0)
