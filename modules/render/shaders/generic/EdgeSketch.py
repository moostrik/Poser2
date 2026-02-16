from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture


class EdgeSketch(Shader):
    """Sobel edge detection shader for sketch/contour effect."""

    def use(self, tex: Texture, threshold: float = 0.1, strength: float = 1.0, invert: bool = False) -> None:
        if not self.allocated or not self.shader_program:
            print("EdgeSketch shader not allocated or shader program missing.")
            return
        if not tex.allocated:
            print("EdgeSketch shader: input texture not allocated.")
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex.tex_id)

        glUniform1i(self.get_uniform_loc("tex"), 0)
        glUniform1f(self.get_uniform_loc("threshold"), threshold)
        glUniform1f(self.get_uniform_loc("strength"), strength)
        glUniform1i(self.get_uniform_loc("invert"), 1 if invert else 0)
        glUniform2f(self.get_uniform_loc("texelSize"), 1.0 / tex.width, 1.0 / tex.height)

        draw_quad()
