from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture


class CelShade(Shader):
    """Cel shading with integrated color correction."""

    def use(self, tex: Texture,
            exposure: float = 1.0, gamma: float = 1.0, offset: float = 0.0,
            contrast: float = 1.0,
            levels: int = 4, smoothness: float = 0.1,
            saturation: float = 1.0) -> None:
        if not self.allocated or not self.shader_program:
            print("CelShade shader not allocated or shader program missing.")
            return
        if not tex.allocated:
            print("CelShade shader: input texture not allocated.")
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex.tex_id)

        glUniform1i(self.get_uniform_loc("tex"), 0)
        glUniform1f(self.get_uniform_loc("exposure"), exposure)
        glUniform1f(self.get_uniform_loc("gamma"), gamma)
        glUniform1f(self.get_uniform_loc("offset"), offset)
        glUniform1f(self.get_uniform_loc("contrast"), contrast)
        glUniform1i(self.get_uniform_loc("levels"), levels)
        glUniform1f(self.get_uniform_loc("smoothness"), smoothness)
        glUniform1f(self.get_uniform_loc("saturation"), saturation)

        draw_quad()
