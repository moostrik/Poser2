from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture


class HueShift(Shader):
    """Shift image hue toward a target color while preserving luminance."""

    def use(self, tex: Texture, 
            target_r: float = 1.0, target_g: float = 0.0, target_b: float = 0.0,
            strength: float = 0.5) -> None:
        if not self.allocated or not self.shader_program:
            print("HueShift shader not allocated or shader program missing.")
            return
        if not tex.allocated:
            print("HueShift shader: input texture not allocated.")
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex.tex_id)

        glUniform1i(self.get_uniform_loc("tex"), 0)
        glUniform3f(self.get_uniform_loc("targetColor"), target_r, target_g, target_b)
        glUniform1f(self.get_uniform_loc("strength"), strength)

        draw_quad()
