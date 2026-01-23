"""Temperature Bridge Shader.

Combines RGB color interpretation with velocity magnitude mask.
R=warm, G=neutral dampening, B=cold
"""

from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture


class TemperatureBridge(Shader):
    """Temperature bridge: interprets RGB as warm/neutral/cold × mask."""

    def use(self, color: Texture, mask: Texture, scale: float) -> None:
        """Combine color interpretation with velocity magnitude mask.

        Args:
            color: RGB color (R=warm, G=neutral, B=cold)
            mask: Velocity magnitude mask (R32F)
            scale: Output multiplier
        """
        if not self.allocated or not self.shader_program:
            print("TemperatureBridge shader not allocated or shader program missing.")
            return
        if not color.allocated or not mask.allocated:
            print("TemperatureBridge shader: input texture(s) not allocated.")
            return

        glUseProgram(self.shader_program)

        # Bind textures
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, color.tex_id)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, mask.tex_id)

        glUniform1i(self.get_uniform_loc("uColor"), 0)
        glUniform1i(self.get_uniform_loc("uMask"), 1)
        glUniform1f(self.get_uniform_loc("uScale"), scale)

        # Draw
        draw_quad()

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glUseProgram(0)