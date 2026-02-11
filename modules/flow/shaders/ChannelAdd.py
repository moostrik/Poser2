"""Channel Add shader.

Adds a single-channel texture to a specific channel of an RGBA destination.
"""

from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture


class ChannelAdd(Shader):
    """Add single-channel texture to specific RGBA channel."""

    def use(self, dst: Texture, src: Texture, channel: int, strength: float = 1.0) -> None:
        """Add source R channel to destination's specified channel.

        Args:
            dst: Destination RGBA texture (provides channels to preserve)
            src: Source single-channel texture (R32F)
            channel: Target channel index (0=R, 1=G, 2=B, 3=A)
            strength: Multiplier for the addition (default 1.0)
        """
        if not self.allocated or not self.shader_program:
            print("ChannelAdd shader not allocated or shader program missing.")
            return
        if not dst.allocated or not src.allocated:
            print("ChannelAdd shader: input texture not allocated.")
            return

        glUseProgram(self.shader_program)

        # Bind destination (existing RGBA)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, dst.tex_id)
        glUniform1i(self.get_uniform_loc("dst"), 0)

        # Bind source (single-channel)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, src.tex_id)
        glUniform1i(self.get_uniform_loc("src"), 1)

        # Set channel uniform
        glUniform1i(self.get_uniform_loc("channel"), channel)

        # Set strength uniform
        glUniform1f(self.get_uniform_loc("strength"), strength)

        draw_quad()
