"""Channel Add Region shader.

Adds source texture to a specific channel at a specific region.
Source R channel is added to the target channel.
"""

from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture


class ChannelAddRegion(Shader):
    """Add source to a specific channel at a specific region."""

    def use(self, dst: Texture, src: Texture, channel: int,
            x: float, y: float, w: float, h: float,
            strength: float = 1.0) -> None:
        """Render with source added to target channel in region.

        Args:
            dst: Destination texture (passed through, channel modified in region)
            src: Source texture (R channel used)
            channel: Target channel (0=R, 1=G, 2=B, 3=A)
            x: Region X position in UV coordinates [0, 1]
            y: Region Y position in UV coordinates [0, 1]
            w: Region width in UV coordinates
            h: Region height in UV coordinates
            strength: Multiplier for the addition
        """
        if not self.allocated or not self.shader_program:
            print("ChannelAddRegion shader not allocated or shader program missing.")
            return
        if not dst.allocated or not src.allocated:
            print("ChannelAddRegion shader: input textures not allocated.")
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, dst.tex_id)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, src.tex_id)

        glUniform1i(self.get_uniform_loc("dst"), 0)
        glUniform1i(self.get_uniform_loc("src"), 1)
        glUniform1i(self.get_uniform_loc("channel"), channel)
        glUniform1f(self.get_uniform_loc("strength"), strength)
        glUniform4f(self.get_uniform_loc("region"), x, y, w, h)

        draw_quad()
