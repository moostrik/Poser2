"""Channel Copy shader.

Copies a single-channel texture to a specific channel of an RGBA destination.
"""

from OpenGL.GL import glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture

import logging
logger = logging.getLogger(__name__)


class ChannelCopy(Shader):
    """Copy single-channel texture to specific RGBA channel."""

    def use(self, dst: Texture, src: Texture, channel: int) -> None:
        """Copy source R channel to destination's specified channel.

        Args:
            dst: Destination RGBA texture (provides channels to preserve)
            src: Source single-channel texture (R32F)
            channel: Target channel index (0=R, 1=G, 2=B, 3=A)
        """
        if not self.allocated or not self.shader_program:
            logger.warning("ChannelCopy shader not allocated or shader program missing.")
            return
        if not dst.allocated or not src.allocated:
            logger.warning("ChannelCopy shader: input texture not allocated.")
            return

        glUseProgram(self.shader_program)

        # Bind destination (existing RGBA)
        dst.bind(0)

        # Bind source (single-channel)
        src.bind(1)

        # Set channel uniform
        glUniform1i(self.get_uniform_loc("channel"), channel)

        draw_quad()
