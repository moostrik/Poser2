"""Velocity Change Region Speed shader.

Writes the speeds of the two trails, for a fall's region means.
"""

from OpenGL.GL import glUseProgram
from modules.gl import Shader, draw_quad, Texture


class VelocityBandpassRegionSpeed(Shader):
    """Writes (fast speed, slow speed) to RG."""

    def use(self, fast: Texture, slow: Texture) -> None:
        """Write the speeds.

        Args:
            fast: RGB fast trail
            slow: RGB slow trail
        """
        if not self.allocated or not self.shader_program:
            return
        if not fast.allocated or not slow.allocated:
            return

        glUseProgram(self.shader_program)

        fast.bind(0)
        slow.bind(1)


        draw_quad()
