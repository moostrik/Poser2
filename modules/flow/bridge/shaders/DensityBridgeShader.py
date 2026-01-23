"""Density Bridge shader.

Combines RGB density with velocity magnitude to create alpha-driven density.
Ported from ofxFlowTools ftDensityBridgeShader.h
"""

from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture


class DensityBridgeShader(Shader):
    """Combines density RGB with velocity magnitude for alpha channel."""

    def use(self, density: Texture, velocity: Texture, speed: float) -> None:
        """Combine density with velocity magnitude.

        Args:
            density: RGB density input texture
            velocity: RG velocity input texture
            speed: Speed multiplier for alpha calculation
        """
        if not self.allocated or not self.shader_program:
            return
        if not density.allocated or not velocity.allocated:
            return

        glUseProgram(self.shader_program)

        # Bind textures
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, density.tex_id)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, velocity.tex_id)

        # Set uniforms (using cached locations)
        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform1i(self.get_uniform_loc("tex1"), 1)
        glUniform1f(self.get_uniform_loc("speed"), speed)

        draw_quad()

        # Cleanup
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glUseProgram(0)
