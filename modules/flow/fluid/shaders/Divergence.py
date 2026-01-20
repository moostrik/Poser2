"""Divergence shader - Compute divergence of velocity field."""

from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture


class Divergence(Shader):
    """Compute velocity field divergence."""

    def __init__(self) -> None:
        super().__init__()

    def use(self, velocity: Texture, obstacle: Texture, obstacle_offset: Texture, grid_scale: float) -> None:
        """Compute divergence.

        Args:
            velocity: Velocity field (RG32F)
            obstacle: Obstacle mask (R8/R32F)
            obstacle_offset: Neighbor obstacle info (RGBA8)
            grid_scale: Grid scaling factor (typically 1)
        """
        if not self.allocated or not self.shader_program:
            print("Divergence shader not allocated or shader program missing.")
            return
        if not velocity.allocated or not obstacle.allocated or not obstacle_offset.allocated:
            print("Divergence shader: input texture(s) not allocated.")
            return

        glUseProgram(self.shader_program)

        # Bind textures
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, velocity.tex_id)
        glUniform1i(glGetUniformLocation(self.shader_program, "uVelocity"), 0)

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, obstacle.tex_id)
        glUniform1i(glGetUniformLocation(self.shader_program, "uObstacle"), 1)

        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, obstacle_offset.tex_id)
        glUniform1i(glGetUniformLocation(self.shader_program, "uObstacleOffset"), 2)

        # Set uniforms
        glUniform1f(glGetUniformLocation(self.shader_program, "uHalfRdx"), 0.5 / grid_scale)

        draw_quad()

        # Cleanup
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glUseProgram(0)
