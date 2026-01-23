"""Gradient shader - Subtract pressure gradient from velocity."""

from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture


class Gradient(Shader):
    """Subtract pressure gradient from velocity (projection step)."""

    def __init__(self) -> None:
        super().__init__()

    def use(self, velocity: Texture, pressure: Texture, obstacle: Texture, obstacle_offset: Texture, grid_scale: float, aspect: float) -> None:
        """Apply pressure gradient subtraction.

        Args:
            velocity: Current velocity field (RG32F)
            pressure: Pressure field (R32F)
            obstacle: Obstacle mask (R8/R32F)
            obstacle_offset: Neighbor obstacle info (RGBA8)
            grid_scale: Grid scaling factor (typically simulation_scale)
            aspect: Aspect ratio (width/height) for isotropic derivatives
        """
        if not self.allocated or not self.shader_program:
            print("Gradient shader not allocated or shader program missing.")
            return
        if not velocity.allocated or not pressure.allocated or not obstacle.allocated or not obstacle_offset.allocated:
            print("Gradient shader: input texture(s) not allocated.")
            return

        glUseProgram(self.shader_program)

        # Bind textures
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, velocity.tex_id)
        glUniform1i(self.get_uniform_loc("uVelocity"), 0)

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, pressure.tex_id)
        glUniform1i(self.get_uniform_loc("uPressure"), 1)

        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, obstacle.tex_id)
        glUniform1i(self.get_uniform_loc("uObstacle"), 2)

        glActiveTexture(GL_TEXTURE3)
        glBindTexture(GL_TEXTURE_2D, obstacle_offset.tex_id)
        glUniform1i(self.get_uniform_loc("uObstacleOffset"), 3)

        # Set uniforms (aspect-corrected grid scales)
        half_rdx_x = 0.5 / grid_scale
        half_rdx_y = (0.5 / grid_scale) / aspect
        glUniform2f(self.get_uniform_loc("uHalfRdxInv"), half_rdx_x, half_rdx_y)

        draw_quad()

        # Cleanup
        glActiveTexture(GL_TEXTURE3)
        glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glUseProgram(0)
