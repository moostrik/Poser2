"""Divergence shader - Compute divergence of velocity field."""

from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture


class Divergence(Shader):
    """Compute velocity field divergence."""

    def __init__(self) -> None:
        super().__init__()

    def use(self, velocity: Texture, obstacle: Texture, grid_scale: float, aspect: float,
            has_obstacles: bool = True) -> None:
        """Compute divergence.

        Args:
            velocity: Velocity field (RG32F)
            obstacle: Obstacle mask (R8, CLAMP_TO_BORDER=1)
            grid_scale: Grid scaling factor (derived from config width / output width)
            aspect: Aspect ratio (width/height) for isotropic derivatives
            has_obstacles: When False, skip obstacle texture reads for performance
        """
        if not self.allocated or not self.shader_program:
            print("Divergence shader not allocated or shader program missing.")
            return
        if not velocity.allocated or not obstacle.allocated:
            print("Divergence shader: input texture(s) not allocated.")
            return

        glUseProgram(self.shader_program)

        # Obstacle flag
        glUniform1i(self.get_uniform_loc("uHasObstacles"), int(has_obstacles))

        # Bind textures
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, velocity.tex_id)
        glUniform1i(self.get_uniform_loc("uVelocity"), 0)

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, obstacle.tex_id)
        glUniform1i(self.get_uniform_loc("uObstacle"), 1)

        # Set uniforms (aspect-corrected grid scales)
        half_rdx_x = 0.5 / grid_scale
        half_rdx_y = (0.5 / grid_scale) / aspect
        glUniform2f(self.get_uniform_loc("uHalfRdxInv"), half_rdx_x, half_rdx_y)

        draw_quad()
