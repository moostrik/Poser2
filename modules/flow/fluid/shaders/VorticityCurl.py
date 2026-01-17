"""VorticityCurl shader - Compute curl of velocity field."""

from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture


class VorticityCurl(Shader):
    """Compute curl (vorticity) of velocity field."""

    def __init__(self) -> None:
        super().__init__()

    def use(self, velocity: Texture, obstacle: Texture, grid_scale: float) -> None:
        """Compute velocity curl.

        Args:
            velocity: Velocity field (RG32F)
            obstacle: Obstacle mask (R8/R32F)
            grid_scale: Grid spacing (typically 1.0)
        """
        halfrdx = 0.5 / grid_scale

        # Bind shader program
        glUseProgram(self.shader_program)

        # Bind textures
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, velocity.tex_id)
        glUniform1i(glGetUniformLocation(self.shader_program, "uVelocity"), 0)

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, obstacle.tex_id)
        glUniform1i(glGetUniformLocation(self.shader_program, "uObstacle"), 1)

        # Set uniform
        glUniform1f(glGetUniformLocation(self.shader_program, "uHalfRdx"), halfrdx)

        # Draw fullscreen quad
        draw_quad()
