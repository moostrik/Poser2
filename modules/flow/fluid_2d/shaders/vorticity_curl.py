"""VorticityCurl shader - Compute curl of velocity field."""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUniform2f, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class VorticityCurl(Shader):
    """Compute curl (vorticity) of velocity field."""

    def __init__(self) -> None:
        super().__init__()

    def use(self, velocity: Texture, obstacle: Texture, cell: tuple[float, float], radius: float,
            has_obstacles: bool = True) -> None:
        """Compute velocity curl.

        Args:
            velocity: Velocity field (RG32F)
            obstacle: Obstacle mask (R8/R32F)
            cell: A cell's width and height, widths
            radius: The sampling radius, widths; at least one cell
            has_obstacles: When False, skip obstacle texture reads for performance
        """
        # the radius in cells per axis, at least one: the curl cannot be measured closer
        reach = (max(radius / cell[0], 1.0), max(radius / cell[1], 1.0))

        # Bind shader program
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

        # Set uniform
        glUniform2f(self.get_uniform_loc("uRadius"), *reach)
        glUniform2f(self.get_uniform_loc("uHalfRdxInv"), 0.5 / (cell[0] * reach[0]), 0.5 / (cell[1] * reach[1]))

        # Draw fullscreen quad
        draw_quad()
