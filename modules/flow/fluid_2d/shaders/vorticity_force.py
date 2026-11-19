"""VorticityForce shader - Compute vorticity confinement force."""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUniform2f, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class VorticityForce(Shader):
    """Compute vorticity confinement force for turbulent swirls."""

    def __init__(self) -> None:
        super().__init__()

    def use(self, curl: Texture, obstacle: Texture, cell: tuple[float, float], timestep: float,
            has_obstacles: bool = True) -> None:
        """Compute vorticity confinement force.

        Args:
            curl: Curl magnitude field (R32F)
            obstacle: Obstacle mask (R8, CLAMP_TO_BORDER=1)
            cell: A cell's width and height, widths
            timestep: The strength epsilon × the cell's width × dt
            has_obstacles: When False, skip obstacle texture reads for performance
        """

        # Bind shader program
        glUseProgram(self.shader_program)

        # Obstacle flag
        glUniform1i(self.get_uniform_loc("uHasObstacles"), int(has_obstacles))

        # Bind textures
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, curl.tex_id)
        glUniform1i(self.get_uniform_loc("uCurl"), 0)

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, obstacle.tex_id)
        glUniform1i(self.get_uniform_loc("uObstacle"), 1)

        # Set uniforms
        glUniform2f(self.get_uniform_loc("uHalfRdxInv"), 0.5 / cell[0], 0.5 / cell[1])
        glUniform1f(self.get_uniform_loc("uTimestep"), timestep)

        # Draw fullscreen quad
        draw_quad()
