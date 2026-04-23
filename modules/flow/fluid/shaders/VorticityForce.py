"""VorticityForce shader - Compute vorticity confinement force."""

from OpenGL.GL import *  # type: ignore
from modules.gl import Shader, draw_quad, Texture


class VorticityForce(Shader):
    """Compute vorticity confinement force for turbulent swirls."""

    def __init__(self) -> None:
        super().__init__()

    def use(self, curl: Texture, obstacle: Texture, grid_scale: float, aspect: float, timestep: float,
            has_obstacles: bool = True) -> None:
        """Compute vorticity confinement force.

        Args:
            curl: Curl magnitude field (R32F)
            obstacle: Obstacle mask (R8, CLAMP_TO_BORDER=1)
            grid_scale: Grid spacing (derived from config width / output width)
            aspect: Aspect ratio (width/height) for isotropic derivatives
            timestep: Vorticity timestep (controls turbulence strength)
            has_obstacles: When False, skip obstacle texture reads for performance
        """
        half_rdx_x = 0.5 / grid_scale
        half_rdx_y = (0.5 / grid_scale) / aspect

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
        glUniform2f(self.get_uniform_loc("uHalfRdxInv"), half_rdx_x, half_rdx_y)
        glUniform1f(self.get_uniform_loc("uTimestep"), timestep)

        # Draw fullscreen quad
        draw_quad()
