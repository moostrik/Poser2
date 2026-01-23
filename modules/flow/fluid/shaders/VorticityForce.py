"""VorticityForce shader - Compute vorticity confinement force."""

from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture


class VorticityForce(Shader):
    """Compute vorticity confinement force for turbulent swirls."""

    def __init__(self) -> None:
        super().__init__()

    def use(self, curl: Texture, grid_scale: float, aspect: float, timestep: float) -> None:
        """Compute vorticity confinement force.

        Args:
            curl: Curl magnitude field (R32F)
            grid_scale: Grid spacing (typically simulation_scale)
            aspect: Aspect ratio (width/height) for isotropic derivatives
            timestep: Vorticity timestep (controls turbulence strength)
        """
        half_rdx_x = 0.5 / grid_scale
        half_rdx_y = (0.5 / grid_scale) / aspect

        # Bind shader program
        glUseProgram(self.shader_program)

        # Bind texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, curl.tex_id)
        glUniform1i(self.get_uniform_loc("uCurl"), 0)

        # Set uniforms
        glUniform2f(self.get_uniform_loc("uHalfRdxInv"), half_rdx_x, half_rdx_y)
        glUniform1f(self.get_uniform_loc("uTimestep"), timestep)

        # Draw fullscreen quad
        draw_quad()

        # Cleanup
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glUseProgram(0)
