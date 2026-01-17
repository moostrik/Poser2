"""VorticityForce shader - Compute vorticity confinement force."""

from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture


class VorticityForce(Shader):
    """Compute vorticity confinement force for turbulent swirls."""

    def __init__(self) -> None:
        super().__init__()

    def use(self, curl: Texture, grid_scale: float, timestep: float) -> None:
        """Compute vorticity confinement force.

        Args:
            curl: Curl magnitude field (R32F)
            grid_scale: Grid spacing (typically 1.0)
            timestep: Vorticity timestep (controls turbulence strength)
        """
        halfrdx = 0.5 / grid_scale

        # Bind shader program
        glUseProgram(self.shader_program)

        # Bind texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, curl.tex_id)
        glUniform1i(glGetUniformLocation(self.shader_program, "uCurl"), 0)

        # Set uniforms
        glUniform1f(glGetUniformLocation(self.shader_program, "uHalfRdx"), halfrdx)
        glUniform1f(glGetUniformLocation(self.shader_program, "uTimestep"), timestep)

        # Draw fullscreen quad
        draw_quad()
