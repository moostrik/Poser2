"""Buoyancy shader - Compute temperature-driven buoyancy forces."""

from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture


class Buoyancy(Shader):
    """Compute buoyancy force from temperature and density."""

    def __init__(self) -> None:
        super().__init__()

    def use(self, velocity: Texture, temperature: Texture, density: Texture,
            sigma: float, kappa: float, ambient_temperature: float) -> None:
        """Compute buoyancy force: F = σ(T - T_ambient) - κρ

        Args:
            velocity: Current velocity field (RG32F) - passed for consistency but not used
            temperature: Temperature field (R32F)
            density: Density field (RGBA32F)
            sigma: Thermal buoyancy coefficient σ (already includes dt * scale)
            kappa: Density weight coefficient κ (already includes dt * scale)
            ambient_temperature: Reference temperature
        """
        # Bind shader program
        glUseProgram(self.shader_program)

        # Bind textures
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, velocity.tex_id)
        glUniform1i(self.get_uniform_loc("uVelocity"), 0)

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, temperature.tex_id)
        glUniform1i(self.get_uniform_loc("uTemperature"), 1)

        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, density.tex_id)
        glUniform1i(self.get_uniform_loc("uDensity"), 2)

        # Set uniforms
        glUniform1f(self.get_uniform_loc("uSigma"), sigma)
        glUniform1f(self.get_uniform_loc("uKappa"), kappa)
        glUniform1f(self.get_uniform_loc("uAmbientTemperature"), ambient_temperature)

        # Draw fullscreen quad
        draw_quad()

        # Cleanup
        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glUseProgram(0)
