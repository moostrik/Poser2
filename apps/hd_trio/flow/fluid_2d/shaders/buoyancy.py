"""Buoyancy shader - Compute temperature-driven buoyancy forces."""

from OpenGL.GL import glUniform1f, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class Buoyancy(Shader):
    """Compute buoyancy force from temperature and density."""

    def __init__(self) -> None:
        super().__init__()

    def use(self, velocity: Texture, temperature: Texture, density: Texture,
            obstacle: Texture, sigma: float, kappa: float, ambient_temperature: float,
            has_obstacles: bool = True) -> None:
        """Compute buoyancy force: F = σ(T - T_ambient) - κρ

        Args:
            velocity: Current velocity field (RG32F) - passed for consistency but not used
            temperature: Temperature field (R32F)
            density: Density field (RGBA32F)
            obstacle: Obstacle mask (R8, CLAMP_TO_BORDER=1)
            sigma: Thermal buoyancy coefficient σ (already includes dt * scale)
            kappa: Density weight coefficient κ (already includes dt * scale)
            ambient_temperature: Reference temperature
            has_obstacles: When False, skip obstacle texture reads for performance
        """
        # Bind shader program
        glUseProgram(self.shader_program)

        # Obstacle flag
        glUniform1i(self.get_uniform_loc("uHasObstacles"), int(has_obstacles))

        # Bind textures
        velocity.bind(0)
        temperature.bind(1)
        density.bind(2)
        obstacle.bind(3)

        # Set uniforms
        glUniform1f(self.get_uniform_loc("uSigma"), sigma)
        glUniform1f(self.get_uniform_loc("uKappa"), kappa)
        glUniform1f(self.get_uniform_loc("uAmbientTemperature"), ambient_temperature)

        # Draw fullscreen quad
        draw_quad()
