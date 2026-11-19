"""Buoyancy shader - the density's weight and the warmth's lift along gravity, an acceleration (widths/s²)."""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE2, GL_TEXTURE3, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUniform2f, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class Buoyancy(Shader):
    """The buoyancy F = gravity × (κ·density − σ·T), widths/s², from the temperature and the density's amount; or
    a velocity plus F × dt, in one pass."""

    def use(self, temperature: Texture, density: Texture, obstacle: Texture, sigma: float, kappa: float,
            gravity: tuple[float, float], has_obstacles: bool = True, velocity: Texture | None = None,
            strength: float = 0.0) -> None:
        """Compute the buoyancy, or with `velocity` that velocity plus the buoyancy × `strength`.

        Args:
            temperature: Temperature field (R16F), the warmth; read only while `sigma` is above 0
            density: Density field (RGBA16F, premultiplied), its amount in alpha
            obstacle: Obstacle mask (R8, CLAMP_TO_BORDER=1)
            sigma: Lift per unit of warmth (widths/s²)
            kappa: Weight per unit of amount (widths/s²)
            gravity: Gravity's direction on the grid, (0, -1) down the screen
            has_obstacles: When False, skip obstacle texture reads for performance
            velocity: The velocity (RG16F) the buoyancy is added to; None for the buoyancy itself
            strength: With `velocity`, dt (s)
        """
        if not self.allocated or not self.shader_program:
            return
        glUseProgram(self.shader_program)
        glUniform1i(self.get_uniform_loc("uHasObstacles"), int(has_obstacles))

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, temperature.tex_id)
        glUniform1i(self.get_uniform_loc("uTemperature"), 0)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, density.tex_id)
        glUniform1i(self.get_uniform_loc("uDensity"), 1)
        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, obstacle.tex_id)
        glUniform1i(self.get_uniform_loc("uObstacle"), 2)
        glActiveTexture(GL_TEXTURE3)
        glBindTexture(GL_TEXTURE_2D, velocity.tex_id if velocity is not None else temperature.tex_id)
        glUniform1i(self.get_uniform_loc("uVelocity"), 3)
        glUniform1i(self.get_uniform_loc("uAdd"), int(velocity is not None))

        glUniform1f(self.get_uniform_loc("uSigma"), sigma)
        glUniform1f(self.get_uniform_loc("uKappa"), kappa)
        glUniform2f(self.get_uniform_loc("uGravity"), *gravity)
        glUniform1f(self.get_uniform_loc("uStrength"), strength)

        draw_quad()
