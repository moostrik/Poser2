"""Advect shader - Semi-Lagrangian advection with dissipation."""

from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture


class Advect(Shader):
    """Semi-Lagrangian advection shader with dissipation."""

    def __init__(self) -> None:
        super().__init__()

    def use(self, source: Texture, velocity: Texture, obstacle: Texture,
            grid_scale: float, timestep: float, dissipation: float) -> None:
        """Apply advection.

        Args:
            source: Field to advect (density, velocity, temperature, pressure)
            velocity: Velocity field (RG32F)
            obstacle: Obstacle mask (R8/R32F)
            grid_scale: Grid scaling factor (typically 1)
            timestep: Time step (speed * deltaTime * 100)
            dissipation: Energy loss multiplier (1.0 - dissipation_rate * deltaTime)
        """
        if not self.allocated or not self.shader_program:
            print("Advect shader not allocated or shader program missing.")
            return
        if not source.allocated or not velocity.allocated or not obstacle.allocated:
            print("Advect shader: input texture(s) not allocated.")
            return

        glUseProgram(self.shader_program)

        # Bind textures
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, velocity.tex_id)
        glUniform1i(glGetUniformLocation(self.shader_program, "uVelocity"), 0)

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, source.tex_id)
        glUniform1i(glGetUniformLocation(self.shader_program, "uSource"), 1)

        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, obstacle.tex_id)
        glUniform1i(glGetUniformLocation(self.shader_program, "uObstacle"), 2)

        # Set uniforms
        glUniform1f(glGetUniformLocation(self.shader_program, "uTimestep"), timestep)
        glUniform1f(glGetUniformLocation(self.shader_program, "uRdx"), 1.0 / grid_scale)
        glUniform1f(glGetUniformLocation(self.shader_program, "uDissipation"), dissipation)

        # Scale for resolution differences
        scale_x = velocity.width / source.width
        scale_y = velocity.height / source.height
        glUniform2f(glGetUniformLocation(self.shader_program, "uScale"), scale_x, scale_y)

        draw_quad()

        # Cleanup
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glUseProgram(0)
