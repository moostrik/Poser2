"""Advect shader - Semi-Lagrangian advection with dissipation.

Advection formula:  ΔUV = dt × speed × velocity × (1.0, 1.0/aspect)
At speed=1.0, a velocity of 1.0 crosses the full texture width in 1 second.
Aspect correction ensures isotropic transport on non-square grids.
"""

from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture


class Advect(Shader):
    """Semi-Lagrangian advection shader with dissipation and aspect correction."""

    def __init__(self) -> None:
        super().__init__()

    def use(self, source: Texture, velocity: Texture, obstacle: Texture,
            aspect: float, timestep: float, dissipation: float) -> None:
        """Apply advection.

        Args:
            source: Field to advect (density, velocity, temperature, pressure)
            velocity: Velocity field (RG32F)
            obstacle: Obstacle mask (R8/R32F)
            aspect: Width/height ratio for isotropic advection
            timestep: dt × speed — advection distance per frame in UV space
            dissipation: Exponential decay multiplier (0.99 = 1% loss per frame)
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
        glUniform1i(self.get_uniform_loc("uVelocity"), 0)

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, source.tex_id)
        glUniform1i(self.get_uniform_loc("uSource"), 1)

        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, obstacle.tex_id)
        glUniform1i(self.get_uniform_loc("uObstacle"), 2)

        # Set uniforms
        glUniform1f(self.get_uniform_loc("uTimestep"), timestep)
        # Aspect-corrected inverse grid scale:
        # X: velocity=1.0 traces 1 UV unit (full width)
        # Y: scaled by 1/aspect so same velocity traces same physical distance
        rdx_x = 1.0
        rdx_y = 1.0 / aspect if aspect > 0.0 else 1.0
        glUniform2f(self.get_uniform_loc("uRdx"), rdx_x, rdx_y)
        glUniform1f(self.get_uniform_loc("uDissipation"), dissipation)

        draw_quad()
