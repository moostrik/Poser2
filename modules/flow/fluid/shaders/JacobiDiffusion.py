"""JacobiDiffusion shader - Iterative diffusion solver for velocity viscosity."""

from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture


class JacobiDiffusion(Shader):
    """Jacobi iterative solver for diffusion (viscosity)."""

    def __init__(self) -> None:
        super().__init__()

    def use(self, source: Texture, obstacle: Texture, obstacle_offset: Texture,
            grid_scale: float, timestep: float) -> None:
        """Apply one Jacobi iteration for diffusion.

        Args:
            source: Previous iteration of field to diffuse (velocity RG32F)
            obstacle: Obstacle mask (R8/R32F)
            obstacle_offset: Neighbor obstacle info (RGBA8)
            grid_scale: Grid spacing (typically 1.0)
            timestep: Diffusion timestep (controls diffusion rate)
        """
        # Compute Jacobi parameters
        alpha = (grid_scale * grid_scale) / timestep
        beta = 1.0 / (4.0 + alpha)

        # Bind shader program
        glUseProgram(self.shader_program)

        # Bind textures
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, source.tex_id)
        glUniform1i(glGetUniformLocation(self.shader_program, "uSource"), 0)

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, obstacle.tex_id)
        glUniform1i(glGetUniformLocation(self.shader_program, "uObstacle"), 1)

        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, obstacle_offset.tex_id)
        glUniform1i(glGetUniformLocation(self.shader_program, "uObstacleOffset"), 2)

        # Set uniforms
        glUniform1f(glGetUniformLocation(self.shader_program, "uAlpha"), alpha)
        glUniform1f(glGetUniformLocation(self.shader_program, "uBeta"), beta)

        # Draw fullscreen quad
        draw_quad()
