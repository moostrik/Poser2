"""Depth Step Filter shader.

The depth without its edges, where its 3×3 depth samples span two surfaces.
"""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUniform2f, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class DepthStepFilter(Shader):
    """Writes the depth (R, m) with 0 where its 3×3 depth samples span a step beyond the tolerance."""

    def use(self, depth: Texture, depth_step: tuple[float, float], tolerance: float) -> None:
        """Take the edges off.

        Args:
            depth: Depth (m), 0 where there is none
            depth_step: uv between two depth samples
            tolerance: A step beyond this is two surfaces (m)
        """
        if not self.allocated or not self.shader_program:
            return
        if not depth.allocated:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, depth.tex_id)

        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform2f(self.get_uniform_loc("depth_step"), *depth_step)
        glUniform1f(self.get_uniform_loc("tolerance"), tolerance)

        draw_quad()
