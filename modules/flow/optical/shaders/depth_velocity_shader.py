"""Depth Velocity shader.

The motion toward the camera in view widths per frame, over a window of depth samples.
"""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE2, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUniform2f, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class DepthVelocityShader(Shader):
    """Writes the signed motion toward the camera to the red channel."""

    def use(self, depth: Texture, previous: Texture, view_span: float, depth_step: tuple[float, float],
            window: int, tolerance: float, coverage: float, displacement: Texture | None = None) -> None:
        """Measure over the window, in view widths per frame.

        Args:
            depth: Depth now (m)
            previous: Depth the frame before (m)
            view_span: Metres one view width spans per metre of depth; 0 or less reads no motion
            depth_step: uv between two depth samples
            window: Radius in depth samples
            tolerance: Neighbours further than this in depth are not averaged in (m)
            coverage: Share of the window averaged in that the motion needs; less reads 0
            displacement: uv per frame; when given, each sample's previous depth is taken where it came from
        """
        if not self.allocated or not self.shader_program:
            return
        if not depth.allocated or not previous.allocated:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, depth.tex_id)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, previous.tex_id)

        warp: bool = displacement is not None and displacement.allocated
        if warp:
            assert displacement is not None
            glActiveTexture(GL_TEXTURE2)
            glBindTexture(GL_TEXTURE_2D, displacement.tex_id)

        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform1i(self.get_uniform_loc("tex1"), 1)
        glUniform1i(self.get_uniform_loc("displacement"), 2)
        glUniform1i(self.get_uniform_loc("warp"), int(warp))
        glUniform2f(self.get_uniform_loc("depth_step"), *depth_step)
        glUniform1i(self.get_uniform_loc("window"), window)
        glUniform1f(self.get_uniform_loc("tolerance"), tolerance)
        glUniform1f(self.get_uniform_loc("coverage"), coverage)
        glUniform1f(self.get_uniform_loc("view_span"), view_span)

        draw_quad()
