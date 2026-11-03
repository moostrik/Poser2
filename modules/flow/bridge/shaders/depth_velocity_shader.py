"""Depth Velocity shader.

The motion toward the camera in view widths per frame, from two smoothed depths.
"""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE2, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class DepthVelocityShader(Shader):
    """Writes the signed motion toward the camera to the red channel."""

    def use(self, depth: Texture, previous: Texture, view_span: float, jump: float, displacement: Texture | None = None) -> None:
        """Difference two depths, in view widths per frame.

        Args:
            depth: Smoothed depth
            previous: Previous smoothed depth
            view_span: Metres one view width spans per metre of depth; 0 or less reads no motion
            jump: More change than this is an edge (m)
            displacement: uv per frame; when given, the previous depth is taken where the pixel came from
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
        glUniform1f(self.get_uniform_loc("view_span"), view_span)
        glUniform1f(self.get_uniform_loc("jump"), jump)

        draw_quad()
