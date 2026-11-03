"""Depth Velocity shader.

One frame of the depth track: the motion toward the camera in view widths per frame, from a track in
disparity carried along the motion and corrected by the mean over a window of depth samples.
"""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE2, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUniform2f, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class DepthVelocityShader(Shader):
    """Writes the track: z (view widths per frame), disparity (1/m), its rate (1/m per second) and its anchor (m)."""

    def use(self, depth: Texture, track: Texture, alpha: float, delta_time: float, view_span: float, depth_step: tuple[float, float],
            window: int, tolerance: float, dead_zone: float, displacement: Texture | None = None) -> None:
        """Correct the track by the new depth.

        Args:
            depth: Depth now (m)
            track: The track the frame before
            alpha: Weight of a full window's measurement on the disparity (1 takes it whole)
            delta_time: Seconds since the last depth
            view_span: Metres one view width spans per metre of depth; 0 or less reads no motion
            depth_step: uv between two depth samples
            window: Radius in depth samples
            tolerance: Depth further than this is another surface: not averaged in, the track starts over (m)
            dead_zone: Depth the track wanders within its anchor without z (m); 0 is none
            displacement: uv per frame; when given, the track is carried from where each pixel came from
        """
        if not self.allocated or not self.shader_program:
            return
        if not depth.allocated or not track.allocated:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, depth.tex_id)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, track.tex_id)

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
        glUniform1f(self.get_uniform_loc("alpha"), alpha)
        glUniform1f(self.get_uniform_loc("delta_time"), delta_time)
        glUniform1f(self.get_uniform_loc("view_span"), view_span)
        glUniform1f(self.get_uniform_loc("dead_zone"), dead_zone)

        draw_quad()
