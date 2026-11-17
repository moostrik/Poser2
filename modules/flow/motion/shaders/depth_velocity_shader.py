"""Depth Velocity shader.

One frame of the depth track: the motion toward the camera in view widths per frame, from a track in
disparity carried along the motion and corrected by the mean over a window of depth samples.
"""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE2, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUniform2f, glUseProgram

from modules.gl import Shader, draw_quad, Texture
from ...finite_guard import GuardSlot


class DepthVelocityShader(Shader):
    """Writes the track: z (view widths per frame), disparity (1/m), its rate (1/m per second) and its
    anchor (m)."""

    def use(self, depth: Texture, track: Texture, smooth: float, halving_speed: float, rest_speed: float, delta_time: float,
            view_span: float, depth_step: tuple[float, float], window: int, edge: float, dead_zone: float,
            displacement: Texture | None = None, max_speed: float = 0.0, guard: GuardSlot | None = None) -> None:
        """Correct the track by the new depth into the bound target.

        Args:
            depth: Depth now (m)
            track: The track the frame before
            smooth: Time the track follows over at rest (ms); 0 takes the measurement whole
            halving_speed: speed (the larger of x/y's and the track's own) past `rest_speed` at which that time is half (px per frame); 0 is `smooth` throughout
            rest_speed: speed below which the motion is at rest (px per frame)
            delta_time: Seconds since the last depth
            view_span: Metres one view width spans per metre of depth; 0 or less reads no motion
            depth_step: uv between two depth samples
            window: Radius in depth samples
            edge: Depth step to another surface within a frame, not averaged in (m)
            dead_zone: Depth the track wanders within its anchor without z (m); 0 is none
            displacement: uv per frame; when given, the track is carried from where each pixel came from
            max_speed: The fastest a body part moves in depth (m/s): a jump or a speed past it starts the track over. 0 is no check
            guard: The depth velocity's slot in the motion's FiniteGuard; None no guard
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
        glUniform1f(self.get_uniform_loc("edge"), edge)
        glUniform1f(self.get_uniform_loc("smoothing"), smooth)
        glUniform1f(self.get_uniform_loc("halving_speed"), halving_speed)
        glUniform1f(self.get_uniform_loc("rest_speed"), rest_speed)
        glUniform1f(self.get_uniform_loc("delta_time"), delta_time)
        glUniform1f(self.get_uniform_loc("view_span"), view_span)
        glUniform1f(self.get_uniform_loc("dead_zone"), dead_zone)
        glUniform1f(self.get_uniform_loc("max_speed"), max_speed)
        if guard is not None:
            guard.use(self.get_uniform_loc("uGuardSlot"))
        else:
            glUniform1i(self.get_uniform_loc("uGuardSlot"), -1)

        draw_quad()
