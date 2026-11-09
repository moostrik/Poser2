"""Depth Velocity shader.

One frame of the depth track: the motion toward the camera in view widths per frame, from a track in
disparity carried along the motion and corrected by the mean over a window of depth samples.
"""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE2, GL_TEXTURE3, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUniform2f, glUseProgram

from modules.gl import Shader, draw_quad, Texture
from ...finite_guard import GuardSlot


class DepthVelocityShader(Shader):
    """Writes the track: z (view widths per frame), disparity (1/m), its rate (1/m per second) and its
    anchor (m); or with `catch_up_pass`, the catch-up: the Emas of the miss of the prediction, signed and its size (m)."""

    def use(self, depth: Texture, track: Texture, catch_up: Texture, catch_up_pass: bool, alpha: float, smooth: float, catch_up_smooth: float,
            ema: float, motion_floor: float, delta_time: float, view_span: float, depth_step: tuple[float, float], window: int,
            tolerance: float, dead_zone: float, displacement: Texture | None = None, max_speed: float = 0.0,
            guard: GuardSlot | None = None) -> None:
        """Correct the track by the new depth, or with `catch_up_pass` write this frame's catch-up.

        Args:
            depth: Depth now (m)
            track: The track the frame before
            catch_up: The catch-up the frame before
            catch_up_pass: Write the catch-up instead of the track
            alpha: Weight of a full window's measurement on the disparity (1 takes it whole)
            smooth: Time the track follows over (ms); 0 takes the measurement whole
            catch_up_smooth: Time it follows over while catching up (ms); 0 is never
            ema: Weight of a new miss on its Emas
            motion_floor: Displacement the image must move by to catch up (px per frame)
            delta_time: Seconds since the last depth
            view_span: Metres one view width spans per metre of depth; 0 or less reads no motion
            depth_step: uv between two depth samples
            window: Radius in depth samples
            tolerance: Depth further than this is another surface: not averaged in, the track starts over (m)
            dead_zone: Depth the track wanders within its anchor without z (m); 0 is none
            displacement: uv per frame; when given, the track is carried from where each pixel came from
            max_speed: The fastest a body part moves in depth (m/s); a track past it starts over. 0 is none
            guard: The depth velocity's slot in the motion's FiniteGuard; None no guard
        """
        if not self.allocated or not self.shader_program:
            return
        if not depth.allocated or not track.allocated or not catch_up.allocated:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, depth.tex_id)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, track.tex_id)
        glActiveTexture(GL_TEXTURE3)
        glBindTexture(GL_TEXTURE_2D, catch_up.tex_id)

        warp: bool = displacement is not None and displacement.allocated
        if warp:
            assert displacement is not None
            glActiveTexture(GL_TEXTURE2)
            glBindTexture(GL_TEXTURE_2D, displacement.tex_id)

        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform1i(self.get_uniform_loc("tex1"), 1)
        glUniform1i(self.get_uniform_loc("displacement"), 2)
        glUniform1i(self.get_uniform_loc("tex2"), 3)
        glUniform1i(self.get_uniform_loc("warp"), int(warp))
        glUniform1i(self.get_uniform_loc("catch_up_pass"), int(catch_up_pass))
        glUniform2f(self.get_uniform_loc("depth_step"), *depth_step)
        glUniform1i(self.get_uniform_loc("window"), window)
        glUniform1f(self.get_uniform_loc("tolerance"), tolerance)
        glUniform1f(self.get_uniform_loc("alpha"), alpha)
        glUniform1f(self.get_uniform_loc("smoothing"), smooth)
        glUniform1f(self.get_uniform_loc("catch_up_smoothing"), catch_up_smooth)
        glUniform1f(self.get_uniform_loc("ema"), ema)
        glUniform1f(self.get_uniform_loc("motion_floor"), motion_floor)
        glUniform1f(self.get_uniform_loc("delta_time"), delta_time)
        glUniform1f(self.get_uniform_loc("view_span"), view_span)
        glUniform1f(self.get_uniform_loc("dead_zone"), dead_zone)
        glUniform1f(self.get_uniform_loc("max_speed"), max_speed)
        if guard is not None:
            guard.use(self.get_uniform_loc("uGuardSlot"))
        else:
            glUniform1i(self.get_uniform_loc("uGuardSlot"), -1)

        draw_quad()
