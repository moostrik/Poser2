"""Depth Velocity shader.

One frame of the depth track: the motion toward the camera in view widths per frame, from a track in
disparity carried along the motion and corrected by the mean over a window of depth samples that passes the
gate, with the misses in a row beside it.
"""

from OpenGL.GL import glUniform1f, glUniform1i, glUniform2f, glUseProgram

from modules.gl import Shader, draw_quad, Texture
from ...finite_guard import GuardSlot


class DepthVelocityShader(Shader):
    """Writes the track: z (view widths per frame), disparity (1/m), its rate (1/m per second) and its
    anchor (m); and to the second target its misses in a row."""

    def use(self, depth: Texture, track: Texture, smooth: float, halving_speed: float, rest_speed: float, delta_time: float,
            view_span: float, depth_step: tuple[float, float], window: int, edge: float, dead_zone: float,
            misses: Texture, depth_noise: float, displacement: Texture | None = None, max_speed: float = 0.0,
            guard: GuardSlot | None = None) -> None:
        """Correct the track by the new depth into the bound target.

        Args:
            depth: Depth now (m) in red, its confidence (0 to 1) in green
            track: The track the frame before
            smooth: Time the track follows over at rest (ms); 0 takes the measurement whole
            halving_speed: speed (the larger of x/y's and the track's own) past `rest_speed` at which that time is half (px per frame); 0 is `smooth` throughout
            rest_speed: speed below which the motion is at rest (px per frame)
            delta_time: Seconds since the last depth
            view_span: Meters one view width spans per meter of depth; 0 or less reads no motion
            depth_step: uv between two depth samples
            window: Radius in depth samples
            edge: Depth step to another surface within a frame, not averaged in (m)
            dead_zone: Depth the track wanders within its anchor without z (m); 0 is none
            misses: The track's misses in a row the frame before
            depth_noise: The camera's depth noise at 1 m depth and full confidence (m), the gate's noise
            displacement: uv per frame; when given, the track is carried from where each pixel came from
            max_speed: The fastest a body part moves in depth (m/s): the gate's motion; a speed past it starts the track over. 0 is no gate
            guard: The depth velocity's slot in the motion's FiniteGuard; None no guard
        """
        if not self.allocated or not self.shader_program:
            return
        if not depth.allocated or not track.allocated or not misses.allocated:
            return

        glUseProgram(self.shader_program)

        depth.bind(0)
        track.bind(1)
        misses.bind(3)

        warp: bool = displacement is not None and displacement.allocated
        if warp:
            assert displacement is not None
            displacement.bind(2)

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
        glUniform1f(self.get_uniform_loc("depth_noise"), depth_noise)
        if guard is not None:
            guard.use(self.get_uniform_loc("uGuardSlot"))
        else:
            glUniform1i(self.get_uniform_loc("uGuardSlot"), -1)

        draw_quad()
