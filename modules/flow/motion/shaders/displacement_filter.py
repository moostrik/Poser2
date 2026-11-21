"""Displacement filter shader: x/y through the depth track's filter over z's time, so it lags as z does."""

from OpenGL.GL import glUniform1f, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture
from ...finite_guard import GuardSlot


class DisplacementFilter(Shader):
    """Writes the filtered displacement (RG, uv per frame) and the position error (BA, uv)."""

    def use(self, displacement: Texture, state: Texture, carry: Texture, smooth: float, rest_speed: float, halving_speed: float,
            delta_time: float, guard: GuardSlot | None = None) -> None:
        """Filter one frame.

        Args:
            displacement: The measured displacement, uv per frame
            state: The filter the frame before
            carry: The displacement the state is carried by and the speed is read from, uv per frame
            smooth: Time followed over at rest (ms); 0 takes the measurement whole
            rest_speed: speed below which it is at rest (px per frame)
            halving_speed: speed (the larger of x/y's and the track's own) past `rest_speed` at which the time is half (px per frame); 0 is `smooth` throughout
            delta_time: Seconds since the last frame
            guard: The displacement's slot in the motion's FiniteGuard; None no guard
        """
        if not self.allocated or not self.shader_program:
            return
        if not displacement.allocated or not state.allocated or not carry.allocated:
            return

        glUseProgram(self.shader_program)

        displacement.bind(0)
        state.bind(1)
        carry.bind(2)

        glUniform1f(self.get_uniform_loc("smoothing"), smooth)
        glUniform1f(self.get_uniform_loc("rest_speed"), rest_speed)
        glUniform1f(self.get_uniform_loc("halving_speed"), halving_speed)
        glUniform1f(self.get_uniform_loc("delta_time"), delta_time)
        if guard is not None:
            guard.use(self.get_uniform_loc("uGuardSlot"))
        else:
            glUniform1i(self.get_uniform_loc("uGuardSlot"), -1)

        draw_quad()
