"""Lowpass Step 2D shader.

One step of the 2D lowpass's trail: moved with the trail, then one Ema step, and its rate.
"""

from OpenGL.GL import (GL_FRAMEBUFFER_BARRIER_BIT, GL_TEXTURE_FETCH_BARRIER_BIT, glMemoryBarrier, glUniform1f, glUniform1i,
                       glUniform2f, glUniform2i, glUseProgram)
from modules.gl import ComputeShader, Texture


class LowpassStep2D(ComputeShader):
    """Writes the trail and, when asked, its rate (RG16F, xy)."""

    WORKGROUP_SIZE_X = 16
    WORKGROUP_SIZE_Y = 16

    def use(self, trail: Texture, velocity: Texture, trail_out: Texture, acceleration_out: Texture | None,
            carry: tuple[float, float], attack: float, release: float, rate: float) -> None:
        """One step.

        Args:
            trail: The trail before (xy widths/s), 0 outside
            velocity: The input velocity
            trail_out: The trail now (RG16F)
            acceleration_out: Its rate (RG16F), or None: not written
            carry: uv per frame per unit of trail velocity, times the advect amount; (0, 0) none
            attack: What remains of the trail where the speed rises
            release: What remains elsewhere
            rate: 1 / dt, for the rate
        """
        if not self.allocated or not self.shader_program:
            return
        if not all(texture.allocated for texture in (trail, velocity, trail_out)):
            return
        glUseProgram(self.shader_program)
        self.bind_texture(0, trail)
        self.bind_texture(1, velocity)
        self.bind_image_write(0, trail_out)
        if acceleration_out is not None:
            self.bind_image_write(1, acceleration_out)
        glUniform1i(self.get_uniform_loc("uAcceleration"), int(acceleration_out is not None))
        glUniform2f(self.get_uniform_loc("uCarry"), carry[0], carry[1])
        glUniform1f(self.get_uniform_loc("uAttack"), attack)
        glUniform1f(self.get_uniform_loc("uRelease"), release)
        glUniform1f(self.get_uniform_loc("uRate"), rate)
        glUniform2i(self.get_uniform_loc("uSize"), trail_out.width, trail_out.height)
        self.dispatch(trail_out.width, trail_out.height)
        glMemoryBarrier(int(GL_TEXTURE_FETCH_BARRIER_BIT) | int(GL_FRAMEBUFFER_BARRIER_BIT))    # read next by textures and FBOs
