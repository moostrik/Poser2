"""Lowpass Step shader.

One step of the lowpass's trail and its z position: moved with the trail, then one Ema step, and its rate.
"""

from OpenGL.GL import (GL_FRAMEBUFFER_BARRIER_BIT, GL_TEXTURE_FETCH_BARRIER_BIT, glMemoryBarrier, glUniform1f, glUniform1i,
                       glUniform2f, glUniform2i, glUseProgram)
from modules.gl import ComputeShader, Texture


class LowpassStep(ComputeShader):
    """Writes the trail, its z position and, when asked, its rate."""

    WORKGROUP_SIZE_X = 16
    WORKGROUP_SIZE_Y = 16

    def use(self, trail: Texture, z_position: Texture, velocity: Texture, current: Texture | None,
            trail_out: Texture, z_position_out: Texture, acceleration_out: Texture | None,
            carry: tuple[float, float], attack: float, release: float, rate: float) -> None:
        """One step.

        Args:
            trail: The trail before (xyz widths/s), 0 outside
            z_position: Its z position before (RG32F, red over green), 0 outside
            velocity: The input velocity
            current: The motion's z position now (red over green), or None: no z position comes in
            trail_out: The trail now (RGBA16F)
            z_position_out: Its z position now (RG32F)
            acceleration_out: Its rate (RGBA16F), or None: not written
            carry: uv per frame per unit of trail velocity, times the advect amount; (0, 0) none
            attack: What remains of the trail where the speed rises
            release: What remains elsewhere
            rate: 1 / dt, for the rate
        """
        if not self.allocated or not self.shader_program:
            return
        if not all(texture.allocated for texture in (trail, z_position, velocity, trail_out, z_position_out)):
            return
        glUseProgram(self.shader_program)
        self.bind_texture(0, trail, "uTrail")
        self.bind_texture(1, z_position, "uZPosition")
        self.bind_texture(2, velocity, "uVelocity")
        has_current: bool = current is not None and current.allocated
        if has_current:
            assert current is not None
            self.bind_texture(3, current, "uCurrent")
        glUniform1i(self.get_uniform_loc("uHasCurrent"), int(has_current))
        self.bind_image_write(0, trail_out)
        self.bind_image_write(1, z_position_out)
        if acceleration_out is not None:
            self.bind_image_write(2, acceleration_out)
        glUniform1i(self.get_uniform_loc("uAcceleration"), int(acceleration_out is not None))
        glUniform2f(self.get_uniform_loc("uCarry"), carry[0], carry[1])
        glUniform1f(self.get_uniform_loc("uAttack"), attack)
        glUniform1f(self.get_uniform_loc("uRelease"), release)
        glUniform1f(self.get_uniform_loc("uRate"), rate)
        glUniform2i(self.get_uniform_loc("uSize"), trail_out.width, trail_out.height)
        self.dispatch(trail_out.width, trail_out.height)
        glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT | GL_FRAMEBUFFER_BARRIER_BIT)    # read next by textures and FBOs
