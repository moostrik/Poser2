"""Bandpass Trails shader.

One step of the bandpass's fast and slow trail and their z positions.
"""

from OpenGL.GL import GL_FRAMEBUFFER_BARRIER_BIT, GL_TEXTURE_FETCH_BARRIER_BIT, glMemoryBarrier, glUniform1f, glUniform1i, glUniform2i, glUseProgram
from modules.gl import ComputeShader, Texture


class BandpassTrails(ComputeShader):
    """Writes the fast and the slow trail and their z positions, one Ema step each."""

    WORKGROUP_SIZE_X = 16
    WORKGROUP_SIZE_Y = 16

    def use(self, velocity: Texture, fast: Texture, slow: Texture, fast_z_position: Texture, slow_z_position: Texture,
            current: Texture | None, fast_out: Texture, slow_out: Texture, fast_z_position_out: Texture,
            slow_z_position_out: Texture, fast_weight: float, slow_weight: float) -> None:
        """One step: `fast`, `slow` and their z positions before, into the `_out`s.

        Args:
            velocity: The input velocity (xyz widths/s)
            current: The motion's z position now (red over green), or None: no z position comes in
            fast_weight: What remains of the fast trail a step
            slow_weight: And of the slow one
        """
        if not self.allocated or not self.shader_program:
            return
        textures = (velocity, fast, slow, fast_z_position, slow_z_position, fast_out, slow_out, fast_z_position_out, slow_z_position_out)
        if not all(texture.allocated for texture in textures):
            return
        glUseProgram(self.shader_program)
        for unit, (texture, name) in enumerate(((velocity, "uVelocity"), (fast, "uFast"), (slow, "uSlow"),
                                                (fast_z_position, "uFastZPosition"), (slow_z_position, "uSlowZPosition"))):
            self.bind_texture(unit, texture, name)
        has_current: bool = current is not None and current.allocated
        if has_current:
            assert current is not None
            self.bind_texture(5, current, "uCurrent")
        glUniform1i(self.get_uniform_loc("uHasCurrent"), int(has_current))
        for unit, texture in enumerate((fast_out, slow_out, fast_z_position_out, slow_z_position_out)):
            self.bind_image_write(unit, texture)
        glUniform1f(self.get_uniform_loc("uFastWeight"), fast_weight)
        glUniform1f(self.get_uniform_loc("uSlowWeight"), slow_weight)
        glUniform2i(self.get_uniform_loc("uSize"), fast_out.width, fast_out.height)
        self.dispatch(fast_out.width, fast_out.height)
        glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT | GL_FRAMEBUFFER_BARRIER_BIT)    # read next by textures and FBOs
