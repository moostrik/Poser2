"""Bandpass Trails 2D shader.

One step of the 2D bandpass's fast and slow trail.
"""

from OpenGL.GL import GL_FRAMEBUFFER_BARRIER_BIT, GL_TEXTURE_FETCH_BARRIER_BIT, glMemoryBarrier, glUniform1f, glUniform2i, glUseProgram
from modules.gl import ComputeShader, Texture


class BandpassTrails2D(ComputeShader):
    """Writes the fast and the slow trail (RG16F, xy), one Ema step each."""

    WORKGROUP_SIZE_X = 16
    WORKGROUP_SIZE_Y = 16

    def use(self, velocity: Texture, fast: Texture, slow: Texture, fast_out: Texture, slow_out: Texture,
            fast_weight: float, slow_weight: float) -> None:
        """One step: `fast` and `slow` before, into the `_out`s.

        Args:
            velocity: The input velocity (xy widths/s)
            fast_weight: What remains of the fast trail a step
            slow_weight: And of the slow one
        """
        if not self.allocated or not self.shader_program:
            return
        if not all(texture.allocated for texture in (velocity, fast, slow, fast_out, slow_out)):
            return
        glUseProgram(self.shader_program)
        for unit, (texture, name) in enumerate(((velocity, "uVelocity"), (fast, "uFast"), (slow, "uSlow"))):
            self.bind_texture(unit, texture, name)
        self.bind_image_write(0, fast_out)
        self.bind_image_write(1, slow_out)
        glUniform1f(self.get_uniform_loc("uFastWeight"), fast_weight)
        glUniform1f(self.get_uniform_loc("uSlowWeight"), slow_weight)
        glUniform2i(self.get_uniform_loc("uSize"), fast_out.width, fast_out.height)
        self.dispatch(fast_out.width, fast_out.height)
        glMemoryBarrier(int(GL_TEXTURE_FETCH_BARRIER_BIT) | int(GL_FRAMEBUFFER_BARRIER_BIT))    # read next by textures and FBOs
