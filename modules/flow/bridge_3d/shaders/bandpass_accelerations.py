"""Bandpass Accelerations shader.

The bandpass's whole change and, when asked, every reading as an acceleration, in one dispatch.
"""

from OpenGL.GL import GL_FRAMEBUFFER_BARRIER_BIT, GL_TEXTURE_FETCH_BARRIER_BIT, glMemoryBarrier, glUniform1f, glUniform1i, glUniform2i, glUseProgram
from modules.gl import ComputeShader, Texture


class BandpassAccelerations(ComputeShader):
    """Writes the whole change, and with `accelerations` the change and each part as an acceleration."""

    WORKGROUP_SIZE_X = 16
    WORKGROUP_SIZE_Y = 16

    def use(self, fast: Texture, slow: Texture, velocities: tuple[Texture, Texture, Texture, Texture], change: Texture,
            accelerations: tuple[Texture, Texture, Texture, Texture, Texture] | None,
            scale: float, acceleration_scale: float, peak: float) -> None:
        """The whole change and the accelerations.

        Args:
            fast: The fast trail
            slow: The slow trail
            velocities: Start, bow, stop and wake as velocities
            change: The whole change (RGBA16F)
            accelerations: The change's, then start, bow, stop and wake's (RGBA16F); None: not written
            scale: 1 / the trails' step peak
            acceleration_scale: 1 / the band's width
            peak: The step peak, so a part's velocity × acceleration_scale × peak is its pulse over the band's width
        """
        if not self.allocated or not self.shader_program or not fast.allocated or not slow.allocated:
            return
        glUseProgram(self.shader_program)
        for unit, (texture, name) in enumerate(((fast, "uFast"), (slow, "uSlow"), *zip(velocities, ("uStart", "uBow", "uStop", "uWake")))):
            self.bind_texture(unit, texture, name)
        self.bind_image_write(0, change)
        if accelerations is not None:
            for unit, texture in enumerate(accelerations, start=1):
                self.bind_image_write(unit, texture)
        glUniform1i(self.get_uniform_loc("uAcceleration"), int(accelerations is not None))
        glUniform1f(self.get_uniform_loc("uScale"), scale)
        glUniform1f(self.get_uniform_loc("uAccelerationScale"), acceleration_scale)
        glUniform1f(self.get_uniform_loc("uPartScale"), acceleration_scale * peak)
        glUniform2i(self.get_uniform_loc("uSize"), fast.width, fast.height)
        self.dispatch(fast.width, fast.height)
        glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT | GL_FRAMEBUFFER_BARRIER_BIT)    # read next by textures and FBOs
