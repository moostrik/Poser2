"""Bandpass Changes shader.

The bandpass's four parts in one dispatch, each its size and its velocity.
"""

from OpenGL.GL import GL_FRAMEBUFFER_BARRIER_BIT, GL_TEXTURE_FETCH_BARRIER_BIT, glMemoryBarrier, glUniform1f, glUniform2i, glUseProgram
from modules.gl import ComputeShader, Texture


class BandpassChanges(ComputeShader):
    """Writes each part's size and its velocity: start, bow, stop and wake."""

    WORKGROUP_SIZE_X = 16
    WORKGROUP_SIZE_Y = 16

    def use(self, fast: Texture, slow: Texture, rise_fast: Texture, rise_slow: Texture, fall_speeds: Texture,
            speeds: tuple[Texture, Texture, Texture, Texture], velocities: tuple[Texture, Texture, Texture, Texture],
            scale: float, rise_weight: float, fall_weight: float, fall_lod: float) -> None:
        """The four parts.

        Args:
            fast: The fast trail
            slow: The slow trail
            rise_fast: A rise's region: its mean moving velocity through the fast trail, a texel per region
            rise_slow: And through the slow trail
            fall_speeds: A fall's region: mean speeds, fast R, slow G, with mipmaps
            speeds: Start, bow, stop and wake's sizes (R16F)
            velocities: Their velocities (RGBA16F)
            scale: 1 / the trails' step peak; 0 = no changes
            rise_weight: start_region_weight; 0 = every rise is a start
            fall_weight: stop_region_weight; 0 = every fall is a stop
            fall_lod: A fall's region's mip level
        """
        if not self.allocated or not self.shader_program or not fast.allocated or not slow.allocated:
            return
        glUseProgram(self.shader_program)
        for unit, texture in enumerate((fast, slow, rise_fast, rise_slow, fall_speeds)):
            texture.bind(unit)
        for unit, texture in enumerate((*speeds, *velocities)):
            self.bind_image_write(unit, texture)
        glUniform1f(self.get_uniform_loc("uScale"), scale)
        glUniform1f(self.get_uniform_loc("uRiseWeight"), rise_weight)
        glUniform1f(self.get_uniform_loc("uFallWeight"), fall_weight)
        glUniform1f(self.get_uniform_loc("uFallLod"), fall_lod)
        glUniform2i(self.get_uniform_loc("uSize"), fast.width, fast.height)
        self.dispatch(fast.width, fast.height)
        glMemoryBarrier(int(GL_TEXTURE_FETCH_BARRIER_BIT) | int(GL_FRAMEBUFFER_BARRIER_BIT))    # read next by textures and FBOs
