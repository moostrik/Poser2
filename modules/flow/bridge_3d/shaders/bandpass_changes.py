"""Bandpass Changes shader.

The bandpass's four parts in one dispatch, each its size and its velocity.
"""

from enum import IntEnum

from OpenGL.GL import GL_FRAMEBUFFER_BARRIER_BIT, GL_TEXTURE_FETCH_BARRIER_BIT, glMemoryBarrier, glUniform1f, glUniform2i, glUseProgram
from modules.gl import ComputeShader, Texture


class Change(IntEnum):
    """The four changes: a rise of fast - slow along the motion that its region shares (START) or not (BOW,
    a limb arriving), a fall that its region shares (STOP) or not (WAKE, a limb leaving)."""
    START = 0
    BOW = 1
    STOP = 2
    WAKE = 3

    @property
    def sign(self) -> float:
        """+1 for a rise, -1 for a fall."""
        return 1.0 if self in (Change.START, Change.BOW) else -1.0

    @property
    def edge(self) -> bool:
        """Whether it is the part its region does not share."""
        return self in (Change.BOW, Change.WAKE)


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
        for unit, (texture, name) in enumerate(((fast, "uFast"), (slow, "uSlow"), (rise_fast, "uRiseFast"),
                                                (rise_slow, "uRiseSlow"), (fall_speeds, "uFallSpeeds"))):
            self.bind_texture(unit, texture, name)
        for unit, texture in enumerate((*speeds, *velocities)):
            self.bind_image_write(unit, texture)
        glUniform1f(self.get_uniform_loc("uScale"), scale)
        glUniform1f(self.get_uniform_loc("uRiseWeight"), rise_weight)
        glUniform1f(self.get_uniform_loc("uFallWeight"), fall_weight)
        glUniform1f(self.get_uniform_loc("uFallLod"), fall_lod)
        glUniform2i(self.get_uniform_loc("uSize"), fast.width, fast.height)
        self.dispatch(fast.width, fast.height)
        glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT | GL_FRAMEBUFFER_BARRIER_BIT)    # read next by textures and FBOs
