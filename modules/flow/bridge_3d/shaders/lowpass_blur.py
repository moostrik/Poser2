"""Lowpass Blur shader.

The lowpass's textbook Gaussian blur by one sigma, up to three fields in one dispatch.
"""

from OpenGL.GL import GL_FRAMEBUFFER_BARRIER_BIT, GL_TEXTURE_FETCH_BARRIER_BIT, glMemoryBarrier, glUniform1fv, glUniform1i, glUniform2i, glUseProgram
from modules.gl import ComputeShader, Texture


class LowpassBlur(ComputeShader):
    """Blurs each source into its target by the weights given, both directions in one dispatch."""

    WORKGROUP_SIZE_X = 16
    WORKGROUP_SIZE_Y = 16
    MAX_RADIUS: int = 8         # taps on each side one dispatch reaches: MAX_RADIUS in the shader

    def use(self, sources: list[Texture | None], targets: list[Texture | None], weights: list[float]) -> None:
        """Blur each source into its target, the same size, up to three slots, a slot None or left out unused.

        Args:
            sources: Fields to blur, read as textures
            targets: Where each goes, written as images: RGBA16F in the first two slots, RG32F in the third
            weights: Tap k's weight for k = 0..radius, normalized over both sides (`gaussian_weights`)
        """
        if not self.allocated or not self.shader_program or len(sources) != len(targets):
            return
        slots: list[tuple[int, Texture, Texture]] = [(unit, source, target) for unit, (source, target) in enumerate(zip(sources[:3], targets[:3]))
                                                     if source is not None and target is not None and source.allocated and target.allocated]
        if not slots:
            return
        radius: int = min(len(weights) - 1, self.MAX_RADIUS)
        glUseProgram(self.shader_program)
        fields: int = 0
        for unit, source, target in slots:
            self.bind_texture(unit, source, f"uSource{unit}")
            self.bind_image_write(unit, target)
            fields |= 1 << unit
        glUniform1i(self.get_uniform_loc("uFields"), fields)
        glUniform1i(self.get_uniform_loc("uRadius"), radius)
        padded: list[float] = (list(weights[:radius + 1]) + [0.0] * (self.MAX_RADIUS + 1))[:self.MAX_RADIUS + 1]
        glUniform1fv(self.get_uniform_loc("uWeights"), self.MAX_RADIUS + 1, padded)
        width, height = slots[0][2].width, slots[0][2].height
        glUniform2i(self.get_uniform_loc("uSize"), width, height)
        self.dispatch(width, height)
        glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT | GL_FRAMEBUFFER_BARRIER_BIT)    # read next by textures and FBOs
