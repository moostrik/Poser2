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

    def use(self, sources: list[Texture], targets: list[Texture], weights: list[float]) -> None:
        """Blur `sources` into `targets`, the same size, one to three of each.

        Args:
            sources: Fields to blur, read as textures
            targets: Where each goes, written as images: RGBA16F for the first two, RG32F for the third
            weights: Tap k's weight for k = 0..radius, normalized over both sides (`gaussian_weights`)
        """
        if not self.allocated or not self.shader_program or not sources or len(sources) != len(targets):
            return
        if not all(texture.allocated for texture in (*sources, *targets)):
            return
        radius: int = min(len(weights) - 1, self.MAX_RADIUS)
        glUseProgram(self.shader_program)
        for unit, (source, target) in enumerate(zip(sources[:3], targets[:3])):
            self.bind_texture(unit, source, f"uSource{unit}")
            self.bind_image_write(unit, target)
        glUniform1i(self.get_uniform_loc("uCount"), min(len(sources), 3))
        glUniform1i(self.get_uniform_loc("uRadius"), radius)
        padded: list[float] = (list(weights[:radius + 1]) + [0.0] * (self.MAX_RADIUS + 1))[:self.MAX_RADIUS + 1]
        glUniform1fv(self.get_uniform_loc("uWeights"), self.MAX_RADIUS + 1, padded)
        width, height = targets[0].width, targets[0].height
        glUniform2i(self.get_uniform_loc("uSize"), width, height)
        self.dispatch(width, height)
        glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT | GL_FRAMEBUFFER_BARRIER_BIT)    # read next by textures and FBOs
