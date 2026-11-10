"""ParticleComposite - the particles' depth slices marched front to back into the 2D output."""
from __future__ import annotations

from OpenGL.GL import GL_RGBA16F, glUniform1f, glUniform1i, glUniform2i, glUseProgram

from modules.gl import ComputeShader, Texture, Texture2DArray


class ParticleComposite(ComputeShader):
    """March `depth` slices of the slice array into the output: Beer–Lambert, tint, fog."""

    def use(self, slices: Texture2DArray, depth: int, output: Texture, *, emission: float, absorption: float,
            tint: float, fog: float) -> None:
        if not self.allocated or not self.shader_program:
            return
        glUseProgram(self.shader_program)
        self.bind_texture_2d_array(0, slices, "uSlices")
        self.bind_image_write(0, output, GL_RGBA16F)
        glUniform1i(self.get_uniform_loc("uDepth"), depth)
        glUniform2i(self.get_uniform_loc("uOutputSize"), output.width, output.height)
        glUniform1f(self.get_uniform_loc("uEmission"), emission)
        glUniform1f(self.get_uniform_loc("uAbsorption"), absorption)
        glUniform1f(self.get_uniform_loc("uTint"), tint)
        glUniform1f(self.get_uniform_loc("uFog"), fog)
        self.dispatch(output.width, output.height)
