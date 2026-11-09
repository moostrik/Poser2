"""LightBlur3D - the light grid blurred one axis a pass, for multiple scattering."""
from __future__ import annotations

from OpenGL.GL import GL_RGBA16F, glUniform1f, glUniform1i, glUniform3i, glUseProgram

from modules.gl import Texture3D, ComputeShader


class LightBlur3D(ComputeShader):
    """One axis of a separable 5-tap blur of the light; the last axis mixes it into the direct light."""

    WORKGROUP_SIZE_X = 8
    WORKGROUP_SIZE_Y = 8
    WORKGROUP_SIZE_Z = 4

    def use(self, source: Texture3D, direct: Texture3D, result: Texture3D, axis: int, scattering: float) -> None:
        if not self.allocated or not self.shader_program:
            return
        glUseProgram(self.shader_program)
        self.bind_texture_3d(0, source, "uSource")
        self.bind_texture_3d(1, direct, "uDirect")
        self.bind_image_3d_write(0, result, GL_RGBA16F)
        glUniform3i(self.get_uniform_loc("uSize"), result.width, result.height, result.depth)
        glUniform1i(self.get_uniform_loc("uAxis"), axis)
        glUniform1f(self.get_uniform_loc("uScattering"), scattering)
        self.dispatch(result.width, result.height, result.depth)
