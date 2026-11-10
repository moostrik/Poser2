"""ParticleSurfaceShade - the surface's normals, absorption and light into the 2D output."""
from __future__ import annotations

from OpenGL.GL import GL_RGBA16F, glUniform1f, glUniform2i, glUniform3f, glUseProgram

from modules.gl import ComputeShader, Texture


class ParticleSurfaceShade(ComputeShader):
    """Shade the smoothed depth: color through thickness, diffuse, specular, Fresnel, fog."""

    def use(self, depth: Texture, thickness: Texture, color: Texture, output: Texture, *, depth_per_pixel: float,
            emission: float, absorption: float, fog: tuple[float, float, float]) -> None:
        if not self.allocated or not self.shader_program:
            return
        glUseProgram(self.shader_program)
        self.bind_texture(0, depth, "uDepth")
        self.bind_texture(1, thickness, "uThickness")
        self.bind_texture(2, color, "uColor")
        self.bind_image_write(0, output, GL_RGBA16F)
        glUniform2i(self.get_uniform_loc("uSize"), output.width, output.height)
        glUniform1f(self.get_uniform_loc("uDepthPerPixel"), depth_per_pixel)
        glUniform1f(self.get_uniform_loc("uEmission"), emission)
        glUniform1f(self.get_uniform_loc("uAbsorption"), absorption)
        glUniform3f(self.get_uniform_loc("uFog"), *fog)
        self.dispatch(output.width, output.height)
