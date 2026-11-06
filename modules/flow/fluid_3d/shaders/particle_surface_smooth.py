"""ParticleSurfaceSmooth - one direction of a separable bilateral filter on the spheres' depth."""
from __future__ import annotations

from OpenGL.GL import GL_R32F, glUniform1f, glUniform1i, glUniform2i, glUseProgram

from modules.gl import ComputeShader, Texture


class ParticleSurfaceSmooth(ComputeShader):
    """Smooth the depth along one direction, by distance and by depth difference, keeping the edges against
    empty water; a horizontal and a vertical run make a pass."""

    def use(self, depth: Texture, result: Texture, horizontal: bool, radius: int, spatial: float,
            depth_range: float) -> None:
        if not self.allocated or not self.shader_program:
            return
        glUseProgram(self.shader_program)
        self.bind_texture(0, depth, "uDepth")
        self.bind_image_write(0, result, GL_R32F)
        glUniform2i(self.get_uniform_loc("uSize"), result.width, result.height)
        glUniform2i(self.get_uniform_loc("uDirection"), 1 if horizontal else 0, 0 if horizontal else 1)
        glUniform1i(self.get_uniform_loc("uRadius"), radius)
        glUniform1f(self.get_uniform_loc("uSpatial"), spatial)
        glUniform1f(self.get_uniform_loc("uRange"), depth_range)
        self.dispatch(result.width, result.height)
