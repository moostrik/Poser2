"""Density To RGB 2D shader: the 2D density as the picture, RGB premultiplied, from Oklab where it is held so."""

from OpenGL.GL import glUniform1i, glUseProgram
from modules.gl import Shader, Texture, draw_quad


class DensityToRgb2D(Shader):
    """Writes the density as RGB, premultiplied by its amount."""

    def use(self, density: Texture, oklab: bool) -> None:
        if not self.allocated or not self.shader_program or not density.allocated:
            return
        glUseProgram(self.shader_program)
        density.bind(0)
        glUniform1i(self.get_uniform_loc("uOklab"), int(oklab))
        draw_quad()
