"""Inject Density 2D shader: the bridge's dye into the 2D density by the blend."""

from OpenGL.GL import glUniform1f, glUniform1i, glUseProgram
from modules.gl import Shader, Texture, draw_quad
from ...density_blend import DensityBlend


class InjectDensity2D(Shader):
    """Writes the density with the color × the amount blended in (`density_blend.glsl`); draw into the density's other
    buffer."""

    def use(self, density: Texture, amount: Texture, color: Texture, strength: float, release: float,
            blend: DensityBlend, gain: float, oklab: bool) -> None:
        """Blend the dye in.

        Args:
            density: The density before (RGBA, premultiplied)
            amount: The bridge's amount (red)
            color: Its color, premultiplied by its alpha, the coverage
            strength: Over's and fill's drag share this step, or add's dt
            release: The release's share this step where the texel falls; 0 = never falls
            blend: Over, add or fill
            gain: The amount times this
            oklab: The density holds Oklab
        """
        if not self.allocated or not self.shader_program:
            return
        if not all(texture.allocated for texture in (density, amount, color)):
            return
        glUseProgram(self.shader_program)
        for unit, texture in enumerate((density, amount, color)):
            texture.bind(unit)
        glUniform1f(self.get_uniform_loc("uStrength"), strength)
        glUniform1f(self.get_uniform_loc("uRelease"), release)
        glUniform1i(self.get_uniform_loc("uOklab"), int(oklab))
        glUniform1i(self.get_uniform_loc("uBlend"), int(blend))
        glUniform1f(self.get_uniform_loc("uGain"), gain)
        draw_quad()
