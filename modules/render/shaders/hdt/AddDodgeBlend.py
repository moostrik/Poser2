"""AddDodgeBlend shader - Blend foreground onto colored mask with add-to-dodge transition."""

from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture


class AddDodgeBlend(Shader):
    """Blend foreground onto a pre-colored mask, transitioning from additive to dodge.

    Base is the own camera's tinted mask. Foreground (already hue-shifted)
    adds energy/brightness, shifting from additive to color dodge.
    """

    def __init__(self) -> None:
        super().__init__()

    def use(
        self,
        base: Texture,
        frg: Texture,
        strength: float,
        dodge_intensity: float = 0.5,
        add_curve: float = 2.0,
        dodge_curve: float = 1.5,
        opacity_curve: float = 0.3
    ) -> None:
        """Apply add-to-dodge blend.

        Args:
            base: Own camera's tinted mask (RGBA from Tint pass)
            frg: Foreground texture (cel-shaded + hue-shifted, with alpha)
            strength: Blend strength (0 = base only, 1 = full dodge)
            dodge_intensity: How strongly foreground dodges base (0-1)
            add_curve: Additive falloff exponent (higher = stays longer)
            dodge_curve: Dodge ramp exponent (higher = kicks in later)
            opacity_curve: Foreground visibility ramp (lower = appears faster)
        """
        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, base.tex_id)
        glUniform1i(self.get_uniform_loc("uBase"), 0)

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, frg.tex_id)
        glUniform1i(self.get_uniform_loc("uFrg"), 1)

        glUniform1f(self.get_uniform_loc("uStrength"), strength)
        glUniform1f(self.get_uniform_loc("uDodgeIntensity"), dodge_intensity)
        glUniform1f(self.get_uniform_loc("uAddCurve"), add_curve)
        glUniform1f(self.get_uniform_loc("uDodgeCurve"), dodge_curve)
        glUniform1f(self.get_uniform_loc("uOpacityCurve"), opacity_curve)

        draw_quad()
