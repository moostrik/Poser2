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
    ) -> None:
        """Apply add-to-dodge blend.

        Args:
            base: Own camera's tinted mask (RGBA from Tint pass)
            frg: Foreground texture (cel-shaded + hue-shifted)
            strength: Blend strength (0 = base only, 1 = full effect)
            add_curve: Additive peaks early then fades (higher = stays longer)
            dodge_curve: Dodge ramps up (higher = kicks in later)
            frg_curve: Foreground visibility curve (lower = appears faster)
        """
        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, base.tex_id)
        glUniform1i(self.get_uniform_loc("uBase"), 0)

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, frg.tex_id)
        glUniform1i(self.get_uniform_loc("uFrg"), 1)

        glUniform1f(self.get_uniform_loc("uStrength"), strength)

        draw_quad()
