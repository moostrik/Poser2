"""AddDodgeBlend shader - Two-phase blend from additive to color dodge with foreground reveal."""

from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture


class AddDodgeBlend(Shader):
    """Two-phase blend transitioning from additive to color dodge.

    Phase 1 (Additive): Brightens base by adding foreground. Fades as strength increases.
    Phase 2 (Color Dodge): Vibrant highlights. Kicks in later, ramps up with strength.
    Foreground Reveal: At full strength, foreground layer visible on top (masked by alpha).

    Phases overlap for smooth crossfade, avoiding sudden transitions.
    """

    def __init__(self) -> None:
        super().__init__()

    def use(self, base: Texture, frg: Texture, blend: float) -> None:
        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, base.tex_id)
        glUniform1i(self.get_uniform_loc("uTex0"), 0)

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, frg.tex_id)
        glUniform1i(self.get_uniform_loc("uTex1"), 1)

        glUniform1f(self.get_uniform_loc("uBlend"), blend)

        draw_quad()
