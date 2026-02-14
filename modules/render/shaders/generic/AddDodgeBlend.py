"""AddDodgeBlend shader - Transition from additive to color dodge lighting."""

from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture


class AddDodgeBlend(Shader):
    """Blend foreground with track color, transitioning from additive to dodge.

    Starts with bright additive blending, gradually shifts to vibrant color dodge.
    """

    def __init__(self) -> None:
        super().__init__()

    def use(
        self,
        base: Texture,
        frg: Texture,
        mask: Texture,
        track_r: float,
        track_g: float,
        track_b: float,
        strength: float
    ) -> None:
        """Apply add-to-dodge blend.

        Args:
            base: Base texture (ms_mask output)
            frg: Foreground texture
            mask: Mask texture for this camera (R16F)
            track_r, track_g, track_b: Track color RGB
            strength: Blend strength (0 = base only, 1 = full effect)
        """
        if not self.allocated or not self.shader_program:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        if base.allocated:
            glBindTexture(GL_TEXTURE_2D, base.tex_id)
        glUniform1i(self.get_uniform_loc("uBase"), 0)

        glActiveTexture(GL_TEXTURE1)
        if frg.allocated:
            glBindTexture(GL_TEXTURE_2D, frg.tex_id)
        glUniform1i(self.get_uniform_loc("uFrg"), 1)

        glActiveTexture(GL_TEXTURE2)
        if mask.allocated:
            glBindTexture(GL_TEXTURE_2D, mask.tex_id)
        glUniform1i(self.get_uniform_loc("uMask"), 2)

        glUniform3f(self.get_uniform_loc("uTrackColor"), track_r, track_g, track_b)
        glUniform1f(self.get_uniform_loc("uStrength"), strength)

        draw_quad()
