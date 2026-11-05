"""Lucas–Kanade shader: one mip level of the displacement between two luminance frames."""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE2, GL_TEXTURE3, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class LucasKanadeShader(Shader):
    """Writes the displacement so far, refined at one mip level, in uv per frame (RG)."""

    def use(self, previous: Texture, current: Texture, estimate: Texture, level: int, window: int, damping: float,
            depth: Texture | None = None, tolerance: float = 0.0) -> None:
        """Refine `estimate` at mip `level`.

        Args:
            previous: Previous luminance frame, mipmapped
            current: Current luminance frame, mipmapped
            estimate: The displacement from the coarser levels, uv per frame
            level: The mip level to solve at
            window: Window radius in texels of that level
            damping: Added to the window's gradient sums, so a flat window solves to no motion
            depth: Metres (R); when given, only the samples within `tolerance` of the centre's depth count
            tolerance: Same-surface depth difference (m), with depth
        """
        if not self.allocated or not self.shader_program:
            return
        if not previous.allocated or not current.allocated or not estimate.allocated:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, previous.tex_id)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, current.tex_id)
        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, estimate.tex_id)

        glUniform1i(self.get_uniform_loc("previous"), 0)
        glUniform1i(self.get_uniform_loc("current"), 1)
        glUniform1i(self.get_uniform_loc("estimate"), 2)
        glUniform1i(self.get_uniform_loc("level"), level)
        glUniform1i(self.get_uniform_loc("window"), window)
        glUniform1f(self.get_uniform_loc("damping"), damping)
        same_surface: bool = depth is not None and depth.allocated
        if same_surface:
            assert depth is not None
            glActiveTexture(GL_TEXTURE3)
            glBindTexture(GL_TEXTURE_2D, depth.tex_id)
        glUniform1i(self.get_uniform_loc("depth"), 3)
        glUniform1i(self.get_uniform_loc("same_surface"), int(same_surface))
        glUniform1f(self.get_uniform_loc("tolerance"), tolerance)

        draw_quad()
