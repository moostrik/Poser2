"""Z Position Mix shader: an output's z position from its readings'."""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1fv, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class ZPositionMix(Shader):
    """Writes the readings' z positions weighted by their sizes at their shares, red over green."""

    def use(self, readings: list[tuple[Texture, float]], trails: tuple[Texture, Texture, Texture],
            z_positions: tuple[Texture, Texture, Texture],
            depth_mask: tuple[Texture, Texture, float] | None = None) -> None:
        """The z position of an output mixing `readings`.

        Args:
            readings: The six readings in order (trail, change, start, bow, stop, wake), each with its share
            trails: The lowpass's trail, the bandpass's fast and slow trail
            z_positions: Their z positions (red over green)
            depth_mask: The depth mask, the motion's z position now and the depth mask's share; None for none
        """
        if not self.allocated or not self.shader_program or len(readings) != 6:
            return
        glUseProgram(self.shader_program)
        mask_textures: tuple[Texture, Texture] = (depth_mask[0], depth_mask[1]) if depth_mask is not None else (trails[0], z_positions[0])
        unit: int = 0
        for i, (texture, _) in enumerate(readings):
            glActiveTexture(int(GL_TEXTURE0) + unit)
            glBindTexture(GL_TEXTURE_2D, texture.tex_id)
            glUniform1i(self.get_uniform_loc(f"tex{i}"), unit)
            unit += 1
        for name, texture in zip(("trail", "fast", "slow", "trail_z_position", "fast_z_position", "slow_z_position",
                                  "depth_mask", "z_position"), (*trails, *z_positions, *mask_textures)):
            glActiveTexture(int(GL_TEXTURE0) + unit)
            glBindTexture(GL_TEXTURE_2D, texture.tex_id)
            glUniform1i(self.get_uniform_loc(name), unit)
            unit += 1
        glUniform1fv(self.get_uniform_loc("shares"), 6, [share for _, share in readings])
        glUniform1f(self.get_uniform_loc("depth_mask_share"), depth_mask[2] if depth_mask is not None else 0.0)
        draw_quad()
