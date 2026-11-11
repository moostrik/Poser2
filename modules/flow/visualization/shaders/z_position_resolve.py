"""Z Position Resolve shader: a z position as a monitor draws it."""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class ZPositionResolve(Shader):
    """Writes the z position (red over green) to R and its weight to G."""

    def use(self, z_position: Texture) -> None:
        if not self.allocated or not self.shader_program or not z_position.allocated:
            return
        glUseProgram(self.shader_program)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, z_position.tex_id)
        glUniform1i(self.get_uniform_loc("tex0"), 0)
        draw_quad()
