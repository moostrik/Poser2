"""VorticityForce shader - the vorticity confinements' force × dt, summed."""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1fv, glUniform1i, glUniform2f, glUseProgram
from modules.gl import Shader, draw_quad, Texture

from .vorticity_curl import VorticityCurl


class VorticityForce(Shader):
    """The confinements' force × dt, ε h (N × ω) Δt each from its own curl's channel, summed."""

    def use(self, curl: Texture, obstacle: Texture, cell: tuple[float, float], timesteps: list[float],
            has_obstacles: bool = True) -> None:
        """Compute the confinements' force × dt.

        Args:
            curl: Each confinement's curl, one channel each (RGBA16F, CLAMP_TO_EDGE)
            obstacle: Obstacle mask (R8, CLAMP_TO_BORDER=1)
            cell: A cell's width and height, widths
            timesteps: Each confinement's strength epsilon × the cell's width × dt, 1 to VorticityCurl.MAX_CURLS
            has_obstacles: When False, skip obstacle texture reads for performance
        """
        if not self.allocated or not self.shader_program or not timesteps:
            return
        count: int = min(len(timesteps), VorticityCurl.MAX_CURLS)
        glUseProgram(self.shader_program)
        glUniform1i(self.get_uniform_loc("uHasObstacles"), int(has_obstacles))

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, curl.tex_id)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, obstacle.tex_id)

        glUniform2f(self.get_uniform_loc("uHalfRdxInv"), 0.5 / cell[0], 0.5 / cell[1])
        glUniform1i(self.get_uniform_loc("uCount"), count)
        padded: list[float] = (list(timesteps[:count]) + [0.0] * VorticityCurl.MAX_CURLS)[:VorticityCurl.MAX_CURLS]
        glUniform1fv(self.get_uniform_loc("uTimestep"), VorticityCurl.MAX_CURLS, padded)

        draw_quad()
