"""VorticityCurl shader - the velocity's curl at each confinement's radius, one channel each."""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUniform2fv, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class VorticityCurl(Shader):
    """The velocity's curl at up to MAX_CURLS radii, each into its own channel (r, g, b) of one texture, in one pass."""

    MAX_CURLS: int = 3       # vorticity_curl.frag's MAX_CURLS: the confinements at once

    def use(self, velocity: Texture, obstacle: Texture, cell: tuple[float, float], radii: list[float],
            has_obstacles: bool = True, closed: bool = True, wall_keep: float = 1.0) -> None:
        """Compute the velocity's curls.

        Args:
            velocity: Velocity field (RG16F), widths/s
            obstacle: Obstacle mask (R8/R32F)
            cell: A cell's width and height, widths
            radii: Each curl's sampling radius, widths, at least one cell; 1 to MAX_CURLS
            has_obstacles: When False, skip obstacle texture reads for performance
            closed: The walls closed, else open (walls_2d.glsl)
            wall_keep: The share of the velocity along a closed wall kept a step; 1 free-slip
        """
        if not self.allocated or not self.shader_program or not radii:
            return
        count: int = min(len(radii), self.MAX_CURLS)
        # each radius in cells per axis, at least one: the curl cannot be measured closer
        reaches: list[float] = []
        half_rdx_inv: list[float] = []
        for k in range(self.MAX_CURLS):
            radius: float = radii[min(k, count - 1)]
            reach = (max(radius / cell[0], 1.0), max(radius / cell[1], 1.0))
            reaches += reach
            half_rdx_inv += [0.5 / (cell[0] * reach[0]), 0.5 / (cell[1] * reach[1])]

        glUseProgram(self.shader_program)
        glUniform1i(self.get_uniform_loc("uHasObstacles"), int(has_obstacles))
        glUniform1i(self.get_uniform_loc("uClosed"), int(closed))
        glUniform1f(self.get_uniform_loc("uWallKeep"), wall_keep)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, velocity.tex_id)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, obstacle.tex_id)

        glUniform1i(self.get_uniform_loc("uCount"), count)
        glUniform2fv(self.get_uniform_loc("uRadius"), self.MAX_CURLS, reaches)
        glUniform2fv(self.get_uniform_loc("uHalfRdxInv"), self.MAX_CURLS, half_rdx_inv)

        draw_quad()
