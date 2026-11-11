"""VorticityForce3D - 3D vorticity confinement force."""
from __future__ import annotations

from OpenGL.GL import GL_RGBA16F, glUniform1fv, glUniform1i, glUniform3f, glUniform3i, glUseProgram

from modules.gl import Texture3D, ComputeShader


class VorticityForce3D(ComputeShader):
    """The 3D vorticity confinement force, ε h (N × ω) dt, added to the velocity in place."""

    WORKGROUP_SIZE_X = 8
    WORKGROUP_SIZE_Y = 8
    WORKGROUP_SIZE_Z = 4

    def __init__(self) -> None:
        super().__init__()

    MAX_CONFINEMENTS: int = 3       # MAX_CONFINEMENTS in the shader

    def use(self, curls: list[Texture3D], obstacle: Texture3D, force_out: Texture3D, cell: tuple[float, float, float],
            timesteps: list[float], has_obstacles: bool = True) -> None:
        """The confinements' force × dt at the cells, summed.

        Args:
            curls: Each confinement's curl (RGBA16F), one to MAX_CONFINEMENTS
            obstacle: 3D obstacle mask (R8)
            force_out: The force × dt at the cells (RGBA16F), for AddToFaces3D
            cell: A cell's sides in widths, x, y and z
            timesteps: Each one's strength × h × dt
            has_obstacles: Whether obstacle logic is active
        """
        count: int = min(len(curls), len(timesteps), self.MAX_CONFINEMENTS)
        if not self.allocated or not self.shader_program or count == 0:
            return

        glUseProgram(self.shader_program)

        for k, unit in enumerate((0, 2, 3)):
            self.bind_texture_3d(unit, curls[min(k, count - 1)], f"uCurl{k}")     # an unused slot repeats the last
        self.bind_texture_3d(1, obstacle, "uObstacle")
        self.bind_image_3d_write(0, force_out, GL_RGBA16F)

        glUniform3f(self.get_uniform_loc("uHalfRdxInv"), 0.5 / cell[0], 0.5 / cell[1], 0.5 / cell[2])
        glUniform1i(self.get_uniform_loc("uCount"), count)
        glUniform1fv(self.get_uniform_loc("uTimestep"), self.MAX_CONFINEMENTS,
                     (list(timesteps[:count]) + [0.0] * self.MAX_CONFINEMENTS)[:self.MAX_CONFINEMENTS])
        glUniform3i(self.get_uniform_loc("uSize"), force_out.width, force_out.height, force_out.depth)
        glUniform1i(self.get_uniform_loc("uHasObstacles"), int(has_obstacles))

        self.dispatch(force_out.width, force_out.height, force_out.depth)
