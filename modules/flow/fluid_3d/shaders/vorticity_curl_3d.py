"""VorticityCurl3D - Compute 3D curl (vorticity vector) of velocity field."""
from __future__ import annotations

from OpenGL.GL import GL_RGBA16F, glUniform1f, glUniform1i, glUniform3fv, glUseProgram

from modules.gl import Texture3D, ComputeShader
from .face_binding import Cell, bind_faces


class VorticityCurl3D(ComputeShader):
    """Compute 3D curl: ω = ∇ × v (vorticity is a vector in 3D)."""

    WORKGROUP_SIZE_X = 8
    WORKGROUP_SIZE_Y = 8
    WORKGROUP_SIZE_Z = 4

    def __init__(self) -> None:
        super().__init__()

    MAX_CURLS: int = 3      # MAX_CURLS in the shader

    def use(self, faces: Texture3D, obstacle: Texture3D, curls: list[Texture3D], cell: Cell, radii: list[float],
            has_obstacles: bool = True, closed: tuple[bool, bool, bool] = (True, True, True), wall_keep: float = 1.0) -> None:
        """The velocity's curl at each radius, into its own volume, in one pass.

        Args:
            faces: The velocity's faces (`FaceVelocity`)
            obstacle: 3D obstacle mask (R8)
            curls: The curl volumes (RGBA16F: xyz the curl vector), one to MAX_CURLS
            cell: A cell's sides in widths, x, y and z
            radii: Each curl's radius, widths
            has_obstacles: Whether obstacle logic is active
            closed: Per axis, whether the box's walls are closed (walls.glsl)
            wall_keep: The velocity's share along a closed wall kept a step; 1 free-slip
        """
        count: int = min(len(curls), len(radii), self.MAX_CURLS)
        if not self.allocated or not self.shader_program or count == 0:
            return

        glUseProgram(self.shader_program)

        cells = (curls[0].width, curls[0].height, curls[0].depth)
        bind_faces(self, faces, cells, cell, closed, has_obstacles, obstacle)
        for unit in range(count):
            self.bind_image_3d_write(unit, curls[unit], GL_RGBA16F)

        # each radius in cells per axis, at least one: the curl cannot be measured closer
        half_rdx_inv: list[float] = []
        reaches: list[float] = []
        for k in range(self.MAX_CURLS):
            radius: float = radii[min(k, count - 1)]
            reach = [max(radius / side, 1.0) for side in cell]
            reaches += reach
            half_rdx_inv += [0.5 / (side * r) for side, r in zip(cell, reach)]
        glUniform3fv(self.get_uniform_loc("uHalfRdxInv"), self.MAX_CURLS, half_rdx_inv)
        glUniform3fv(self.get_uniform_loc("uRadius"), self.MAX_CURLS, reaches)
        glUniform1i(self.get_uniform_loc("uCount"), count)
        glUniform1f(self.get_uniform_loc("uWallKeep"), wall_keep)

        self.dispatch(*cells)
