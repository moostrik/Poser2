"""Gradient3D - the pressure's gradient subtracted from the velocity on the cells' faces."""
from __future__ import annotations

from OpenGL.GL import glUniform1f, glUseProgram

from modules.gl import Texture3D, ComputeShader
from .face_binding import bind_faces, bind_face_writes


class Gradient3D(ComputeShader):
    """Each face less its two cells' pressure difference over h (the projection's last pass)."""

    WORKGROUP_SIZE_X = 8
    WORKGROUP_SIZE_Y = 8
    WORKGROUP_SIZE_Z = 4

    def use(self, source: Texture3D, pressure: Texture3D, obstacle: Texture3D, result: Texture3D,
            cell: float, has_obstacles: bool = True, closed: tuple[bool, bool, bool] = (True, True, True)) -> None:
        """v_new = v_old - grad(p) on the faces.

        Args:
            source: The faces before (`FaceVelocity`)
            pressure: The pressure (R32F at the cells)
            obstacle: The obstacle mask (R8 at the cells)
            result: The faces written
            cell: The cubic cell's size in widths
            has_obstacles: Whether obstacle logic is active
            closed: Per axis, whether the walls are closed
        """
        if not self.allocated or not self.shader_program:
            return
        glUseProgram(self.shader_program)
        cells = (pressure.width, pressure.height, pressure.depth)
        bind_faces(self, source, cells, closed, has_obstacles, obstacle)
        bind_face_writes(self, result)
        self.bind_texture_3d(0, pressure, "uPressure")
        glUniform1f(self.get_uniform_loc("uRdxInv"), 1.0 / cell)
        self.dispatch(cells[0] + 1, cells[1] + 1, cells[2] + 1)
