"""Divergence3D - the divergence of the velocity on the cells' faces."""
from __future__ import annotations

from OpenGL.GL import GL_R16F, glUniform1f, glUseProgram

from modules.gl import Texture3D, ComputeShader
from .face_binding import bind_faces


class Divergence3D(ComputeShader):
    """Per cell, its six faces' one-cell differences over h."""

    WORKGROUP_SIZE_X = 8
    WORKGROUP_SIZE_Y = 8
    WORKGROUP_SIZE_Z = 4

    def use(self, faces: Texture3D, obstacle: Texture3D,
            divergence_out: Texture3D, cell: float, has_obstacles: bool = True,
            closed: tuple[bool, bool, bool] = (True, True, True)) -> None:
        """The velocity's divergence into `divergence_out` (R16F at the cells).

        Args:
            faces: The velocity's faces (`FaceVelocity`)
            obstacle: The obstacle mask (R8 at the cells)
            divergence_out: The divergence, 1/s
            cell: The cubic cell's size in widths
            has_obstacles: Whether obstacle logic is active
            closed: Per axis, whether the walls are closed
        """
        if not self.allocated or not self.shader_program:
            return
        glUseProgram(self.shader_program)
        cells = (divergence_out.width, divergence_out.height, divergence_out.depth)
        bind_faces(self, faces, cells, closed, has_obstacles, obstacle)
        self.bind_image_3d_write(0, divergence_out, GL_R16F)
        glUniform1f(self.get_uniform_loc("uRdxInv"), 1.0 / cell)
        self.dispatch(*cells)
