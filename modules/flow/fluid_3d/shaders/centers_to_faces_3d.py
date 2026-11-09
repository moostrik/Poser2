"""CentersToFaces3D - the velocity at the cells' centers put on their faces (the scaffold)."""
from __future__ import annotations

from OpenGL.GL import glUseProgram

from modules.gl import Texture3D, ComputeShader
from .face_binding import bind_faces, bind_face_writes


class CentersToFaces3D(ComputeShader):
    """Each face the mean of its two cells; 0 where held, an open wall's its one cell's."""

    WORKGROUP_SIZE_X = 8
    WORKGROUP_SIZE_Y = 8
    WORKGROUP_SIZE_Z = 4

    def use(self, centers: Texture3D, faces: Texture3D,
            obstacle: Texture3D, closed: tuple[bool, bool, bool], has_obstacles: bool) -> None:
        if not self.allocated or not self.shader_program:
            return
        glUseProgram(self.shader_program)
        cells = (centers.width, centers.height, centers.depth)
        bind_faces(self, faces, cells, closed, has_obstacles, obstacle)     # the uniforms; the faces are not read
        bind_face_writes(self, faces)
        self.bind_texture_3d(0, centers, 'uCenters')
        self.dispatch(cells[0] + 1, cells[1] + 1, cells[2] + 1)
