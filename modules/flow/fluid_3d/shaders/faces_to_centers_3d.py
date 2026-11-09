"""FacesToCenters3D - the velocity at the cells' centers, from their faces."""
from __future__ import annotations

from OpenGL.GL import GL_RGBA16F, glUseProgram

from modules.gl import Texture3D, ComputeShader
from .face_binding import bind_faces


class FacesToCenters3D(ComputeShader):
    """Each component the mean of its cell's two faces, into an RGBA16F volume at the cells."""

    WORKGROUP_SIZE_X = 8
    WORKGROUP_SIZE_Y = 8
    WORKGROUP_SIZE_Z = 4

    def use(self, faces: Texture3D, centers: Texture3D,
            obstacle: Texture3D, closed: tuple[bool, bool, bool], has_obstacles: bool) -> None:
        if not self.allocated or not self.shader_program:
            return
        glUseProgram(self.shader_program)
        cells = (centers.width, centers.height, centers.depth)
        bind_faces(self, faces, cells, closed, has_obstacles, obstacle)
        self.bind_image_3d_write(0, centers, GL_RGBA16F)
        self.dispatch(*cells)
