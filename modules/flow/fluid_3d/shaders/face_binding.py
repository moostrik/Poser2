"""Binding the velocity's faces for the shaders that include `faces.glsl`."""

from OpenGL.GL import GL_RGBA16F, GL_READ_WRITE, GL_WRITE_ONLY, glUniform1i, glUniform3i

from modules.gl import ComputeShader, Texture3D

FACE_SAMPLER_UNIT: int = 4      # uFaces
FACE_IMAGE_UNIT: int = 4        # uOutFaces


def bind_faces(shader: ComputeShader, faces: Texture3D, cells: tuple[int, int, int], closed: tuple[bool, bool, bool],
               has_obstacles: bool, obstacle: Texture3D) -> None:
    """The faces read (`faces.glsl`'s sampler), the cell counts, the walls and the obstacle."""
    shader.bind_texture_3d(FACE_SAMPLER_UNIT, faces, 'uFaces')
    glUniform3i(shader.get_uniform_loc('uCells'), *cells)
    glUniform3i(shader.get_uniform_loc('uClosed'), *(int(c) for c in closed))
    glUniform1i(shader.get_uniform_loc('uHasObstacles'), int(has_obstacles))
    shader.bind_texture_3d(2, obstacle, 'uObstacle')


def bind_face_writes(shader: ComputeShader, faces: Texture3D, in_place: bool = False) -> None:
    """The faces written (`store_faces`); read too where the shader works in place (`FACES_IN_PLACE`)."""
    shader.bind_image_3d(FACE_IMAGE_UNIT, faces, GL_READ_WRITE if in_place else GL_WRITE_ONLY, GL_RGBA16F)
