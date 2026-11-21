"""Binding the velocity's faces for the shaders that include `faces.glsl`, each part for the passes that read it."""

from OpenGL.GL import GL_RGBA16F, GL_READ_WRITE, GL_WRITE_ONLY, glUniform1i, glUniform3f, glUniform3i

from modules.gl import ComputeShader, Texture3D

FACE_SAMPLER_UNIT: int = 4      # uFaces
FACE_IMAGE_UNIT: int = 4        # uOutFaces

Cell = tuple[float, float, float]     # a cell's sides in widths, x, y and z: the box's spacing, stretched in z


def bind_cells(shader: ComputeShader, cells: tuple[int, int, int]) -> None:
    """The cell counts (uCells)."""
    glUniform3i(shader.get_uniform_loc('uCells'), *cells)


def bind_faces(shader: ComputeShader, faces: Texture3D, cells: tuple[int, int, int]) -> None:
    """The faces read (`faces.glsl`'s sampler) and the cell counts."""
    faces.bind(FACE_SAMPLER_UNIT)
    bind_cells(shader, cells)


def bind_cell(shader: ComputeShader, cell: Cell) -> None:
    """The cell's sides (uCell), for a pass that differences over them."""
    glUniform3f(shader.get_uniform_loc('uCell'), *cell)


def bind_walls(shader: ComputeShader, closed: tuple[bool, bool, bool]) -> None:
    """Per axis, whether the box's walls are closed (uClosed, walls.glsl)."""
    glUniform3i(shader.get_uniform_loc('uClosed'), *(int(c) for c in closed))


def bind_obstacle(shader: ComputeShader, obstacle: Texture3D, has_obstacles: bool) -> None:
    """The obstacle mask (R8 at the cells) and whether it is read."""
    glUniform1i(shader.get_uniform_loc('uHasObstacles'), int(has_obstacles))
    obstacle.bind(2)


def bind_face_writes(shader: ComputeShader, faces: Texture3D, in_place: bool = False) -> None:
    """The faces written (`store_faces`); read too where the shader works in place (`FACES_IN_PLACE`)."""
    shader.bind_image(FACE_IMAGE_UNIT, faces, GL_READ_WRITE if in_place else GL_WRITE_ONLY, GL_RGBA16F)
