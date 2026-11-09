"""Advect3D - 3D Semi-Lagrangian advection compute shader."""
from __future__ import annotations

from OpenGL.GL import GL_RGBA16F, glUniform1f, glUniform1i, glUniform3f, glUniform3i, glUseProgram

from modules.gl import Texture3D, ComputeShader
from ...finite_guard import GuardSlot


class Advect3D(ComputeShader):
    """3D Semi-Lagrangian advection with trilinear interpolation.

    Backward-traces through the 3D velocity field and samples the source
    volume using hardware trilinear filtering. Border colors provide
    implicit boundary conditions. The field may sit on a finer grid than
    the velocity: the velocity and the obstacle are sampled by uv.
    """

    WORKGROUP_SIZE_X = 8
    WORKGROUP_SIZE_Y = 4
    WORKGROUP_SIZE_Z = 8

    def __init__(self) -> None:
        super().__init__()

    def advect(self, source_read: Texture3D, result_write: Texture3D,
               velocity: Texture3D, obstacle: Texture3D,
               aspect: float, z_extent: float, timestep: float, dissipation: float,
               internal_format: int = GL_RGBA16F,
               has_obstacles: bool = True, evaporation: float = 0.0, measure: int = 0,
            guard: GuardSlot | None = None,
            closed: tuple[bool, bool, bool] = (True, True, True), faces: Texture3D | None = None) -> None:
        """Advect from source_read into result_write (separate read/write buffers).

        Use this for ping-pong: read from back buffer, write to front buffer.

        Args:
            source_read: Source field to read from (back buffer)
            result_write: Destination to write to (front buffer)
            velocity: 3D velocity field
            obstacle: 3D obstacle mask
            aspect: Width over height: a width of y crosses this share of the uv
            z_extent: Slab thickness in sim-width units (uRdx.z = 1/z_extent)
            timestep: dt * speed
            dissipation: Decay multiplier
            internal_format: Image format for output (default RGBA16F)
            has_obstacles: Whether obstacle logic is active
            evaporation: The size taken off this step, every channel scaled with it
            measure: How the size is read: 0 a premultiplied density's alpha, 1 a vector's length, 2 |r|
            guard: The field's slot in its stage's FiniteGuard: a non-finite value 0 and counted; None no guard
            closed: Per axis, whether the box's walls are closed (walls.glsl)
            faces: The velocity's faces (`FaceVelocity`), read instead of `velocity` for a field on the
                simulation's grid; None reads `velocity`
        """
        if not self.allocated or not self.shader_program:
            return

        glUseProgram(self.shader_program)

        # Bind input volumes as samplers (trilinear filtering)
        self.bind_texture_3d(0, velocity, "uVelocity")
        self.bind_texture_3d(1, source_read, "uSource")
        self.bind_texture_3d(2, obstacle, "uObstacle")

        # Bind output as 3D image (separate from input)
        self.bind_image_3d_write(0, result_write, internal_format)

        # Uniforms
        glUniform1f(self.get_uniform_loc("uTimestep"), timestep)
        rdx_y = aspect          # a width per second of y crosses width / height of the uv
        rdx_z = 1.0 / z_extent if z_extent > 0.0 else 1.0
        glUniform3f(self.get_uniform_loc("uRdx"), 1.0, rdx_y, rdx_z)
        glUniform1f(self.get_uniform_loc("uDissipation"), dissipation)
        glUniform1f(self.get_uniform_loc("uEvaporation"), evaporation)
        glUniform1i(self.get_uniform_loc("uMeasure"), measure)
        glUniform3i(self.get_uniform_loc("uClosed"), *(int(c) for c in closed))
        glUniform1i(self.get_uniform_loc("uFromFaces"), int(faces is not None))
        if faces is not None:
            self.bind_texture_3d(4, faces, "uFaces")
            glUniform3i(self.get_uniform_loc("uCells"), faces.width - 1, faces.height - 1, faces.depth - 1)
        if guard is not None:
            guard.use(self.get_uniform_loc("uGuardSlot"))
        else:
            glUniform1i(self.get_uniform_loc("uGuardSlot"), -1)
        glUniform3i(self.get_uniform_loc("uSize"),
                    result_write.width, result_write.height, result_write.depth)
        glUniform1i(self.get_uniform_loc("uHasObstacles"), int(has_obstacles))

        self.dispatch(result_write.width, result_write.height, result_write.depth)
