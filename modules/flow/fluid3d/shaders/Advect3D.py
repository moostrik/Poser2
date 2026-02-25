"""Advect3D - 3D Semi-Lagrangian advection compute shader."""
from __future__ import annotations

from OpenGL.GL import *  # type: ignore

from modules.gl.Texture3D import Texture3D
from modules.gl.ComputeShader import ComputeShader


class Advect3D(ComputeShader):
    """3D Semi-Lagrangian advection with trilinear interpolation.

    Backward-traces through the 3D velocity field and samples the source
    volume using hardware trilinear filtering. Border colors provide
    implicit boundary conditions.
    """

    WORKGROUP_SIZE_X = 16
    WORKGROUP_SIZE_Y = 16
    WORKGROUP_SIZE_Z = 1

    def __init__(self) -> None:
        super().__init__()

    def use(self, source: Texture3D, velocity: Texture3D, obstacle: Texture3D,
            aspect: float, depth_scale: float, timestep: float, dissipation: float,
            has_obstacles: bool = True) -> None:
        """Apply 3D advection.

        Args:
            source: Field to advect (density, velocity, temperature, pressure)
            velocity: 3D velocity field (RGBA16F: xyz = u,v,w)
            obstacle: 3D obstacle mask (R8)
            aspect: Width/height ratio for isotropic XY advection
            depth_scale: Z grid spacing relative to XY (controls volume thickness)
            timestep: dt * speed — advection distance per frame
            dissipation: Exponential decay multiplier
            has_obstacles: Whether obstacle logic is active
        """
        if not self.allocated or not self.shader_program:
            return

        glUseProgram(self.shader_program)

        # Bind input volumes as samplers (trilinear filtering)
        self.bind_texture_3d(0, velocity, "uVelocity")
        self.bind_texture_3d(1, source, "uSource")
        self.bind_texture_3d(2, obstacle, "uObstacle")

        # Bind output as 3D image
        self.bind_image_3d_write(0, source, GL_RGBA16F)

        # Uniforms
        glUniform1f(self.get_uniform_loc("uTimestep"), timestep)
        rdx_y = 1.0 / aspect if aspect > 0.0 else 1.0
        rdx_z = 1.0 / depth_scale if depth_scale > 0.0 else 1.0
        glUniform3f(self.get_uniform_loc("uRdx"), 1.0, rdx_y, rdx_z)
        glUniform1f(self.get_uniform_loc("uDissipation"), dissipation)
        glUniform3i(self.get_uniform_loc("uSize"), source.width, source.height, source.depth)
        glUniform1i(self.get_uniform_loc("uHasObstacles"), int(has_obstacles))

        self.dispatch(source.width, source.height, source.depth)

    def advect(self, source_read: Texture3D, result_write: Texture3D,
               velocity: Texture3D, obstacle: Texture3D,
               aspect: float, depth_scale: float, timestep: float, dissipation: float,
               internal_format: int = GL_RGBA16F,
               has_obstacles: bool = True) -> None:
        """Advect from source_read into result_write (separate read/write buffers).

        Use this for ping-pong: read from back buffer, write to front buffer.

        Args:
            source_read: Source field to read from (back buffer)
            result_write: Destination to write to (front buffer)
            velocity: 3D velocity field
            obstacle: 3D obstacle mask
            aspect: Width/height ratio
            depth_scale: Z grid spacing relative to XY
            timestep: dt * speed
            dissipation: Decay multiplier
            internal_format: Image format for output (default RGBA16F)
            has_obstacles: Whether obstacle logic is active
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
        rdx_y = 1.0 / aspect if aspect > 0.0 else 1.0
        rdx_z = 1.0 / depth_scale if depth_scale > 0.0 else 1.0
        glUniform3f(self.get_uniform_loc("uRdx"), 1.0, rdx_y, rdx_z)
        glUniform1f(self.get_uniform_loc("uDissipation"), dissipation)
        glUniform3i(self.get_uniform_loc("uSize"),
                    result_write.width, result_write.height, result_write.depth)
        glUniform1i(self.get_uniform_loc("uHasObstacles"), int(has_obstacles))

        self.dispatch(result_write.width, result_write.height, result_write.depth)
