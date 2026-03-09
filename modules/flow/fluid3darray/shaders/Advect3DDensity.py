"""Advect3DDensity - Density advection using GL_TEXTURE_2D_ARRAY.

Subclass of Advect3D that overrides bindings to use sampler2DArray / image2DArray
for density, while keeping velocity and obstacle as sampler3D.
"""
from __future__ import annotations
from pathlib import Path

from OpenGL.GL import *  # type: ignore

from modules.gl.Texture3D import Texture3D
from modules.gl.Texture2DArray import Texture2DArray
from modules.gl.ComputeShader import ComputeShader
from ...fluid3d.shaders.Advect3D import Advect3D


class Advect3DDensity(Advect3D):
    """3D Semi-Lagrangian advection for density stored as GL_TEXTURE_2D_ARRAY.

    Velocity and obstacle remain sampler3D.
    Source/result are sampler2DArray / image2DArray with manual Z interpolation.
    """

    def __init__(self) -> None:
        super().__init__()
        # Override compute shader path to density-specific variant
        self.compute_file_path = Path(__file__).parent / "advect3d_density.comp"

    def advect(self, source_read: Texture2DArray, result_write: Texture2DArray,
               velocity: Texture3D, obstacle: Texture3D,
               aspect: float, depth_scale: float, timestep: float, dissipation: float,
               internal_format: int = GL_RGBA16F,
               has_obstacles: bool = True) -> None:
        """Advect density from source_read into result_write (ping-pong).

        Args:
            source_read: Density back buffer (Texture2DArray — sampler2DArray)
            result_write: Density front buffer (Texture2DArray — image2DArray)
            velocity: 3D velocity field (Texture3D — sampler3D)
            obstacle: 3D obstacle mask (Texture3D — sampler3D)
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

        # Velocity and obstacle stay sampler3D
        self.bind_texture_3d(0, velocity, "uVelocity")
        # Density source as sampler2DArray
        self.bind_texture_2d_array(1, source_read, "uSource")
        # Obstacle stays sampler3D
        self.bind_texture_3d(2, obstacle, "uObstacle")

        # Density output as image2DArray
        self.bind_image_2d_array_write(0, result_write, internal_format)

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
