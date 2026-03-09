"""Buoyancy3DDensity - Buoyancy force with density as GL_TEXTURE_2D_ARRAY.

Subclass of Buoyancy3D that binds density as sampler2DArray instead of sampler3D.
Temperature and obstacle remain sampler3D.
"""
from __future__ import annotations
from pathlib import Path

from OpenGL.GL import *  # type: ignore

from modules.gl.Texture3D import Texture3D
from modules.gl.Texture2DArray import Texture2DArray
from ...fluid3d.shaders.Buoyancy3D import Buoyancy3D


class Buoyancy3DDensity(Buoyancy3D):
    """Compute 3D buoyancy force with density stored as GL_TEXTURE_2D_ARRAY."""

    def __init__(self) -> None:
        super().__init__()
        self.compute_file_path = Path(__file__).parent / "buoyancy3d_density.comp"

    def use(self, temperature: Texture3D, density: Texture2DArray,
            obstacle: Texture3D, force_out: Texture3D,
            sigma: float, kappa: float, ambient_temperature: float,
            has_obstacles: bool = True) -> None:
        """Compute buoyancy force with density as 2D array.

        Args:
            temperature: 3D temperature field (Texture3D — sampler3D)
            density: Density field (Texture2DArray — sampler2DArray)
            obstacle: 3D obstacle mask (Texture3D — sampler3D)
            force_out: Output force volume (Texture3D — image3D)
            sigma: Thermal buoyancy coefficient
            kappa: Density weight coefficient
            ambient_temperature: Reference temperature
            has_obstacles: Whether obstacle logic is active
        """
        if not self.allocated or not self.shader_program:
            return

        glUseProgram(self.shader_program)

        # Temperature and obstacle stay sampler3D
        self.bind_texture_3d(0, temperature, "uTemperature")
        # Density as sampler2DArray
        self.bind_texture_2d_array(1, density, "uDensity")
        self.bind_texture_3d(2, obstacle, "uObstacle")
        self.bind_image_3d_write(0, force_out, GL_RGBA16F)

        glUniform1f(self.get_uniform_loc("uSigma"), sigma)
        glUniform1f(self.get_uniform_loc("uKappa"), kappa)
        glUniform1f(self.get_uniform_loc("uAmbientTemperature"), ambient_temperature)
        glUniform3i(self.get_uniform_loc("uSize"),
                    force_out.width, force_out.height, force_out.depth)
        glUniform1i(self.get_uniform_loc("uHasObstacles"), int(has_obstacles))
        glUniform1i(self.get_uniform_loc("uDensityLayers"), density.depth)

        self.dispatch(force_out.width, force_out.height, force_out.depth)
