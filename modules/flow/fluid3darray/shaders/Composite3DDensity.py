"""Composite3DDensity - Composite density stored as GL_TEXTURE_2D_ARRAY into 2D output.

Subclass of Composite3D that binds density as sampler2DArray instead of sampler3D.
"""
from __future__ import annotations
from pathlib import Path

from OpenGL.GL import *  # type: ignore

from modules.gl import Texture, Texture2DArray
from ...fluid3d.shaders.Composite3D import Composite3D


class Composite3DDensity(Composite3D):
    """Composite density volume (GL_TEXTURE_2D_ARRAY) into 2D output."""

    def __init__(self) -> None:
        super().__init__()
        self.compute_file_path = Path(__file__).parent / "composite3d_density.comp"

    def use(self, density: Texture2DArray, output_2d: Texture,
            mode: int = 0, absorption: float = 4.0,
            ray_steps: int = 32) -> None:
        """Composite density 2D array volume into 2D output.

        Args:
            density: Density volume as Texture2DArray
            output_2d: 2D output texture
            mode: Composite mode (0=alpha, 1=additive, 2=max, 3=emission-absorption, 4=debug)
            absorption: Beer's law absorption coefficient
            ray_steps: Number of ray-march steps
        """
        if not self.allocated or not self.shader_program:
            return

        glUseProgram(self.shader_program)

        # Bind density as sampler2DArray
        self.bind_texture_2d_array(0, density, "uDensity")

        # Bind 2D output as image
        self.bind_image_write(0, output_2d, GL_RGBA16F)

        # Uniforms
        glUniform1i(self.get_uniform_loc("uMode"), mode)
        glUniform1i(self.get_uniform_loc("uDepth"), density.depth)
        glUniform2i(self.get_uniform_loc("uOutputSize"), output_2d.width, output_2d.height)
        glUniform1f(self.get_uniform_loc("uAbsorption"), absorption)
        glUniform1i(self.get_uniform_loc("uRaySteps"), ray_steps)

        # Dispatch 2D
        self.dispatch(output_2d.width, output_2d.height)
