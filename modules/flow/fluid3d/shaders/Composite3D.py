"""Composite3D - Composite 3D density volume into 2D output."""
from __future__ import annotations

from OpenGL.GL import *  # type: ignore

from modules.gl.Texture import Texture
from modules.gl.Texture3D import Texture3D
from modules.gl.ComputeShader import ComputeShader


class Composite3D(ComputeShader):
    """Composite 3D density volume into 2D image.

    Modes:
        0 = Front-to-back alpha compositing
        1 = Additive blending
        2 = Maximum intensity projection
        3 = Emission-absorption (Beer's law volumetric)
    """

    WORKGROUP_SIZE_X = 16
    WORKGROUP_SIZE_Y = 16
    WORKGROUP_SIZE_Z = 1

    def __init__(self) -> None:
        super().__init__()

    def use(self, density: Texture3D, output_2d: Texture,
            mode: int = 0, absorption: float = 4.0,
            ray_steps: int = 32) -> None:
        """Composite 3D volume to 2D image.

        Args:
            density: 3D density volume (RGBA16F)
            output_2d: 2D output texture (RGBA16F) — written via imageStore
            mode: 0=front-to-back alpha, 1=additive, 2=max intensity,
                  3=emission-absorption (Beer's law)
            absorption: Absorption coefficient for mode 3 (higher = more opaque)
            ray_steps: Number of ray-march steps for mode 3 (more = smoother)
        """
        if not self.allocated or not self.shader_program:
            return

        glUseProgram(self.shader_program)

        # Bind 3D density as sampler (trilinear filtering)
        self.bind_texture_3d(0, density, "uDensity")

        # Bind 2D output as image
        self.bind_image_write(0, output_2d, GL_RGBA16F)

        # Uniforms
        glUniform1i(self.get_uniform_loc("uMode"), mode)
        glUniform1i(self.get_uniform_loc("uDepth"), density.depth)
        glUniform2i(self.get_uniform_loc("uOutputSize"), output_2d.width, output_2d.height)
        glUniform1f(self.get_uniform_loc("uAbsorption"), absorption)
        glUniform1i(self.get_uniform_loc("uRaySteps"), ray_steps)

        # Dispatch 2D (no depth dimension for output)
        self.dispatch(output_2d.width, output_2d.height)
