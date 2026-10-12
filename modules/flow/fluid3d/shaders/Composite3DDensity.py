"""Composite3DDensity - Composite the GL_TEXTURE_2D_ARRAY density volume into 2D output.

Front-to-back volumetric march with depth cues: per-channel Beer-Lambert
occlusion, additive glow mix (glow = 1 reproduces the legacy additive sum),
per-layer parallax/perspective from a virtual eye, water tint + fog, and an
optional depth-graded blur. Debug mode shows the dominant layer as hue.
"""
from __future__ import annotations
from pathlib import Path

from OpenGL.GL import *  # type: ignore

from modules.gl import Texture, Texture2DArray, ComputeShader


class Composite3DDensity(ComputeShader):
    """Composite density volume (GL_TEXTURE_2D_ARRAY) into 2D output."""

    def __init__(self) -> None:
        super().__init__()
        self.compute_file_path = Path(__file__).parent / "composite3d_density.comp"

    def use(self, density: Texture2DArray, output_2d: Texture, *,
            emission: float, gamma: float, absorption: float, tint: float, fog: float,
            glow: float, eye: tuple[float, float], perspective: float,
            depth_blur: float, debug_depth: bool) -> None:
        """Composite the density array volume into the 2D output.

        Ray-march step count is derived in the shader (2x the layer count —
        the manual Z-lerp reveals nothing beyond that).

        Args:
            density: Density volume as Texture2DArray (RGBA16F)
            output_2d: 2D output texture (RGBA16F)
            emission: Ink brightness gain
            gamma: Density response exponent (>1 hides thin haze, pops dense cores)
            absorption: Extinction per unit density
            tint: 0..1 mix toward per-channel water extinction (red dies first)
            fog: Medium extinction over the full depth
            glow: 0 = occluded volumetric, 1 = legacy additive
            eye: Virtual eye offset in UV units (parallax/perspective origin)
            perspective: Per-layer shrink toward the vanishing point
            depth_blur: Jitter radius in texels at the back layer (0 = off)
            debug_depth: Show dominant depth layer as hue instead
        """
        if not self.allocated or not self.shader_program:
            return

        glUseProgram(self.shader_program)

        self.bind_texture_2d_array(0, density, "uDensity")
        self.bind_image_write(0, output_2d, GL_RGBA16F)

        glUniform1i(self.get_uniform_loc("uDepth"), density.depth)
        glUniform2i(self.get_uniform_loc("uOutputSize"), output_2d.width, output_2d.height)
        glUniform1f(self.get_uniform_loc("uEmission"), emission)
        glUniform1f(self.get_uniform_loc("uGamma"), gamma)
        glUniform1f(self.get_uniform_loc("uAbsorption"), absorption)
        glUniform1f(self.get_uniform_loc("uTint"), tint)
        glUniform1f(self.get_uniform_loc("uFog"), fog)
        glUniform1f(self.get_uniform_loc("uGlow"), glow)
        glUniform2f(self.get_uniform_loc("uEye"), eye[0], eye[1])
        glUniform1f(self.get_uniform_loc("uPerspective"), perspective)
        glUniform1f(self.get_uniform_loc("uDepthBlur"), depth_blur)
        glUniform1i(self.get_uniform_loc("uDebugDepth"), int(debug_depth))

        self.dispatch(output_2d.width, output_2d.height)
