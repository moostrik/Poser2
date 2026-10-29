"""Magnitude.

Extracts vector magnitude from any vector field (RGBA, RGB, RG).
Outputs scalar magnitude to R channel.

Ported from ofxFlowTools ftVelocityMaskFlow.h
"""
from OpenGL.GL import *  # type: ignore

from modules.gl import Texture
from .. import FlowBase, FlowUtil
from ..shaders import Magnitude as MagnitudeShader


class Magnitude(FlowBase):
    """Extracts vector magnitude from any vector field.

    Pipeline:
        1. Receive vector input via set_input() → input_fbo (configurable format)
        2. Extract magnitude: length(vector) → R channel
        3. Output R32F magnitude via .output property

    Data flow:
        Vector Field → set_input() → input_fbo (RGBA/RGB/RG32F)
        Extract magnitude → output_fbo (R32F)
    """

    def __init__(self, input_format: int = GL_RGBA16F) -> None:
        super().__init__()

        # Define internal formats (input configurable, output always R32F)
        self._input_internal_format = input_format
        self._output_internal_format = GL_R32F   # Scalar magnitude output

        # Shader
        self._magnitude_shader: MagnitudeShader = MagnitudeShader()

    @property
    def magnitude(self) -> Texture:
        """R32F magnitude output (main result)."""
        return self._output

    @property
    def vector_input(self) -> Texture:
        """Vector field input buffer."""
        return self._input

    def set_input(self, tex: Texture) -> None:
        """Set vector field input."""
        FlowUtil.blit(self._input_fbo, tex)

    def allocate(self, width: int, height: int, output_width: int | None = None, output_height: int | None = None) -> None:
        """Allocate magnitude extraction FBOs."""
        super().allocate(width, height, output_width, output_height)
        self._magnitude_shader.allocate()

    def deallocate(self) -> None:
        """Release all resources."""
        super().deallocate()
        self._magnitude_shader.deallocate()

    def reset(self) -> None:
        """Reset all FBOs to zero."""
        super().reset()

    def update(self, delta_time: float = 1.0) -> None:
        """Update magnitude extraction."""
        if not self._allocated:
            return

        self._magnitude_shader.reload()

        # Extract vector magnitude to R channel
        self._output_fbo.begin()
        self._magnitude_shader.use(self._input.texture)
        self._output_fbo.end()

        # FlowUtil.magnitude(self._output_fbo, self._input.texture)


class VelocityMagnitude(Magnitude):
    """Convenience class: Magnitude preset for velocity fields (RG32F)."""

    def __init__(self) -> None:
        super().__init__(GL_RG32F)

    def set_velocity(self, velocity: Texture) -> None:
        """Alias for .set_input (velocity-specific terminology)."""
        self.set_input(velocity)
