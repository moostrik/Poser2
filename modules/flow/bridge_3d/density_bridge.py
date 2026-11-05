"""Density Bridge.

Combines RGB density with an amount, its alpha, for fluid simulation: the speed on the trail
bridge, the emission on the kinetic bridge.

Ported from ofxFlowTools ftDensityBridgeFlow.h
"""
from OpenGL.GL import GL_R32F, GL_RGBA16F

from modules.gl import Texture, SwapFbo
from modules.settings import Field, BaseSettings
from .. import FlowBase, FlowUtil
from .shaders import DensityBridgeShader
from ..shaders.hsv import HSV


class DensityBridgeSettings(BaseSettings):
    """Configuration for density bridge."""
    saturation = Field(1.2, min=0.0, max=5.0, description="Color saturation boost")
    brightness = Field(1.0, min=0.0, max=2.0, description="Brightness/value adjustment")


class DensityBridge(FlowBase):
    """Density bridge with an alpha channel from an amount.

    Pipeline:
        1. Receive RGB density via set_color() → input_fbo (RGBA16F)
        2. Receive the amount via set_alpha() → alpha_fbo (R32F)
        3. Combine density RGB + amount → RGBA with alpha
        4. Apply HSV adjustments (saturation, brightness)
        5. Output RGBA16F density via .density property
    """

    def __init__(self, settings: DensityBridgeSettings | None = None) -> None:
        super().__init__()

        self.settings: DensityBridgeSettings = settings or DensityBridgeSettings()

        # Define internal formats
        self._input_internal_format = GL_RGBA16F   # RGB density input
        self._output_internal_format = GL_RGBA16F  # RGBA density output

        self._alpha_fbo: SwapFbo = SwapFbo()

        # Shaders
        self._density_bridge_shader: DensityBridgeShader = DensityBridgeShader()
        self._hsv_shader: HSV = HSV()

    @property
    def density(self) -> Texture:
        """RGBA density output (main result)."""
        return self._output

    @property
    def color_input(self) -> Texture:
        """RGB density input buffer."""
        return self._input

    @property
    def alpha_input(self) -> Texture:
        """The amount the alpha is made from."""
        return self._alpha_fbo

    def allocate(self, width: int, height: int, output_width: int | None = None, output_height: int | None = None) -> None:
        """Allocate density bridge FBOs.

        Args:
            width: Processing width
            height: Processing height
            output_width: Optional output width (defaults to width)
            output_height: Optional output height (defaults to height)
        """
        super().allocate(width, height, output_width, output_height)
        self._alpha_fbo.allocate(width, height, GL_R32F)
        self._density_bridge_shader.allocate()
        self._hsv_shader.allocate()

    def deallocate(self) -> None:
        """Release all FBO resources."""
        super().deallocate()
        self._alpha_fbo.deallocate()
        self._density_bridge_shader.deallocate()
        self._hsv_shader.deallocate()

    def set_color(self, color: Texture) -> None:
        """Set RGB density input (replaces current)."""
        FlowUtil.blit(self._input_fbo, color)

    def set_alpha(self, amount: Texture) -> None:
        """Set the amount the alpha is made from (red)."""
        FlowUtil.blit(self._alpha_fbo, amount)

    def reset(self) -> None:
        """Reset all FBOs to zero."""
        super().reset()
        FlowUtil.zero(self._alpha_fbo)

    def update(self, speed: float = 1.0) -> None:
        """Update density bridge processing.

        Args:
            speed: Speed multiplier for alpha
        """
        if not self._allocated:
            return

        # Stage 1: Combine density RGB + the amount
        self._output_fbo.begin()
        self._density_bridge_shader.use(
            self._input,                    # Density RGB from input_fbo
            self._alpha_fbo.texture,        # The amount
            speed                             # Speed multiplier for alpha
        )
        self._output_fbo.end()

        # Stage 2: Apply HSV adjustments (saturation + brightness)
        self._output_fbo.swap()
        self._output_fbo.begin()
        self._hsv_shader.use(
            self._output_fbo.back_texture,
            hue=0.0,
            saturation=self.settings.saturation,
            value=self.settings.brightness
        )
        self._output_fbo.end()
