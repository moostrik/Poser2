"""Temperature Bridge.

Interprets RGB color as warm/neutral/cold and combines with velocity mask.

Color interpretation:
    - R channel = warm/heat
    - G channel = neutral/dampening
    - B channel = cold

Formula: temp = (warm - cold) * (1.0 - neutral * 0.5) * mask * speed

Ported from ofxFlowTools ftTemperatureBridgeFlow.h
"""
from dataclasses import dataclass, field

from OpenGL.GL import *  # type: ignore

from modules.gl import Texture, SwapFbo
from .. import FlowBase, FlowConfigBase, FlowUtil
from .shaders.TemperatureBridge import TemperatureBridge as TemperatureBridgeShader

from modules.utils.HotReloadMethods import HotReloadMethods


@dataclass
class TemperatureBridgeConfig(FlowConfigBase):
    """Configuration for temperature bridge."""

    scale: float = field(
        default=0.5,
        metadata={"min": -1.0, "max": 2.0, "label": "Scale",
                  "description": "Temperature output multiplier"}
    )


class TemperatureBridge(FlowBase):
    """Temperature bridge interprets RGB color and applies velocity mask.

    Color interpretation:
        - R = warm/heat (positive contribution)
        - G = neutral/dampening (reduces temperature)
        - B = cold (negative contribution)

    Formula: temp = (R - B) * (1.0 - G * 0.5) * mask * speed

    Pipeline:
        1. Receive RGB color via set_color() → input_fbo (RGB32F)
        2. Receive velocity mask via set_mask() → mask_fbo (R32F from MaskBridge)
        3. Interpret color and combine with mask
        4. Output R32F temperature via .temperature property

    Data flow:
        RGB Color → set_color() → input_fbo (RGB32F)
        Velocity Mask → set_mask() → mask_fbo (R32F)
        Interpret → TemperatureBridgeShader → output_fbo (R32F)
    """

    def __init__(self, config: TemperatureBridgeConfig | None = None) -> None:
        super().__init__()

        self.config: TemperatureBridgeConfig = config or TemperatureBridgeConfig()

        # Define internal formats
        self._input_internal_format = GL_RGB16F   # RGB color interpretation
        self._output_internal_format = GL_R16F    # R16F temperature output

        # Mask storage
        self._mask_fbo: SwapFbo = SwapFbo()

        # Shader
        self._temperature_bridge_shader: TemperatureBridgeShader = TemperatureBridgeShader()

        hot_reload = HotReloadMethods(self.__class__, True, True)

    @property
    def temperature(self) -> Texture:
        """R32F temperature field output (main result)."""
        return self._output

    @property
    def color_input(self) -> Texture:
        """RGB color interpretation input buffer."""
        return self._input

    @property
    def mask_input(self) -> Texture:
        """Velocity mask input buffer."""
        return self._mask_fbo.texture

    def allocate(self, width: int, height: int, output_width: int | None = None, output_height: int | None = None) -> None:
        """Allocate temperature bridge FBOs.

        Args:
            width: Processing width
            height: Processing height
            output_width: Optional output width (defaults to width)
            output_height: Optional output height (defaults to height)
        """
        super().allocate(width, height, output_width, output_height)
        self._mask_fbo.allocate(width, height, GL_R32F)
        self._temperature_bridge_shader.allocate()

    def deallocate(self) -> None:
        """Release all FBO resources."""
        super().deallocate()
        self._mask_fbo.deallocate()
        self._temperature_bridge_shader.deallocate()

    def set_color(self, color: Texture) -> None:
        """Set RGB color interpretation input.

        R=warm/heat, G=neutral/dampening, B=cold
        """
        FlowUtil.blit(self._input_fbo, color)

    def set_mask(self, mask: Texture) -> None:
        """Set velocity magnitude mask (from MaskBridge)."""
        FlowUtil.blit(self._mask_fbo, mask)

    def reset(self) -> None:
        """Reset all FBOs to zero."""
        super().reset()
        FlowUtil.zero(self._mask_fbo)

    def update(self, delta_time: float = 1.0) -> None:
        """Update temperature bridge processing.

        Args:
            delta_time: Unused (kept for FlowBase compatibility)
        """
        if not self._allocated:
            return

        self._temperature_bridge_shader.reload()

        # Interpret RGB color and apply velocity mask
        self._output_fbo.begin()
        self._temperature_bridge_shader.use(
            self._input,                    # RGB color (R=warm, G=neutral, B=cold)
            self._mask_fbo.texture,        # Velocity magnitude mask
            self.config.scale
        )
        self._output_fbo.end()

        # FlowUtil.blit(self._output_fbo, self.color_input)
