"""Density Bridge.

Combines RGB density with velocity magnitude for fluid simulation.

Ported from ofxFlowTools ftDensityBridgeFlow.h
"""
from dataclasses import dataclass, field

from OpenGL.GL import *  # type: ignore

from modules.gl import Texture, SwapFbo
from .. import FlowBase, FlowConfigBase, FlowUtil
from .shaders.DensityBridgeShader import DensityBridgeShader
from ..shaders.HSV import HSV

from modules.utils.HotReloadMethods import HotReloadMethods


@dataclass
class DensityBridgeConfig(FlowConfigBase):
    """Configuration for density bridge."""

    saturation: float = field(
        default=2.5,
        metadata={"min": 0.0, "max": 5.0, "label": "Saturation",
                  "description": "Color saturation boost"}
    )
    brightness: float = field(
        default=1.0,
        metadata={"min": 0.0, "max": 2.0, "label": "Brightness",
                  "description": "Brightness/value adjustment"}
    )


class DensityBridge(FlowBase):
    """Density bridge with velocity-driven alpha channel.

    Pipeline:
        1. Receive RGB density via set_density() → input_fbo (RGBA32F)
        2. Receive pre-smoothed velocity via set_velocity() → velocity_fbo (RG32F)
        3. Combine density RGB + velocity magnitude → RGBA with alpha
        4. Apply HSV adjustments (saturation, brightness)
        5. Output RGBA32F density via .density property

    Data flow:
        Density RGB → set_density() → input_fbo (RGBA32F)
        Smoothed Velocity → set_velocity() → velocity_fbo (RG32F)
        Combine → DensityBridgeShader → output_fbo (RGBA32F)
        HSV adjust → output_fbo (in-place)
    """

    def __init__(self, config: DensityBridgeConfig | None = None) -> None:
        super().__init__()

        self.config: DensityBridgeConfig = config or DensityBridgeConfig()

        # Define internal formats
        self._input_internal_format = GL_RGBA16F   # RGB density input
        self._output_internal_format = GL_RGBA16F  # RGBA density output

        # Velocity storage
        self._velocity_fbo: SwapFbo = SwapFbo()

        # Shaders
        self._density_bridge_shader: DensityBridgeShader = DensityBridgeShader()
        self._hsv_shader: HSV = HSV()

        hot_reload = HotReloadMethods(self.__class__, True, True)

    @property
    def density(self) -> Texture:
        """RGBA density output (main result)."""
        return self._output

    @property
    def color_input(self) -> Texture:
        """RGB density input buffer."""
        return self._input

    @property
    def velocity_input(self) -> Texture:
        """Velocity input buffer."""
        return self._velocity_fbo.texture

    def allocate(self, width: int, height: int, output_width: int | None = None, output_height: int | None = None) -> None:
        """Allocate density bridge FBOs.

        Args:
            width: Processing width
            height: Processing height
            output_width: Optional output width (defaults to width)
            output_height: Optional output height (defaults to height)
        """
        super().allocate(width, height, output_width, output_height)
        self._velocity_fbo.allocate(width, height, GL_RG32F)
        self._density_bridge_shader.allocate()
        self._hsv_shader.allocate()

    def deallocate(self) -> None:
        """Release all FBO resources."""
        super().deallocate()
        self._velocity_fbo.deallocate()
        self._density_bridge_shader.deallocate()
        self._hsv_shader.deallocate()

    def set_color(self, color: Texture) -> None:
        """Set RGB density input (replaces current)."""
        FlowUtil.blit(self._input_fbo, color)

    def set_velocity(self, velocity: Texture) -> None:
        """Set pre-smoothed velocity input (from VelocityBridge)."""
        FlowUtil.blit(self._velocity_fbo, velocity)

    def reset(self) -> None:
        """Reset all FBOs to zero."""
        super().reset()
        FlowUtil.zero(self._velocity_fbo)

    def update(self, delta_time: float = 1.0) -> None:
        """Update density bridge processing.

        Args:
            delta_time: Unused (kept for FlowBase compatibility)
        """
        if not self._allocated:
            return

        # Stage 1: Combine density RGB + velocity magnitude
        self._output_fbo.begin()
        self._density_bridge_shader.use(
            self._input,                    # Density RGB from input_fbo
            self._velocity_fbo.texture,    # Pre-smoothed velocity
            1.0                             # Speed multiplier for alpha
        )
        self._output_fbo.end()

        # Stage 2: Apply HSV adjustments (saturation + brightness)
        self._output_fbo.swap()
        self._output_fbo.begin()
        self._hsv_shader.use(
            self._output_fbo.back_texture,
            hue=0.0,
            saturation=self.config.saturation,
            value=self.config.brightness
        )
        self._output_fbo.end()
