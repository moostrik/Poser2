"""Density Bridge.

Combines RGB density with velocity magnitude for fluid simulation.
Ported from ofxFlowTools ftDensityBridgeFlow.h
"""
from dataclasses import dataclass, field

from OpenGL.GL import *  # type: ignore

from modules.gl import Texture, Fbo
from .BridgeBase import BridgeBase, BridgeConfigBase
from .shaders.DensityBridgeShader import DensityBridgeShader
from ..shaders.HSV import HSV
from ..shaders.MultiplyForce import MultiplyForce

from modules.utils.HotReloadMethods import HotReloadMethods


@dataclass
class DensityBridgeConfig(BridgeConfigBase):
    """Configuration for density bridge.

    Inherits trail_weight, blur_radius, blur_steps, and scale from BridgeConfigBase.
    """
    scale: float = field(
        default=10.0,  # Override for density-specific default
        metadata={
            "min": 0.0,
            "max": 200.0,
            "label": "Density Scale",
            "description": "Converts velocity magnitude to density alpha"
        }
    )
    saturation: float = field(
        default=2.5,
        metadata={
            "min": 0.0,
            "max": 5.0,
            "label": "Saturation",
            "description": "Color saturation boost for visualization"
        }
    )


class DensityBridge(BridgeBase):
    """Density bridge with velocity-driven alpha channel.

    Pipeline:
    1. Receive RGB density via set_density()
    2. Receive velocity from optical flow via set_velocity()
    3. Apply temporal smoothing to velocity (trail effect)
    4. Apply Gaussian blur to velocity (spatial smoothing)
    5. Combine density RGB + velocity magnitude → RGBA with alpha channel
    6. Apply HSV saturation adjustment
    7. Output RGBA32F density via .density property
    8. Provide display-ready RGBA8 via .density_visible property
    """

    def __init__(self, config: DensityBridgeConfig | None = None) -> None:
        super().__init__(config or DensityBridgeConfig())

        # Ensure config is DensityBridgeConfig for type checking
        self.config: DensityBridgeConfig

        # Define internal formats
        self.input_internal_format = GL_RGBA32F   # RGB density input
        self.output_internal_format = GL_RGBA32F  # RGBA density output

        # Additional shaders
        self._density_bridge_shader: DensityBridgeShader = DensityBridgeShader()
        self._hsv_shader: HSV = HSV()

        # Display FBO (RGBA8 for rendering)
        self._visible_fbo: Fbo = Fbo()

        hot_reload = HotReloadMethods(self.__class__, True, True)

    @property
    def input_density(self) -> Texture:
        """Input density texture."""
        return self.input

    @property
    def density(self) -> Texture:
        return self.output

    @property
    def density_visible(self) -> Texture:
        """Get RGBA8 density output for display (framerate-scaled)."""
        # Multiply by framerate for visible output
        self._visible_fbo.begin()
        self._multiply_shader.use(self.output, 60.0)  # Default 60 FPS scaling
        self._visible_fbo.end()
        return self._visible_fbo

    def allocate(self, width: int, height: int, output_width: int | None = None, output_height: int | None = None) -> None:
        """Allocate density bridge FBOs.

        Args:
            width: Processing width
            height: Processing height
            output_width: Optional output width (defaults to width)
            output_height: Optional output height (defaults to height)
        """
        # Call parent to setup velocity processing + input/output FBOs
        super().allocate(width, height, output_width, output_height)

        # Allocate display FBO
        out_w = output_width if output_width is not None else width
        out_h = output_height if output_height is not None else height
        self._visible_fbo.allocate(out_w, out_h, GL_RGBA8)
        self._visible_fbo.clear(0.0, 0.0, 0.0, 0.0)

        self._density_bridge_shader.allocate()
        self._hsv_shader.allocate()

    def deallocate(self) -> None:
        """Release all FBO resources."""
        super().deallocate()
        self._visible_fbo.deallocate()
        self._density_bridge_shader.deallocate()
        self._hsv_shader.deallocate()

    def set_density(self, density: Texture) -> None:
        """Set RGB density input (replaces previous).

        Args:
            density: RGB or RGBA density texture
        """
        self.set(density)

    def add_density(self, density: Texture, strength: float = 1.0) -> None:
        """Add RGB density input (accumulates with strength).

        Args:
            density: RGB or RGBA density texture
            strength: Blend strength multiplier
        """
        self.add(density, strength)

    def reset(self) -> None:
        """Reset all FBOs to zero."""
        super().reset()
        self._visible_fbo.clear(0.0, 0.0, 0.0, 0.0)

    def update(self, delta_time: float) -> None:
        """Update density bridge processing.

        Args:
            delta_time: Time since last update in seconds
        """
        if not self._allocated:
            return

        # 1. Process velocity trail (temporal + spatial smoothing)
        self._process_velocity_trail()

        # 2. Combine density RGB + velocity magnitude with timestep scaling
        timestep = delta_time * self.config.scale

        self.output_fbo.begin()
        self._density_bridge_shader.use(
            self.input,                      # Density RGB from input_fbo
            self.velocity,                   # Velocity RG from velocity_trail_fbo
            timestep
        )
        self.output_fbo.end()

        # 3. Apply HSV saturation adjustment
        self.output_fbo.swap()
        self.output_fbo.begin()
        self._hsv_shader.use(
            self.output_fbo.back_texture,
            hue=0.0,
            saturation=self.config.saturation,
            value=1.0
        )
        self.output_fbo.end()
