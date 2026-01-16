"""Velocity Bridge.

Simplest bridge: velocity passthrough with temporal/spatial smoothing.
Uses VelocityProcessor for smoothing, then applies timestep scaling.

Ported from ofxFlowTools ftVelocityBridgeFlow.h
"""
from dataclasses import dataclass, field

from OpenGL.GL import *  # type: ignore

from modules.gl import Texture
from modules.utils.PointsAndRects import Rect
from .BridgeBase import BridgeBase, BridgeConfigBase

from modules.utils.HotReloadMethods import HotReloadMethods


@dataclass
class VelocityBridgeConfig(BridgeConfigBase):
    """Configuration for velocity bridge.

    Inherits trail_weight, blur_radius, blur_steps from BridgeConfigBase.
    Overrides scale default for velocity-specific scaling.
    """
    scale: float = field(
        default=60.0,  # Velocity-specific default (converts optical flow to simulation velocity)
        metadata={
            "min": 0.0,
            "max": 200.0,
            "label": "Velocity Scale",
            "description": "Converts optical flow to simulation velocity"
        }
    )


class VelocityBridge(BridgeBase):
    """Velocity bridge with temporal and spatial smoothing.

    Pipeline:
        1. Receive velocity from optical flow via set_velocity()
        2. VelocityProcessor applies temporal smoothing (trail)
        3. VelocityProcessor applies Gaussian blur (spatial smoothing)
        4. Scale by timestep (framerate independent)
        5. Output smoothed velocity via .velocity property (from VelocityProcessor)
        6. Output timestep-scaled velocity via .output property (for fluid simulation)
    """

    def __init__(self, config: VelocityBridgeConfig | None = None) -> None:
        super().__init__(config or VelocityBridgeConfig())

        # Ensure config is VelocityBridgeConfig for type checking
        self.config: VelocityBridgeConfig

        # Define internal formats (output only - no input_fbo used)
        self.input_internal_format = GL_RG32F   # Not used, but required by FlowBase
        self.output_internal_format = GL_RG32F  # Timestep-scaled velocity output

        hot_reload = HotReloadMethods(self.__class__, True, True)

    @property
    def input(self) -> Texture:
        """Input texture."""
        return self.bridge_velocity_input

    @property
    def velocity(self) -> Texture:
        return self.output

    def set(self, texture: Texture) -> None:
        self.set_velocity(texture)

    def add(self, texture: Texture, strength: float = 1.0) -> None:
        self.add_velocity(texture, strength)

    def update(self, delta_time: float) -> None:
        """Update velocity bridge processing.

        Args:
            delta_time: Time since last update in seconds
        """
        if not self._allocated:
            return

        # Stage 1: Process velocity through VelocityProcessor (trail + blur)
        self._velocity_processor.update()

        # Stage 2: Scale by timestep for framerate independence
        timestep = delta_time * self.config.scale

        self.output_fbo.begin()
        self._multiply_shader.use(self._velocity_processor.output, timestep)
        self.output_fbo.end()
