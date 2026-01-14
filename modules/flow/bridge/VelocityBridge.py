"""Velocity Bridge.

Simplest bridge: velocity passthrough with temporal/spatial smoothing.
Ported from ofxFlowTools ftVelocityBridgeFlow.h
"""
from dataclasses import dataclass

from OpenGL.GL import *  # type: ignore

from modules.gl import Texture
from .BridgeBase import BridgeBase, BridgeConfigBase

from modules.utils.HotReloadMethods import HotReloadMethods


@dataclass
class VelocityBridgeConfig(BridgeConfigBase):
    """Configuration for velocity bridge.

    Inherits trail_weight, blur_radius, and speed from BridgeConfigBase.
    """
    pass


class VelocityBridge(BridgeBase):
    """Velocity bridge with temporal and spatial smoothing.

    Pipeline:
    1. Receive velocity from optical flow via set_velocity()
    2. Apply temporal smoothing (trail effect)
    3. Apply Gaussian blur (spatial smoothing)
    4. Scale by timestep (framerate independent)
    5. Output smoothed velocity via .velocity property

    Usage:
        bridge = VelocityBridge(VelocityBridgeConfig(
            trail_weight=0.5,
            blur_radius=2.0,
            speed=0.3
        ))
        bridge.allocate(256, 256)

        # Each frame
        bridge.set_velocity(optical_flow.velocity)
        bridge.update(delta_time)

        # Use smoothed velocity for fluid
        fluid.set_velocity(bridge.velocity)
    """

    def __init__(self, config: VelocityBridgeConfig | None = None) -> None:
        super().__init__(config or VelocityBridgeConfig())

        # Define internal formats
        self.input_internal_format = GL_RG32F   # Not used (we manage velocity FBOs internally)
        self.output_internal_format = GL_RG32F  # Velocity output

        hot_reload = HotReloadMethods(self.__class__, True, True)

    def update(self, delta_time: float) -> None:
        """Update velocity bridge processing.

        Args:
            delta_time: Time since last update in seconds
        """
        if not self._allocated:
            return

        # Process velocity trail (temporal + spatial smoothing)
        self._process_velocity_trail()

        # Scale by timestep for framerate independence
        timestep = delta_time * self.config.speed * 200.0

        self.output_fbo.begin()
        self._multiply_shader.use(self._velocity_trail_fbo.texture, timestep)
        self.output_fbo.end()
