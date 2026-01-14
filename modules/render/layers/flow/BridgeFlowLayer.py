"""Bridge Flow Layer - smooths and processes optical flow for fluid simulation."""

# Standard library imports
import time

# Third-party imports
from OpenGL.GL import *  # type: ignore

# Local application imports
from modules.gl import Texture, Style
from modules.render.layers.LayerBase import LayerBase, Rect

from modules.flow import VelocityBridge, VelocityBridgeConfig, Velocity, VelocityConfig, VisualizationMode
from modules.render.layers.flow.FlowDefinitions import DrawModes

from modules.utils.HotReloadMethods import HotReloadMethods


class BridgeFlowLayer(LayerBase):
    """Bridge layer for temporal and spatial smoothing of optical flow.

    Sits between optical flow and fluid simulation, providing:
    - Temporal smoothing (trail effect)
    - Spatial smoothing (Gaussian blur)
    - Framerate-independent timestep scaling

    Usage:
        optical_flow = OpticalFlowLayer(camera_layer)
        bridge = BridgeFlowLayer(optical_flow)

        # Each frame:
        bridge.update()
        velocity = bridge.texture  # Smoothed velocity for fluid
    """

    def __init__(self, source: LayerBase) -> None:
        """Initialize bridge layer.

        Args:
            source: Source layer providing velocity input (e.g., OpticalFlowLayer)
        """
        self._source: LayerBase = source

        # Bridge pipeline
        self._bridge: VelocityBridge = VelocityBridge()

        # Visualization
        self._velocity_vis: Velocity = Velocity()

        # Draw mode
        self.draw_mode: DrawModes = DrawModes.FIELD

        # Timestep tracking
        self._last_time: float = time.time()
        self._delta_time: float = 0.016  # Default ~60fps

        hot_reload = HotReloadMethods(self.__class__, True, True)

    @property
    def bridge_config(self) -> VelocityBridgeConfig:
        """Access to bridge configuration."""
        return self._bridge.config  # type: ignore

    @property
    def vis_config(self) -> VelocityConfig:
        """Access to visualization configuration."""
        return self._velocity_vis.config  # type: ignore

    @property
    def texture(self) -> Texture:
        """Output smoothed velocity texture."""
        return self._bridge.velocity

    @property
    def velocity(self) -> Texture:
        """Alias for texture - smoothed velocity field."""
        return self._bridge.velocity

    def allocate(self, width: int, height: int, internal_format: int | None = None) -> None:
        """Allocate bridge layer.

        Args:
            width: Processing width
            height: Processing height
            internal_format: Ignored (uses RG32F internally)
        """
        self._bridge.allocate(width, height)
        self._velocity_vis.allocate(width, height)

    def deallocate(self) -> None:
        """Deallocate all resources."""
        self._bridge.deallocate()
        self._velocity_vis.deallocate()

    def update(self) -> None:
        """Update bridge processing with automatic timestep calculation."""
        # Calculate delta time
        current_time = time.time()
        self._delta_time = current_time - self._last_time
        self._last_time = current_time

        # Clamp delta time to reasonable range (avoid huge jumps)
        self._delta_time = max(0.001, min(self._delta_time, 0.1))

        # Check if source is active
        active: bool = getattr(self._source, "available", True)
        if not active:
            self._bridge.reset()
            return

        # Get velocity from source
        source_velocity: Texture = self._source.texture
        if not source_velocity.allocated:
            return

        # Process through bridge
        self._bridge.set_velocity(source_velocity)
        self._bridge.update(self._delta_time)

    def draw(self, rect: Rect) -> None:
        """Draw bridge output or visualization."""


        self.draw_mode: DrawModes = DrawModes.FIELD

        self._velocity_vis.config.arrow_length = 40.0

        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.DISABLED)

        if self.draw_mode == DrawModes.INPUT:
            # Draw raw input velocity from source
            self._source.draw(rect)
        elif self.draw_mode == DrawModes.OUTPUT:
            # Draw smoothed velocity (raw RG texture)
            self._bridge.draw_output(rect)
        else:
            # Draw visualization (direction map or arrow field)
            Style.set_blend_mode(Style.BlendMode.ADDITIVE)
            self._velocity_vis.config.mode = (
                VisualizationMode.DIRECTION_MAP if self.draw_mode == DrawModes.SCALAR
                else VisualizationMode.ARROW_FIELD
            )
            self._velocity_vis.set(self._bridge.velocity)
            self._velocity_vis.update()
            self._velocity_vis.draw(rect)

        Style.pop_style()

    def reset(self) -> None:
        """Reset bridge state."""
        self._bridge.reset()
        self._last_time = time.time()
