"""Bridge Flow Layer - smooths and processes optical flow for fluid simulation."""

# Standard library imports
from typing import cast

# Third-party imports
from OpenGL.GL import *  # type: ignore

# Local application imports
from modules.gl import Texture, Style
from modules.render.layers.LayerBase import LayerBase, Rect

from modules.flow import VelocityBridge, VelocityBridgeConfig, Visualizer, VisualisationFieldConfig
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

    def __init__(self, source: LayerBase, fps: float = 60) -> None:
        """Initialize bridge layer.

        Args:
            source: Source layer providing velocity input (e.g., OpticalFlowLayer)
        """
        self._source: LayerBase = source
        self._bridge: VelocityBridge = VelocityBridge()
        self._visualizer: Visualizer = Visualizer()

        self.draw_mode: DrawModes = DrawModes.FIELD
        self._delta_time: float = 1 / fps

        hot_reload = HotReloadMethods(self.__class__, True, True)

    @property
    def config(self) -> VelocityBridgeConfig:
        """Access to bridge configuration."""
        return cast(VelocityBridgeConfig, self._bridge.config)

    @property
    def vis_config(self) -> VisualisationFieldConfig:
        """Access to visualization configuration."""
        return self._visualizer.config

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
        self._visualizer.allocate(width, height)

    def deallocate(self) -> None:
        """Deallocate all resources."""
        self._bridge.deallocate()
        self._visualizer.deallocate()

    def update(self) -> None:
        """Update bridge processing with automatic timestep calculation."""

        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.DISABLED)

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

        Style.pop_style()

    def draw(self, rect: Rect) -> None:
        self.draw_mode: DrawModes = DrawModes.FIELD

        self.vis_config.element_length = 40.0
        self.vis_config.toggle_scalar = False
        self.config.trail_weight = 0.9
        self.config.blur_steps = 2
        self.config.blur_radius = 3.0


        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.DISABLED)

        if self.draw_mode == DrawModes.INPUT:
            # Draw raw input velocity from source
            self._source.draw(rect)
        elif self.draw_mode == DrawModes.OUTPUT:
            # Draw smoothed velocity (raw RG texture)
            self._bridge.draw_output(rect)
        else:
            Style.set_blend_mode(Style.BlendMode.ADDITIVE)
            self._visualizer.draw(self._bridge.velocity, rect)

        Style.pop_style()

    def reset(self) -> None:
        """Reset bridge state."""
        self._bridge.reset()
