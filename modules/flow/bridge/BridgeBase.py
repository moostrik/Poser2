"""Base class for bridge layers.

Bridges sit between optical flow and fluid simulation, providing:
- Temporal smoothing (trail effect)
- Spatial smoothing (Gaussian blur)
- Timestep scaling for framerate independence

Ported from ofxFlowTools ftBridgeFlow.h
"""
from dataclasses import dataclass, field
from abc import abstractmethod

from OpenGL.GL import *  # type: ignore

from modules.gl import SwapFbo, Texture
from .. import FlowBase, FlowConfigBase, FlowUtil
from .shaders import BridgeTrail, MultiplyForce, GaussianBlur

from modules.utils.HotReloadMethods import HotReloadMethods


@dataclass
class BridgeConfigBase(FlowConfigBase):
    """Base configuration for bridge layers."""

    trail_weight: float = field(
        default=0.3,
        metadata={"min": 0.0, "max": 0.99, "label": "Trail Weight",
                  "description": "Temporal smoothing (0=no trail, 0.99=long trail)"}
    )
    blur_radius: float = field(
        default=3.0,
        metadata={"min": 0.0, "max": 10.0, "label": "Blur Radius",
                  "description": "Gaussian blur radius in pixels"}
    )
    speed: float = field(
        default=0.3,
        metadata={"min": 0.0, "max": 1.0, "label": "Speed",
                  "description": "Global speed multiplier"}
    )


class BridgeBase(FlowBase):
    """Base class for bridges that smooth and process flow data.

    Bridges provide temporal and spatial smoothing between optical flow
    and fluid simulation, with framerate-independent timestep scaling.

    Internal structure:
    - velocity_input_fbo: Receives raw velocity from optical flow
    - velocity_trail_fbo: Stores temporally smoothed velocity
    - output_fbo: Final processed output (velocity, density, etc.)
    """

    def __init__(self, config: BridgeConfigBase | None = None) -> None:
        super().__init__()

        # Configuration
        self.config: BridgeConfigBase = config or BridgeConfigBase()
        self.config.add_listener(self._on_config_changed)

        # Additional internal FBOs
        self._velocity_input_fbo: SwapFbo = SwapFbo()
        self._velocity_trail_fbo: SwapFbo = SwapFbo()

        # Core bridge shaders
        self._bridge_trail_shader: BridgeTrail = BridgeTrail()
        self._multiply_shader: MultiplyForce = MultiplyForce()
        self._blur_shader: GaussianBlur = GaussianBlur()

        self._needs_update: bool = True

        hot_reload = HotReloadMethods(self.__class__, True, True)

    def allocate(self, width: int, height: int, output_width: int | None = None, output_height: int | None = None) -> None:
        """Allocate bridge layer.

        Args:
            width: Velocity processing width
            height: Velocity processing height
            output_width: Output width (defaults to width)
            output_height: Output height (defaults to height)
        """
        # Allocate shaders
        self._bridge_trail_shader.allocate()
        self._multiply_shader.allocate()
        self._blur_shader.allocate()

        # Velocity FBOs always use RG32F
        self._velocity_input_fbo.allocate(width, height, GL_RG32F)
        self._velocity_trail_fbo.allocate(width, height, GL_RG32F)
        FlowUtil.zero(self._velocity_input_fbo)
        FlowUtil.zero(self._velocity_trail_fbo)

        # Let subclass set input/output formats before calling super
        super().allocate(width, height, output_width, output_height)

    def deallocate(self) -> None:
        """Deallocate all resources."""
        super().deallocate()

        self._velocity_input_fbo.deallocate()
        self._velocity_trail_fbo.deallocate()

        self._bridge_trail_shader.deallocate()
        self._multiply_shader.deallocate()
        self._blur_shader.deallocate()

    def set_velocity(self, velocity: Texture) -> None:
        """Set velocity input from optical flow.

        Args:
            velocity: Velocity texture (RG32F)
        """
        if not self._allocated:
            return
        FlowUtil.stretch(self._velocity_input_fbo, velocity)
        self._needs_update = True

    def add_velocity(self, velocity: Texture, strength: float = 1.0) -> None:
        """Add velocity input (blend with existing).

        Args:
            velocity: Velocity texture
            strength: Blend strength
        """
        if not self._allocated:
            return
        FlowUtil.add(self._velocity_input_fbo, velocity, strength)
        self._needs_update = True

    @property
    def velocity(self) -> Texture:
        """Get processed velocity output."""
        return self._velocity_trail_fbo.texture

    def reset(self) -> None:
        """Reset all buffers."""
        super().reset()
        FlowUtil.zero(self._velocity_input_fbo)
        FlowUtil.zero(self._velocity_trail_fbo)
        self._needs_update = True

    @abstractmethod
    def update(self, delta_time: float) -> None:
        """Update bridge processing.

        Subclasses should call _process_velocity_trail() first,
        then implement their specific processing.

        Args:
            delta_time: Time since last update in seconds
        """
        ...

    def _process_velocity_trail(self) -> None:
        """Process velocity trail with temporal smoothing and blur.

        This is the core bridge processing pipeline:
        1. Blend new velocity input with previous trail
        2. Apply Gaussian blur if enabled
        """
        if not self._needs_update:
            return

        self._needs_update = False

        # Temporal smoothing: blend with previous trail
        self._velocity_trail_fbo.swap()
        self._velocity_trail_fbo.begin()
        self._bridge_trail_shader.use(
            self._velocity_trail_fbo.back_texture,
            self._velocity_input_fbo.texture,
            self.config.trail_weight
        )
        self._velocity_trail_fbo.end()

        # Spatial smoothing: Gaussian blur
        if self.config.blur_radius > 0:
            self._velocity_trail_fbo.swap()
            self._blur_shader.use(
                self._velocity_trail_fbo.back_texture,
                self.config.blur_radius,
                self._velocity_trail_fbo
            )

    def _on_config_changed(self) -> None:
        """Called when config values change."""
        self._needs_update = True
