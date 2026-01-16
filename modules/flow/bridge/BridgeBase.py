"""Base class for bridge layers.

Bridges sit between optical flow and fluid simulation, providing:
- Velocity smoothing via VelocityProcessor (temporal + spatial)
- Timestep scaling for framerate independence
- Subclass-specific processing (velocity passthrough, density blending, etc.)

Ported from ofxFlowTools ftBridgeFlow.h

Architecture:
    - VelocityProcessor: Handles velocity smoothing (trail + blur)
    - input_fbo/output_fbo: From FlowBase, used for subclass-specific data
    - Subclasses define what goes in input_fbo and output_fbo
"""
from dataclasses import dataclass, field
from abc import abstractmethod

from modules.gl import Texture
from .. import FlowBase, FlowConfigBase
from .VelocityProcessor import VelocityProcessor, VelocityProcessorConfig
from .shaders import MultiplyForce

from modules.utils.HotReloadMethods import HotReloadMethods


@dataclass
class BridgeConfigBase(FlowConfigBase):
    """Base configuration for bridge layers.

    Inherits velocity processing parameters from VelocityProcessorConfig
    via composition, plus adds scale parameter for timestep scaling.
    """

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
    blur_steps: int = field(
        default=1,
        metadata={"min": 0, "max": 8, "label": "Blur Steps",
                  "description": "Number of Gaussian blur passes"}
    )
    scale: float = field(
        default=1.0,
        metadata={"min": 0.0, "max": 200.0, "label": "Scale",
                  "description": "Output scaling factor"}
    )


class BridgeBase(FlowBase):
    """Base class for bridges that smooth and process flow data.

    Bridges provide velocity smoothing via VelocityProcessor, plus
    framerate-independent timestep scaling. Subclasses define what
    goes in input_fbo and output_fbo.

    Architecture:
        - VelocityProcessor: Handles velocity trail + blur (2 FBOs internally)
        - input_fbo: Subclass-specific input (from FlowBase)
        - output_fbo: Subclass-specific output (from FlowBase)

    Subclass responsibilities:
        - Set input_internal_format and output_internal_format in __init__
        - Implement update(delta_time) to process data
        - Use _velocity_processor for velocity smoothing
        - Use _multiply_shader for timestep scaling

    Example subclass patterns:
        VelocityBridge: velocity in → VelocityProcessor → timestep scale → velocity out
        DensityBridge: density+velocity in → combine → HSV adjust → density out
    """

    def __init__(self, config: BridgeConfigBase | None = None) -> None:
        super().__init__()

        # Configuration
        self.config: BridgeConfigBase = config or BridgeConfigBase()
        self.config.add_listener(self._on_config_changed)

        # Velocity processing component
        self._velocity_processor: VelocityProcessor = VelocityProcessor(
            VelocityProcessorConfig(
                trail_weight=self.config.trail_weight,
                blur_radius=self.config.blur_radius,
                blur_steps=self.config.blur_steps
            )
        )

        # Timestep scaling shader
        self._multiply_shader: MultiplyForce = MultiplyForce()

        hot_reload = HotReloadMethods(self.__class__, True, True)

    @property
    def bridge_velocity_output(self) -> Texture:
        """Smoothed velocity (after trail + blur)."""
        return self._velocity_processor.output

    @property
    def bridge_velocity_input(self) -> Texture:
        """Smoothed velocity input (after trail + blur)."""
        return self._velocity_processor.input

    def allocate(self, width: int, height: int, output_width: int | None = None, output_height: int | None = None) -> None:
        """Allocate bridge layer.

        Args:
            width: Processing width
            height: Processing height
            output_width: Output width (defaults to width)
            output_height: Output height (defaults to height)
        """
        # Allocate velocity processor
        self._velocity_processor.allocate(width, height)

        # Allocate timestep scaling shader
        self._multiply_shader.allocate()

        # Let subclass allocate input/output FBOs via FlowBase
        super().allocate(width, height, output_width, output_height)

    def deallocate(self) -> None:
        """Deallocate all resources."""
        super().deallocate()
        self._velocity_processor.deallocate()
        self._multiply_shader.deallocate()

    def set_velocity(self, velocity: Texture) -> None:
        """Set velocity input to VelocityProcessor.

        Args:
            velocity: Velocity texture (RG32F)
        """
        self._velocity_processor.set(velocity)

    def add_velocity(self, velocity: Texture, strength: float = 1.0) -> None:
        """Add velocity input to VelocityProcessor.

        Args:
            velocity: Velocity texture
            strength: Blend strength
        """
        self._velocity_processor.add(velocity, strength)

    def reset(self) -> None:
        """Reset all buffers."""
        super().reset()
        self._velocity_processor.reset()

    @abstractmethod
    def update(self, delta_time: float) -> None:
        """Update bridge processing.

        Subclasses should:
        1. Call self._velocity_processor.update() to process velocity
        2. Implement their specific processing logic
        3. Use self.velocity to access smoothed velocity

        Args:
            delta_time: Time since last update in seconds
        """
        ...

    def _sync_config_to_processor(self) -> None:
        """Sync config changes to VelocityProcessor."""
        self._velocity_processor.config.trail_weight = self.config.trail_weight
        self._velocity_processor.config.blur_radius = self.config.blur_radius
        self._velocity_processor.config.blur_steps = self.config.blur_steps

    def _on_config_changed(self) -> None:
        """Called when config values change."""
        self._sync_config_to_processor()
