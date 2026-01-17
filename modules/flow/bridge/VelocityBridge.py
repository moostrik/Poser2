"""Velocity Processor - standalone velocity smoothing component.

Handles temporal smoothing (trail) and spatial smoothing (blur) for velocity fields.
Extracted from BridgeBase to provide clear separation of concerns.
"""

from dataclasses import dataclass, field

from OpenGL.GL import *  # type: ignore

from modules.gl import Texture, Fbo
from .. import FlowBase, FlowConfigBase
from .shaders import BridgeTrail, GaussianBlur, Scale

from modules.utils.HotReloadMethods import HotReloadMethods


@dataclass
class VelocityBridgeConfig(FlowConfigBase):
    """Configuration for velocity processor."""

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
    time_scale: float = field(
        default=60.0,
        metadata={
            "min": 0.0,
            "max": 200.0,
            "label": "Velocity Scale",
            "description": "Converts optical flow to simulation velocity"
        }
    )


class VelocityBridge(FlowBase):
    """Standalone velocity smoothing processor.

    Provides temporal smoothing (trail effect) and spatial smoothing (Gaussian blur)
    for velocity fields. Uses FlowBase input/output FBOs.

    Pipeline:
        1. Blend new velocity with previous trail (temporal smoothing)
        2. Apply Gaussian blur passes (spatial smoothing)
        3. Output smoothed velocity

    FBOs (from FlowBase):
        - input_fbo: Receives raw velocity input
        - output_fbo: Stores temporally smoothed velocity (output)

    Usage:
        processor = VelocityProcessor(config)
        processor.allocate(256, 256)

        # Each frame
        processor.set(optical_flow.output)
        processor.update()
        smoothed = processor.output  # Access smoothed output
    """

    def __init__(self, config: VelocityBridgeConfig | None = None) -> None:
        """Initialize velocity processor.

        Args:
            config: Optional configuration (defaults to VelocityProcessorConfig)
        """
        super().__init__()

        # Define formats for FlowBase
        self._input_internal_format = GL_RG32F
        self._output_internal_format = GL_RG32F

        self.config: VelocityBridgeConfig = config or VelocityBridgeConfig()

        self._output_delta: Fbo = Fbo()

        # Shaders
        self._bridge_trail_shader: BridgeTrail = BridgeTrail()
        self._blur_shader: GaussianBlur = GaussianBlur()
        self._multiply_shader: Scale = Scale()

        hot_reload = HotReloadMethods(self.__class__, True, True)

    @property
    def velocity(self) -> Texture:
        """Smoothed velocity output."""
        return self._output

    @property
    def velocity_delta(self) -> Texture:
        """Smoothed velocity scaled by timestep."""
        return self._output_delta

    @property
    def velocity_input(self) -> Texture:
        """Raw velocity input."""
        return self._input

    def set_velocity(self, velocity: Texture) -> None:
        """Set velocity input to VelocityProcessor.

        Args:
            velocity: Velocity texture (RG32F)
        """
        self._set(velocity)

    def allocate(self, width: int, height: int, output_width: int | None = None, output_height: int | None = None) -> None:
        """Allocate velocity processor.

        Args:
            width: Velocity field width
            height: Velocity field height
            output_width: Optional output width (defaults to width)
            output_height: Optional output height (defaults to height)
        """
        super().allocate(width, height, output_width, output_height)

        self._output_delta.allocate(width, height, self._output_internal_format)
        # Allocate shaders
        self._bridge_trail_shader.allocate()
        self._blur_shader.allocate()
        self._multiply_shader.allocate()

    def deallocate(self) -> None:
        """Deallocate all resources."""
        super().deallocate()
        self._output_delta.deallocate()
        self._bridge_trail_shader.deallocate()
        self._blur_shader.deallocate()

    def reset(self) -> None:
        """Reset all buffers to zero."""
        super().reset()

    def update(self, delta_time: float) -> None:
        """Process velocity with temporal and spatial smoothing."""
        if not self._allocated:
            return

        # Stage 1: Temporal smoothing (trail blend)
        self._output_fbo.swap()
        self._output_fbo.begin()
        self._bridge_trail_shader.use(
            self._output_fbo.back_texture,  # Previous trail
            self._input_fbo.texture,        # New velocity
            self.config.trail_weight
        )
        self._output_fbo.end()

        # Stage 2: Spatial smoothing (Gaussian blur)
        if self.config.blur_steps > 0 and self.config.blur_radius > 0:
            # In-place blur using swap operations (like C++ original)
            for _ in range(self.config.blur_steps):
                # Horizontal pass
                self._output_fbo.swap()
                self._output_fbo.begin()
                self._blur_shader.use(
                    self._output_fbo.back_texture,
                    self.config.blur_radius,
                    horizontal=True
                )
                self._output_fbo.end()

                # Vertical pass
                self._output_fbo.swap()
                self._output_fbo.begin()
                self._blur_shader.use(
                    self._output_fbo.back_texture,
                    self.config.blur_radius,
                    horizontal=False
                )
                self._output_fbo.end()

        timestep: float = delta_time * self.config.time_scale

        self._output_delta.begin()
        self._multiply_shader.use(self._output_fbo, timestep)
        self._output_delta.end()
