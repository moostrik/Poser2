"""Trail - general temporal and spatial smoothing.

Provides temporal smoothing (trail effect) and spatial smoothing (Gaussian blur)
for any field type (velocity, density, etc).
"""

from dataclasses import dataclass, field

from OpenGL.GL import *  # type: ignore

from modules.gl import Texture
from .. import FlowBase, FlowConfigBase, FlowUtil
from .shaders import Trail, GaussianBlur

from modules.utils.HotReloadMethods import HotReloadMethods


@dataclass
class SmoothTrailConfig(FlowConfigBase):
    """Configuration for trail smoothing."""
    scale: float = field(
        default=1.0,
        metadata={
            "min": -10.0,
            "max": 10.0,
            "label": "Velocity Scale",
            "description": "Converts optical flow to simulation velocity"
        }
    )
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


class SmoothTrail(FlowBase):
    """General temporal and spatial smoothing processor.

    Provides temporal smoothing (trail effect) and spatial smoothing (Gaussian blur)
    for any field. Works with any texture format (RG, RGB, RGBA).

    Pipeline:
        1. Blend new input with previous trail (temporal smoothing)
        2. Apply Gaussian blur passes (spatial smoothing)
        3. Output smoothed field

    FBOs (from FlowBase):
        - input_fbo: Receives input
        - output_fbo: Stores smoothed output
    """

    def __init__(self, format: int = GL_RGBA16F, config: SmoothTrailConfig | None = None) -> None:
        """Initialize trail processor."""
        super().__init__()

        # Define formats
        self._input_internal_format = format
        self._output_internal_format = format

        self.config: SmoothTrailConfig = config or SmoothTrailConfig()

        # Shaders
        self._trail_shader: Trail = Trail()
        self._blur_shader: GaussianBlur = GaussianBlur()

        hot_reload = HotReloadMethods(self.__class__, True, True)

    def set_input(self, input_field: Texture) -> None:
        """Set input field with optional scaling."""
        FlowUtil.blit(self._input_fbo, input_field)

    def allocate(self, width: int, height: int, output_width: int | None = None, output_height: int | None = None) -> None:
        """Allocate trail processor.

        Args:
            width: Field width
            height: Field height
            output_width: Optional output width (defaults to width)
            output_height: Optional output height (defaults to height)
        """
        super().allocate(width, height, output_width, output_height)

        # Allocate shaders
        self._trail_shader.allocate()
        self._blur_shader.allocate()

    def deallocate(self) -> None:
        """Deallocate all resources."""
        super().deallocate()
        self._trail_shader.deallocate()
        self._blur_shader.deallocate()

    def reset(self) -> None:
        """Reset all buffers to zero."""
        super().reset()

    def update(self, delta_time: float = 1.0) -> None:
        """Process field with temporal and spatial smoothing.

        Args:
            delta_time: Unused (kept for FlowBase compatibility)
        """
        if not self._allocated:
            return

        # Stage 1: Temporal smoothing (trail blend)
        self._output_fbo.swap()
        self._output_fbo.begin()
        self._trail_shader.use(
            self._output_fbo.back_texture,  # Previous trail
            self._input_fbo.texture,        # New input
            self.config.trail_weight,
            self.config.scale                # Apply scaling on new input
        )
        self._output_fbo.end()

        # Stage 2: Spatial smoothing (Gaussian blur)
        if self.config.blur_steps > 0 and self.config.blur_radius > 0:
            # In-place blur using swap operations
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


class VelocitySmoothTrail(SmoothTrail):
    """Convenience class: Trail preset for velocity fields with scaling."""

    def __init__(self, config: SmoothTrailConfig | None = None) -> None:
        """Initialize velocity smoothing.

        Args:
            scale: Velocity scale (converts optical flow to simulation velocity)
            config: Optional trail configuration
        """
        super().__init__(GL_RG16F, config)

    # Convenience aliases for velocity-specific usage
    @property
    def velocity(self) -> Texture:
        """Smoothed velocity output."""
        return self._output

    @property
    def velocity_input(self) -> Texture:
        """Raw velocity input buffer."""
        return self._input

    def set_velocity(self, velocity: Texture) -> None:
        """Alias for .set_input with velocity scaling."""
        self.set_input(velocity)
