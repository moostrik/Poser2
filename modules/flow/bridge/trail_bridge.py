"""Trail Bridge.

The flow bridge's velocity as one trail: a flow's 3D velocity, its x/y shaped (`boost`, clamped to
±1), smoothed in time (the trail blend) and in space (Gaussian blur).
"""

import numpy as np
from OpenGL.GL import GL_RGBA16F

from modules.gl import Texture
from modules.settings import Field, BaseSettings
from .. import FlowBase
from ..shaders import Trail, GaussianBlur
from .shaders import ShapeVelocity


class TrailBridgeSettings(BaseSettings):
    """Configuration for the trail bridge."""
    scale = Field(1.0, min=-10.0, max=10.0, description="Converts the flow's velocity to simulation velocity")
    trail_weight = Field(0.9, min=0.0, max=0.99, description="Temporal smoothing (0=no trail, 0.99=long trail)")
    blur_radius = Field(8.0, min=0.0, max=10.0, description="Gaussian blur radius in pixels")
    blur_steps = Field(4, min=0, max=8, description="Number of Gaussian blur passes")
    boost = Field(0.0, min=-0.5, max=0.9, description="Power boost for small motions")


class TrailBridge(FlowBase):
    """The trail bridge: a flow's 3D velocity, its x/y shaped (`boost`, clamped to ±1), through one 3D trail.

    Pipeline:
        1. Shape the velocity into the input FBO (`set_velocity`)
        2. Blend it with the previous trail (temporal smoothing)
        3. Apply Gaussian blur passes (spatial smoothing)

    FBOs (from FlowBase):
        - input_fbo: The shaped velocity
        - output_fbo: The smoothed velocity
    """

    def __init__(self, settings: TrailBridgeSettings | None = None) -> None:
        super().__init__()

        self._input_internal_format = GL_RGBA16F
        self._output_internal_format = GL_RGBA16F

        self.settings: TrailBridgeSettings = settings or TrailBridgeSettings()

        # Shaders
        self._shape_shader: ShapeVelocity = ShapeVelocity()
        self._trail_shader: Trail = Trail()
        self._blur_shader: GaussianBlur = GaussianBlur()

    @property
    def smooth_velocity(self) -> Texture:
        """The smoothed velocity, xyz (RGBA16F)."""
        return self._output

    @property
    def velocity_input(self) -> Texture:
        """The shaped velocity the trail reads."""
        return self._input

    def allocate(self, width: int, height: int, output_width: int | None = None, output_height: int | None = None) -> None:
        """Allocate the trail.

        Args:
            width: Field width
            height: Field height
            output_width: Optional output width (defaults to width)
            output_height: Optional output height (defaults to height)
        """
        super().allocate(width, height, output_width, output_height)
        self._shape_shader.allocate()
        self._trail_shader.allocate()
        self._blur_shader.allocate()

    def deallocate(self) -> None:
        """Deallocate all resources."""
        super().deallocate()
        self._shape_shader.deallocate()
        self._trail_shader.deallocate()
        self._blur_shader.deallocate()

    def set_velocity(self, velocity: Texture) -> None:
        """Take a flow's 3D velocity, its x/y shaped, as the trail's next input."""
        if not self._allocated:
            return
        self._input_fbo.begin()
        self._shape_shader.use(velocity, 1.0 - self.settings.boost)
        self._input_fbo.end()

    def update(self) -> None:
        """Advance the trail with the input: temporal, then spatial smoothing."""
        if not self._allocated:
            return

        # Stage 1: Temporal smoothing (trail blend)
        self._output_fbo.swap()
        self._output_fbo.begin()
        self._trail_shader.use(
            self._output_fbo.back_texture,  # Previous trail
            self._input_fbo.texture,        # New input
            self.settings.trail_weight,
            self.settings.scale                # Apply scaling on new input
        )
        self._output_fbo.end()

        # Stage 2: Spatial smoothing (Gaussian blur)
        if self.settings.blur_steps > 0 and self.settings.blur_radius > 0:
            # In-place blur using swap operations
            for _ in range(self.settings.blur_steps):
                # Horizontal pass
                self._output_fbo.swap()
                self._output_fbo.begin()
                self._blur_shader.use(
                    self._output_fbo.back_texture,
                    self.settings.blur_radius,
                    horizontal=True
                )
                self._output_fbo.end()

                # Vertical pass
                self._output_fbo.swap()
                self._output_fbo.begin()
                self._blur_shader.use(
                    self._output_fbo.back_texture,
                    self.settings.blur_radius,
                    horizontal=False
                )
                self._output_fbo.end()


# ── numpy reference ─────────────────────────────────────────────────────

def shape_velocity(velocity: np.ndarray, power: float) -> np.ndarray:
    """A velocity (..., 2) with its length to `power`, clamped to ±1 per axis. Mirrors shape_velocity.frag on x/y."""
    length: np.ndarray = np.linalg.norm(velocity, axis=-1, keepdims=True)
    shaped: np.ndarray = np.divide(velocity * length ** power, length, out=np.zeros_like(velocity), where=length > 1e-4)
    return np.clip(shaped, -1.0, 1.0)
