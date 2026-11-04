"""Trail Bridge.

The flow bridge's velocity as one trail: a flow's 3D velocity, its x/y shaped (`boost`, soft `limit`),
scaled per axis (`scale_x`, `scale_y`, `scale_z`), smoothed in time (a unit-gain Ema over `smooth`) and in space (Gaussian blur).
"""

import numpy as np
from OpenGL.GL import GL_RGBA16F

from modules.gl import Texture
from modules.settings import Field, BaseSettings
from ... import FlowBase
from ...shaders import Trail, GaussianBlur
from .kinetics import decay_weight
from .shaders import ShapeVelocity


class TrailBridgeSettings(BaseSettings):
    """Configuration for the trail bridge."""
    scale_x = Field(10.0, min=-50.0, max=50.0, description="Converts the flow's x to simulation velocity")
    scale_y = Field(10.0, min=-50.0, max=50.0, description="Converts the flow's y to simulation velocity")
    scale_z = Field(10.0, min=-50.0, max=50.0, description="Converts the flow's z to simulation velocity")
    boost = Field(0.0, min=-0.5, max=0.9, newline=True, description="Power boost for small motions")
    limit = Field(4.0, min=0.5, max=20.0, description="Soft limit on the x/y speed before scale (widths/s)")
    smooth = Field(158.0, min=0.0, max=1000.0, step=1.0, newline=True, description="Smoothing time, ms; 0 = none (higher: smoother, more lag)")
    blur_radius = Field(8.0, min=0.0, max=10.0, newline=True, description="Gaussian blur radius in pixels")
    blur_steps = Field(4, min=0, max=8, description="Number of Gaussian blur passes")


class TrailBridge(FlowBase):
    """The trail bridge: a flow's 3D velocity, its x/y shaped (`boost`, soft `limit`), through one 3D trail.

    Pipeline:
        1. Shape the velocity and scale it per axis into the input FBO (`set_velocity`)
        2. Blend it with the previous trail, a unit-gain Ema over `smooth` (temporal smoothing)
        3. Apply Gaussian blur passes (spatial smoothing)

    FBOs (from FlowBase):
        - input_fbo: The shaped, scaled velocity
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
        """The shaped, scaled velocity the trail reads."""
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
        """Take a flow's 3D velocity, its x/y shaped, then scaled per axis, as the trail's next input."""
        if not self._allocated:
            return
        s: TrailBridgeSettings = self.settings
        self._input_fbo.begin()
        self._shape_shader.use(velocity, 1.0 - s.boost, s.limit, (s.scale_x, s.scale_y, s.scale_z))
        self._input_fbo.end()

    def update(self, delta_time: float = 1.0 / 60.0) -> None:
        """Advance the trail with the input by `delta_time` (s): temporal, then spatial smoothing."""
        if not self._allocated:
            return

        # Stage 1: Temporal smoothing, one unit-gain Ema step: weight * previous + (1 - weight) * input
        weight: float = decay_weight(delta_time, self.settings.smooth)
        self._output_fbo.swap()
        self._output_fbo.begin()
        self._trail_shader.use(self._output_fbo.back_texture, self._input_fbo.texture, weight, 1.0 - weight)
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
