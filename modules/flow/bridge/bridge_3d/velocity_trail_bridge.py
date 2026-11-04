"""Velocity Trail Bridge.

The flow bridge's velocity as one trail: a flow's 3D velocity in image widths per second, shaped,
smoothed in time and in space. Its output stays in widths per second; how hard it pushes the fluid is
the fluid's (`input_strength`, `input_x`, `input_y`).

The shape bends the x/y speed around `pivot` by `curve`: speed · (speed / pivot)^(curve − 1), so a move
at the pivot stays as it is, a faster one gets stronger and a slower one weaker; then a soft `limit`
(0 = none).
z stays linear, with its own soft `z_limit`: what is left of z's noise after the flow is spikes at edges,
which a curve would make bigger and a limit caps.

Before each step the trail moves with the motion it holds (`advect`): each pixel takes the trail from
where its own x/y says it came from, so the average follows a moving limb instead of a fixed pixel,
and a fast move is not cut off by a smoothing time longer than it takes to cross.

The smoothing is one unit-gain Ema per pixel. Where the x/y speed rises it follows over `attack`, else over
`smooth`, so a flick reaches the fluid at full strength and a stop still coasts. z follows over `smooth`
alone, so a one-frame spike is averaged instead of passed on. The blur runs once per tick inside the
Ema's loop, so its spread grows with the trail's memory.
"""

import numpy as np
from OpenGL.GL import GL_RGBA16F

from modules.gl import SwapFbo, Texture
from modules.settings import Field, BaseSettings
from ... import FlowBase
from ...shaders import GaussianBlur
from .kinetics import decay_weight
from .shaders import KineticAdvect, ShapeVelocity, VelocityTrail


class VelocityTrailBridgeSettings(BaseSettings):
    """Configuration for the velocity trail bridge."""
    pivot = Field(0.15, min=0.01, max=1.0, step=0.01, description="x/y speed the curve leaves as it is, widths/s")
    curve = Field(1.0, min=0.5, max=3.0, step=0.05, description="Exponent on the x/y speed around the pivot; 1 = linear (higher: fast stronger, slow weaker)")
    limit = Field(4.0, min=0.0, max=4.0, step=0.05, description="Soft limit on the shaped x/y speed, widths/s; 0 = off")
    z_limit = Field(4.0, min=0.1, max=4.0, step=0.05, description="Soft limit on z, widths/s (lower: smaller edge spikes)")
    advect = Field(0.0, min=0.0, max=1.0, step=0.05, newline=True, description="How far the trail moves with its motion before each step; 0 = at a fixed pixel")
    attack = Field(0.0, min=0.0, max=500.0, step=1.0, description="Smoothing time of a rising x/y speed, ms; 0 = as smooth (lower: quicker flicks)")
    smooth = Field(158.0, min=0.0, max=1000.0, step=1.0, description="Smoothing time, ms; the fall of x/y and all of z; 0 = none (higher: longer coast)")
    blur_radius = Field(0.0, min=0.0, max=10.0, newline=True, description="Gaussian blur radius in pixels, once per tick")


class VelocityTrailBridge(FlowBase):
    """The velocity trail bridge: a flow's 3D velocity, shaped, through one 3D trail.

    Pipeline:
        1. Shape the velocity into the input FBO (`set_velocity`): the x/y speed's curve and limit, z's limit
        2. Move the previous trail with its own motion (`advect`), then blend the input in, a unit-gain Ema over
           `attack` or `smooth` (temporal smoothing)
        3. Blur it once (spatial smoothing)

    FBOs (from FlowBase):
        - input_fbo: The shaped velocity
        - output_fbo: The smoothed velocity
    """

    def __init__(self, settings: VelocityTrailBridgeSettings | None = None) -> None:
        super().__init__()

        self._input_internal_format = GL_RGBA16F
        self._output_internal_format = GL_RGBA16F

        self.settings: VelocityTrailBridgeSettings = settings or VelocityTrailBridgeSettings()

        # Shaders
        self._shape_shader: ShapeVelocity = ShapeVelocity()
        self._advect_shader: KineticAdvect = KineticAdvect()
        self._trail_shader: VelocityTrail = VelocityTrail()
        self._blur_shader: GaussianBlur = GaussianBlur()

    @property
    def smooth_velocity(self) -> Texture:
        """The smoothed velocity, xyz in image widths per second (RGBA16F)."""
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
        self._advect_shader.allocate()
        self._trail_shader.allocate()
        self._blur_shader.allocate()

    def deallocate(self) -> None:
        """Deallocate all resources."""
        super().deallocate()
        self._shape_shader.deallocate()
        self._advect_shader.deallocate()
        self._trail_shader.deallocate()
        self._blur_shader.deallocate()

    def set_velocity(self, velocity: Texture) -> None:
        """Take a flow's 3D velocity, shaped, as the trail's next input."""
        if not self._allocated:
            return
        s: VelocityTrailBridgeSettings = self.settings
        self._input_fbo.begin()
        self._shape_shader.use(velocity, s.pivot, s.curve, s.limit, s.z_limit)
        self._input_fbo.end()

    def update(self, delta_time: float = 1.0 / 60.0) -> None:
        """Advance the trail with the input by `delta_time` (s): temporal, then spatial smoothing."""
        if not self._allocated:
            return
        s: VelocityTrailBridgeSettings = self.settings

        # Stage 1: The trail moves with its motion: x is widths/s, y widths/s on the x scale, so uv per step
        # is v * dt, and v * dt * width / height on y
        if s.advect > 0.0:
            step: float = s.advect * delta_time
            fbo: SwapFbo = self._output_fbo
            fbo.swap()
            fbo.begin()
            self._advect_shader.use(fbo.back_texture, fbo.back_texture, (step, step * fbo.width / max(fbo.height, 1)))
            fbo.end()

        # Stage 2: Temporal smoothing, one unit-gain Ema step per pixel: weight * previous + (1 - weight) * input
        release: float = decay_weight(delta_time, s.smooth)
        attack: float = decay_weight(delta_time, s.attack) if s.attack > 0.0 else release
        self._output_fbo.swap()
        self._output_fbo.begin()
        self._trail_shader.use(self._output_fbo.back_texture, self._input_fbo.texture, attack, release)
        self._output_fbo.end()

        # Stage 3: Spatial smoothing (Gaussian blur), once
        if s.blur_radius > 0.0:
            for horizontal in (True, False):
                self._output_fbo.swap()
                self._output_fbo.begin()
                self._blur_shader.use(self._output_fbo.back_texture, s.blur_radius, horizontal=horizontal)
                self._output_fbo.end()


# ── numpy reference ─────────────────────────────────────────────────────

def shape_velocity(velocity: np.ndarray, pivot: float, curve: float, limit: float, z_limit: float) -> np.ndarray:
    """A velocity (..., 3) with its x/y speed s bent to s · (s / pivot)^(curve − 1), then rolled off toward
    `limit` (limit · tanh(s / limit); 0 = none), the direction kept; z rolled off toward `z_limit`. Mirrors shape_velocity.frag."""
    xy: np.ndarray = velocity[..., :2]
    speed: np.ndarray = np.linalg.norm(xy, axis=-1, keepdims=True)
    safe: np.ndarray = np.maximum(speed, 1e-4)
    shaped: np.ndarray = speed * (safe / pivot) ** (curve - 1.0)
    if limit > 0.0:
        shaped = limit * np.tanh(shaped / limit)
    xy = np.where(speed > 1e-4, xy / safe * shaped, 0.0)
    z: np.ndarray = z_limit * np.tanh(velocity[..., 2:3] / z_limit)
    return np.concatenate((xy, z), axis=-1)


def velocity_trail(previous: np.ndarray, velocity: np.ndarray, attack: float, release: float) -> np.ndarray:
    """One step of the trail (..., 3): x/y by `attack` where its speed rises above the trail's, else by
    `release`; z by `release`. Mirrors velocity_trail.frag."""
    rising: np.ndarray = np.linalg.norm(velocity[..., :2], axis=-1, keepdims=True) > np.linalg.norm(previous[..., :2], axis=-1, keepdims=True)
    weight_xy: np.ndarray = np.where(rising, attack, release)
    xy: np.ndarray = weight_xy * previous[..., :2] + (1.0 - weight_xy) * velocity[..., :2]
    z: np.ndarray = release * previous[..., 2:3] + (1.0 - release) * velocity[..., 2:3]
    return np.concatenate((xy, z), axis=-1)
