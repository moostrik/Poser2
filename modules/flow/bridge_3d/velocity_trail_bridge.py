"""Velocity Trail Bridge.

The flow bridge's velocity as one trail: a flow's 3D velocity in image widths per second, smoothed in
time and in space. Its output stays in widths per second; how hard it pushes the fluid is the fluid's
(`gain`, `input_x`, `input_y`).

Before each step the trail moves with the motion it holds (`advect`): each pixel takes the trail from
where its own x/y says it came from, so the average follows a moving limb instead of a fixed pixel,
and a fast move is not cut off by a smoothing time longer than it takes to cross.

The smoothing is one unit-gain Ema per pixel. Where the x/y speed rises it follows over `attack`, else over
`smooth`, so a flick reaches the fluid at full strength and a stop still coasts. z follows over `smooth`
alone, so a one-frame spike is averaged instead of passed on. The blur runs `blur_steps` passes per tick
inside the Ema's loop, so its spread grows with the trail's memory.
"""

import numpy as np
from OpenGL.GL import GL_CLAMP_TO_BORDER, GL_RGBA16F

from modules.gl import SwapFbo, Texture
from modules.settings import Field, BaseSettings
from .. import FlowBase, FlowUtil
from ..shaders import GaussianBlur
from .kinetics import decay_weight
from .shaders import KineticAdvect, VelocityTrail


class VelocityTrailBridgeSettings(BaseSettings):
    """Configuration for the velocity trail bridge."""
    advect = Field(0.0, min=0.0, max=1.0, step=0.05, description="How far the trail moves with its motion before each step; 0 = at a fixed pixel")
    attack = Field(0.0, min=0.0, max=500.0, step=1.0, description="Smoothing time of a rising x/y speed, ms; 0 = as smooth (lower: quicker flicks)")
    smooth = Field(0.16, min=0.0, max=3.0, step=0.01, description="Smoothing time, s; the fall of x/y and all of z; 0 = none (higher: longer coast)")
    blur_radius = Field(0.0, min=0.0, max=10.0, newline=True, description="Gaussian blur radius in pixels; 0 = off")
    blur_steps = Field(1, min=1, max=8, description="Gaussian blur passes per tick (more: wider spread)")


class VelocityTrailBridge(FlowBase):
    """The velocity trail bridge: a flow's 3D velocity through one 3D trail.

    Pipeline:
        1. Take the velocity into the input FBO (`set_velocity`)
        2. Move the previous trail with its own motion (`advect`), then blend the input in, a unit-gain Ema over
           `attack` or `smooth` (temporal smoothing)
        3. Blur it, `blur_steps` passes (spatial smoothing)

    FBOs (from FlowBase):
        - input_fbo: The velocity
        - output_fbo: The smoothed velocity
    """

    def __init__(self, settings: VelocityTrailBridgeSettings | None = None) -> None:
        super().__init__()

        self._input_internal_format = GL_RGBA16F
        self._output_internal_format = GL_RGBA16F

        self.settings: VelocityTrailBridgeSettings = settings or VelocityTrailBridgeSettings()

        # Shaders
        self._advect_shader: KineticAdvect = KineticAdvect()
        self._trail_shader: VelocityTrail = VelocityTrail()
        self._blur_shader: GaussianBlur = GaussianBlur()

    @property
    def smooth_velocity(self) -> Texture:
        """The smoothed velocity, xyz in image widths per second (RGBA16F)."""
        return self._output

    @property
    def velocity_input(self) -> Texture:
        """The velocity the trail reads."""
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
        # No velocity outside the image, as in the fluid: a trail that leaves it is gone, and the edge
        # doesn't copy itself back in through the advect and the blur
        self._output_fbo.set_wrap(GL_CLAMP_TO_BORDER, (0.0, 0.0, 0.0, 0.0))
        self._advect_shader.allocate()
        self._trail_shader.allocate()
        self._blur_shader.allocate()

    def deallocate(self) -> None:
        """Deallocate all resources."""
        super().deallocate()
        self._advect_shader.deallocate()
        self._trail_shader.deallocate()
        self._blur_shader.deallocate()

    def set_velocity(self, velocity: Texture) -> None:
        """Take a flow's 3D velocity as the trail's next input."""
        if not self._allocated:
            return
        FlowUtil.blit(self._input_fbo, velocity)

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
            self._advect_shader.use(fbo.back, fbo.back, (step, step * fbo.width / max(fbo.height, 1)))
            fbo.end()

        # Stage 2: Temporal smoothing, one unit-gain Ema step per pixel: weight * previous + (1 - weight) * input
        release: float = decay_weight(delta_time, s.smooth * 1000.0)
        attack: float = decay_weight(delta_time, s.attack) if s.attack > 0.0 else release
        self._output_fbo.swap()
        self._output_fbo.begin()
        self._trail_shader.use(self._output_fbo.back, self._input_fbo, attack, release)
        self._output_fbo.end()

        # Stage 3: Spatial smoothing (Gaussian blur), `blur_steps` passes
        if s.blur_radius > 0.0:
            for _ in range(s.blur_steps):
                for horizontal in (True, False):
                    self._output_fbo.swap()
                    self._output_fbo.begin()
                    self._blur_shader.use(self._output_fbo.back, s.blur_radius, horizontal=horizontal)
                    self._output_fbo.end()


# ── numpy reference ─────────────────────────────────────────────────────

def velocity_trail(previous: np.ndarray, velocity: np.ndarray, attack: float, release: float) -> np.ndarray:
    """One step of the trail (..., 3): x/y by `attack` where its speed rises above the trail's, else by
    `release`; z by `release`. Mirrors velocity_trail.frag."""
    rising: np.ndarray = np.linalg.norm(velocity[..., :2], axis=-1, keepdims=True) > np.linalg.norm(previous[..., :2], axis=-1, keepdims=True)
    weight_xy: np.ndarray = np.where(rising, attack, release)
    xy: np.ndarray = weight_xy * previous[..., :2] + (1.0 - weight_xy) * velocity[..., :2]
    z: np.ndarray = release * previous[..., 2:3] + (1.0 - release) * velocity[..., 2:3]
    return np.concatenate((xy, z), axis=-1)
