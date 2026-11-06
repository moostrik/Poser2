"""Velocity Lowpass.

The bridge's lowpass: a flow's 3D velocity in image widths per second, smoothed in time and in space, the
reading `trail`. Its output stays in widths per second; how strongly the fluid takes it is the fluid's.

Before each step the trail moves with the motion it holds (`advect`): each pixel takes the trail from
where its own x/y says it came from, so the average follows a moving limb instead of a fixed pixel,
and a fast move is not cut off by a smoothing time longer than it takes to cross.

The smoothing is one unit-gain Ema per pixel. Where the 3D speed rises it follows over `attack`, else over
`smooth`, so a flick reaches the fluid at full strength and a stop still coasts. x, y and z alike: z's noise
is the motion's to filter (its depth velocity's `smooth`). The blur runs `blur_steps` passes per tick inside
the Ema's loop, so its spread grows with the trail's memory.

The trail is read in two units:
    trail                the trail, widths/s
    trail_acceleration   the trail's rate, widths/s²: its Ema step over dt, (1 − w)(velocity − trail) / dt,
                         which is (velocity − trail) / τ for a dt well under τ, τ the attack where the speed
                         rises, else the smooth: the highpass over the lowpass's time, a spike at a start, a
                         brake over the coast at a stop. Its sum × dt is the trail's change, at any frame rate.
                         Advected with the trail and blurred as it is; written only when asked
"""

import numpy as np
from OpenGL.GL import GL_CLAMP_TO_BORDER, GL_RGBA16F

from modules.gl import SwapFbo, Texture
from modules.settings import Field, BaseSettings
from .. import FlowBase, FlowUtil, Port, TextureKind
from ..shaders import GaussianBlur
from .trails import decay_weight
from .shaders import TrailAdvect, VelocityTrail, WeightedSum


class VelocityLowpassSettings(BaseSettings):
    """Configuration for the velocity lowpass."""
    advect = Field(0.0, min=0.0, max=1.0, step=0.05, description="How far the trail moves with its motion before each step; 0 = at a fixed pixel")
    attack = Field(0.0, min=0.0, max=500.0, step=1.0, description="Smoothing time of a rising speed, ms; 0 = as smooth (lower: quicker flicks)")
    smooth = Field(160.0, min=0.0, max=3000.0, step=1.0, description="Smoothing time of a falling speed, ms; 0 = none (higher: longer coast)")
    blur_radius = Field(0.0, min=0.0, max=10.0, newline=True, description="Gaussian blur radius in pixels; 0 = off")
    blur_steps = Field(1, min=1, max=8, description="Gaussian blur passes per tick (more: wider spread)")


class VelocityLowpass(FlowBase):
    """The velocity lowpass: a flow's 3D velocity through one 3D trail, read as a velocity and an acceleration.

    Pipeline:
        1. Take the velocity into the input FBO (`set_velocity`)
        2. Move the previous trail with its own motion (`advect`), then blend the input in, a unit-gain Ema over
           `attack` or `smooth` (temporal smoothing); with `acceleration`, the step over dt
        3. Blur the trail, and the acceleration with it, `blur_steps` passes (spatial smoothing)

    FBOs (from FlowBase):
        - input_fbo: The velocity
        - output_fbo: The trail
    """

    def __init__(self, settings: VelocityLowpassSettings | None = None) -> None:
        super().__init__()

        self._input_internal_format = GL_RGBA16F
        self._output_internal_format = GL_RGBA16F
        self._input_kind = TextureKind.VELOCITY
        self._output_kind = TextureKind.VELOCITY

        self.settings: VelocityLowpassSettings = settings or VelocityLowpassSettings()
        self._acceleration_fbo: SwapFbo = SwapFbo()     # the trail's rate, widths/s² (RGBA16F, xyz)

        # Shaders
        self._advect_shader: TrailAdvect = TrailAdvect()
        self._trail_shader: VelocityTrail = VelocityTrail()
        self._blur_shader: GaussianBlur = GaussianBlur()
        self._sum_shader: WeightedSum = WeightedSum()

    @property
    def trail(self) -> Texture:
        """The trail, xyz in image widths per second (RGBA16F)."""
        return self._output

    @property
    def trail_acceleration(self) -> Texture:
        """The trail's rate, xyz in image widths per second² (RGBA16F), as of the last update that asked for it."""
        return self._acceleration_fbo

    @property
    def velocity_input(self) -> Texture:
        """The velocity the trail reads."""
        return self._input

    @property
    def ports(self) -> dict[str, Port]:
        return {'input': Port(self._input_fbo, TextureKind.VELOCITY),
                'trail': Port(self._output_fbo, TextureKind.VELOCITY),
                'trail_acceleration': Port(self._acceleration_fbo, TextureKind.VELOCITY)}

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
        self._acceleration_fbo.allocate(self._output_fbo.width, self._output_fbo.height, GL_RGBA16F)
        self._acceleration_fbo.set_wrap(GL_CLAMP_TO_BORDER, (0.0, 0.0, 0.0, 0.0))
        FlowUtil.zero(self._acceleration_fbo)
        self._advect_shader.allocate()
        self._trail_shader.allocate()
        self._blur_shader.allocate()
        self._sum_shader.allocate()

    def deallocate(self) -> None:
        """Deallocate all resources."""
        super().deallocate()
        self._acceleration_fbo.deallocate()
        self._advect_shader.deallocate()
        self._trail_shader.deallocate()
        self._blur_shader.deallocate()
        self._sum_shader.deallocate()

    def reset(self) -> None:
        """Reset the trail and its rate to zero."""
        super().reset()
        FlowUtil.zero(self._acceleration_fbo)

    def set_velocity(self, velocity: Texture) -> None:
        """Take a flow's 3D velocity as the trail's next input."""
        if not self._allocated:
            return
        FlowUtil.blit(self._input_fbo, velocity)

    def update(self, delta_time: float = 1.0 / 60.0, acceleration: bool = False) -> None:
        """Advance the trail with the input by `delta_time` (s): temporal, then spatial smoothing; with
        `acceleration`, also its rate."""
        if not self._allocated:
            return
        s: VelocityLowpassSettings = self.settings

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
        release: float = decay_weight(delta_time, s.smooth)
        attack: float = decay_weight(delta_time, s.attack) if s.attack > 0.0 else release
        self._output_fbo.swap()
        self._output_fbo.begin()
        self._trail_shader.use(self._output_fbo.back, self._input_fbo, attack, release)
        self._output_fbo.end()

        # The rate: the Ema's step over dt, new minus previous (advected)
        if acceleration:
            rate: float = 1.0 / delta_time if delta_time > 0.0 else 0.0
            self._acceleration_fbo.begin()
            self._sum_shader.use([(self._output_fbo, rate), (self._output_fbo.back, -rate)])
            self._acceleration_fbo.end()

        # Stage 3: Spatial smoothing (Gaussian blur), `blur_steps` passes, the rate with the trail
        if s.blur_radius > 0.0:
            blurred: tuple[SwapFbo, ...] = (self._output_fbo, self._acceleration_fbo) if acceleration else (self._output_fbo,)
            for _ in range(s.blur_steps):
                for horizontal in (True, False):
                    for fbo in blurred:
                        fbo.swap()
                        fbo.begin()
                        self._blur_shader.use(fbo.back, s.blur_radius, horizontal=horizontal)
                        fbo.end()


# ── numpy reference ─────────────────────────────────────────────────────

def velocity_trail(previous: np.ndarray, velocity: np.ndarray, attack: float, release: float) -> np.ndarray:
    """One step of the trail (..., 3): by `attack` where the 3D speed rises above the trail's, else by `release`.
    Mirrors velocity_trail.frag."""
    rising: np.ndarray = np.linalg.norm(velocity, axis=-1, keepdims=True) > np.linalg.norm(previous, axis=-1, keepdims=True)
    weight: np.ndarray = np.where(rising, attack, release)
    return weight * previous + (1.0 - weight) * velocity


def trail_acceleration(previous: np.ndarray, trail: np.ndarray, delta_time: float) -> np.ndarray:
    """The trail's rate (..., 3), widths/s², from the trail before (advected) and after one step. Mirrors the
    lowpass's rate pass."""
    return (trail - previous) / delta_time if delta_time > 0.0 else np.zeros_like(trail)
