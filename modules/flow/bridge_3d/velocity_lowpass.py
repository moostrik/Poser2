"""Velocity Lowpass.

The bridge's lowpass: a flow's 3D velocity in image widths per second, smoothed in time and in space, the
reading `trail`. Its output stays in widths per second; how strongly the fluid takes it is the fluid's.

Before each step the trail moves with the motion it holds (`advect`): each pixel takes the trail from
where its own x/y says it came from, so the average follows a moving limb instead of a fixed pixel,
and a fast move is not cut off by a smoothing time longer than it takes to cross.

The smoothing is one unit-gain Ema per pixel. Where the 3D speed rises it follows over `attack`, else over
`smooth`, so a flick reaches the fluid at full strength and a stop still coasts. x, y and z alike: z's noise
is the motion's to filter (its depth velocity's `smooth`).

The blur runs inside the Ema's loop, on the trail each step feeds back, so blurring every step is diffusion:
`blur` is how far the trail spreads in one second, in widths at half its height, and each step blurs by
sigma = blur / 2.355 × sqrt(dt) widths (σ² = 2 D dt), the same spread at any frame rate and resolution; what
is seen also grows with the trail's memory. The blur is the textbook Gaussian (`gaussian_blur`).

The trail holds the history of the motion: its velocity and its z position. The z position is where the
motion it holds was: the motion's z position (`set_z_position`) through the same Ema step, weighted by the
speed each step brings, moved and blurred with the trail (red the z position times its weight, green the
weight). So an arm leaving the torso leaves its trail at the arm's depth, as the still torso brings no speed,
and a z position whose motion has died counts for nothing. One dispatch moves, steps and rates the trail
and its z position (`LowpassStep`), one blurs them (`LowpassBlur`).

The trail is read in two units:
    trail                the trail, widths/s
    trail_acceleration   the trail's rate, widths/s²: its Ema step over dt, (1 − w)(velocity − trail) / dt,
                         which is (velocity − trail) / τ for a dt well under τ, τ the attack where the speed
                         rises, else the smooth: the highpass over the lowpass's time, a spike at a start, a
                         brake over the coast at a stop. Its sum × dt is the trail's change, at any frame rate.
                         Advected with the trail and blurred as it is; written only when asked
"""

import math

import numpy as np
from OpenGL.GL import GL_CLAMP_TO_BORDER, GL_RG32F, GL_RGBA16F

from modules.gl import SwapFbo, Texture
from modules.settings import Field, BaseSettings
from .. import FlowBase, FlowUtil, Port, TextureKind
from .trails import decay_weight
from .shaders import LowpassBlur, LowpassStep

FULL_WIDTH_HALF_MAXIMUM: float = 2.0 * math.sqrt(2.0 * math.log(2.0))     # a Gaussian's width at half its height, in sigmas


class VelocityLowpassSettings(BaseSettings):
    """Configuration for the velocity lowpass."""
    advect = Field(0.0, min=0.0, max=1.0, step=0.05, description="How far the trail moves with its motion before each step; 0 = at a fixed pixel")
    attack = Field(0.0, min=0.0, max=500.0, step=1.0, description="Smoothing time of a rising speed, ms; 0 = as smooth (lower: quicker flicks)")
    smooth = Field(160.0, min=0.0, max=3000.0, step=1.0, description="Smoothing time of a falling speed, ms; 0 = none (higher: longer coast)")
    blur = Field(0.0, min=0.0, max=0.3, step=0.001, newline=True, description="How far the trail spreads in 1 s, widths at half its height; 0 = off")


class VelocityLowpass(FlowBase):
    """The velocity lowpass: a flow's 3D velocity through one 3D trail, read as a velocity and an acceleration.

    Pipeline:
        1. Take the velocity into the input FBO (`set_velocity`)
        2. Move the previous trail and its z position with its own motion (`advect`), then blend the input in, a
           unit-gain Ema over `attack` or `smooth` (temporal smoothing); with `acceleration`, the step over dt
        3. Blur the trail, its z position and the acceleration with them, by this step's share of `blur` (spatial diffusion)

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
        self._z_position_fbo: SwapFbo = SwapFbo()       # the trail's z position, red over green (RG32F)
        self._current: Texture | None = None            # the motion's z position now

        # Shaders
        self._step_shader: LowpassStep = LowpassStep()
        self._blur_shader: LowpassBlur = LowpassBlur()

    @property
    def trail(self) -> Texture:
        """The trail, xyz in image widths per second (RGBA16F)."""
        return self._output

    @property
    def trail_acceleration(self) -> Texture:
        """The trail's rate, xyz in image widths per second² (RGBA16F), as of the last update that asked for it."""
        return self._acceleration_fbo

    @property
    def z_position(self) -> Texture:
        """Where the motion the trail holds was, 0 front to 1 back, times its weight in red, the weight (its
        speed) in green (RG32F)."""
        return self._z_position_fbo

    @property
    def velocity_input(self) -> Texture:
        """The velocity the trail reads."""
        return self._input

    @property
    def ports(self) -> dict[str, Port]:
        return {'input': Port(self._input_fbo, TextureKind.VELOCITY),
                'trail': Port(self._output_fbo, TextureKind.VELOCITY),
                'trail_acceleration': Port(self._acceleration_fbo, TextureKind.ACCELERATION),
                'z_position': Port(self._z_position_fbo, TextureKind.Z_POSITION)}

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
        self._z_position_fbo.allocate(self._output_fbo.width, self._output_fbo.height, GL_RG32F)
        self._z_position_fbo.set_wrap(GL_CLAMP_TO_BORDER, (0.0, 0.0, 0.0, 0.0))
        FlowUtil.zero(self._z_position_fbo)
        self._step_shader.allocate()
        self._blur_shader.allocate()

    def deallocate(self) -> None:
        """Deallocate all resources."""
        super().deallocate()
        self._acceleration_fbo.deallocate()
        self._z_position_fbo.deallocate()
        self._step_shader.deallocate()
        self._blur_shader.deallocate()

    def reset(self) -> None:
        """Reset the trail, its rate and its z position to zero."""
        super().reset()
        FlowUtil.zero(self._acceleration_fbo)
        FlowUtil.zero(self._z_position_fbo)

    def set_velocity(self, velocity: Texture) -> None:
        """Take a flow's 3D velocity as the trail's next input."""
        if not self._allocated:
            return
        FlowUtil.blit(self._input_fbo, velocity)

    def set_z_position(self, z_position: Texture | None) -> None:
        """The motion's z position now (red over green), what the step takes in with the velocity; None: none."""
        self._current = z_position

    def update(self, delta_time: float = 1.0 / 60.0, acceleration: bool = False) -> None:
        """Advance the trail with the input by `delta_time` (s): temporal, then spatial smoothing; with
        `acceleration`, also its rate."""
        if not self._allocated:
            return
        s: VelocityLowpassSettings = self.settings

        # Stages 1 and 2 in one dispatch: the trail and its z position move with its motion (x is widths/s, y widths/s
        # on the x scale, so uv per step is v * dt, and v * dt * width / height on y), then one unit-gain Ema step
        # per pixel, weight * previous + (1 - weight) * input; with `acceleration`, the step over dt
        trail: SwapFbo = self._output_fbo
        step: float = s.advect * delta_time
        release: float = decay_weight(delta_time, s.smooth)
        attack: float = decay_weight(delta_time, s.attack) if s.attack > 0.0 else release
        rate: float = 1.0 / delta_time if delta_time > 0.0 else 0.0
        trail.swap()
        self._z_position_fbo.swap()
        self._step_shader.use(trail.back, self._z_position_fbo.back, self._input_fbo, self._current,
                              trail, self._z_position_fbo, self._acceleration_fbo if acceleration else None,
                              (step, step * trail.width / max(trail.height, 1)), attack, release, rate)

        # Stage 3: Spatial diffusion, this step's sigma, the trail, the rate and the z position in one dispatch
        sigma: float = blur_sigma(s.blur, delta_time, trail.width)
        if sigma > 0.0:
            blurred: list[SwapFbo | None] = [trail, self._acceleration_fbo if acceleration else None, self._z_position_fbo]
            count, each = blur_dispatches(sigma, LowpassBlur.MAX_RADIUS)
            weights: list[float] = gaussian_weights(each).tolist()
            for _ in range(count):
                for fbo in blurred:
                    if fbo is not None:
                        fbo.swap()
                self._blur_shader.use([fbo.back if fbo is not None else None for fbo in blurred], list(blurred), weights)


# ── numpy reference ─────────────────────────────────────────────────────

def z_position_trail(previous: np.ndarray, previous_trail: np.ndarray, velocity: np.ndarray, current: np.ndarray,
                     attack: float, release: float) -> np.ndarray:
    """One step of the trail's z position (..., 2, red the z position times its weight, green the weight): the same
    Ema step as the trail's, by `attack` where the 3D speed rises above the trail's, else by `release`, on the
    motion's z position `current` weighted by the speed it brings. Mirrors lowpass_step.comp."""
    speed: np.ndarray = np.linalg.norm(velocity, axis=-1, keepdims=True)
    rising: np.ndarray = speed > np.linalg.norm(previous_trail, axis=-1, keepdims=True)
    weight: np.ndarray = np.where(rising, attack, release)
    return weight * previous + (1.0 - weight) * speed * current


def blur_sigma(blur: float, delta_time: float, width: int) -> float:
    """One step's sigma in pixels: a spread of `blur` widths at half its height in one second, as diffusion
    (σ² = 2 D dt), over a step of `delta_time` s, on a field `width` pixels wide; 0 without a spread or a step."""
    if blur <= 0.0 or delta_time <= 0.0 or not math.isfinite(delta_time):
        return 0.0
    return blur / FULL_WIDTH_HALF_MAXIMUM * math.sqrt(delta_time) * width


def blur_dispatches(sigma: float, max_radius: int) -> tuple[int, float]:
    """How many blurs of which sigma make one of `sigma` with each reaching 3 sigmas within `max_radius` taps:
    k blurs of sigma / sqrt(k) are one of sigma."""
    count: int = max(1, math.ceil((3.0 * sigma / max_radius) ** 2))
    return count, sigma / math.sqrt(count)


def gaussian_weights(sigma: float) -> np.ndarray:
    """The textbook Gaussian's taps k = 0..ceil(3 sigma): exp(-k² / 2σ²), normalized over both sides."""
    if sigma <= 0.0:
        return np.ones(1)
    radius: int = math.ceil(3.0 * sigma)
    k: np.ndarray = np.arange(radius + 1, dtype=np.float64)
    weights: np.ndarray = np.exp(-k * k / (2.0 * sigma * sigma))
    return weights / (weights[0] + 2.0 * weights[1:].sum())


def gaussian_blur(field: np.ndarray, sigma: float) -> np.ndarray:
    """`field` (H, W, C) blurred by the Gaussian of `sigma` pixels, separable, 0 outside. Mirrors lowpass_blur.comp."""
    weights: np.ndarray = gaussian_weights(sigma)
    kernel: np.ndarray = np.concatenate((weights[:0:-1], weights))
    radius: int = len(weights) - 1
    padded: np.ndarray = np.pad(field.astype(np.float64), ((radius, radius), (radius, radius), (0, 0)))
    rows: np.ndarray = sum(kernel[i] * padded[:, i:i + field.shape[1]] for i in range(len(kernel)))
    return sum(kernel[i] * rows[i:i + field.shape[0]] for i in range(len(kernel)))


def velocity_trail(previous: np.ndarray, velocity: np.ndarray, attack: float, release: float) -> np.ndarray:
    """One step of the trail (..., 3): by `attack` where the 3D speed rises above the trail's, else by `release`.
    Mirrors lowpass_step.comp."""
    rising: np.ndarray = np.linalg.norm(velocity, axis=-1, keepdims=True) > np.linalg.norm(previous, axis=-1, keepdims=True)
    weight: np.ndarray = np.where(rising, attack, release)
    return weight * previous + (1.0 - weight) * velocity


def trail_acceleration(previous: np.ndarray, trail: np.ndarray, delta_time: float) -> np.ndarray:
    """The trail's rate (..., 3), widths/s², from the trail before (advected) and after one step. Mirrors the
    lowpass's rate pass."""
    return (trail - previous) / delta_time if delta_time > 0.0 else np.zeros_like(trail)
