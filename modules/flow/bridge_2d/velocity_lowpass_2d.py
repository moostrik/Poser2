"""Velocity Lowpass 2D, the 2D bridge's lowpass: the flow's 2D velocity advected, smoothed over `attack` and `smooth`
and blurred, the reading `trail` (widths/s, RG16F) and its rate `trail_acceleration` (widths/s², written only when
asked). One dispatch moves, steps and rates it (`LowpassStep2D`), one blurs them (`LowpassBlur2D`). How and why:
modules/flow/BRIDGE_2D.md.
"""

from OpenGL.GL import GL_CLAMP_TO_BORDER, GL_RG16F

from modules.gl import SwapFbo, Texture
from .. import FlowBase, FlowUtil, Port, TextureKind
from ..bridge import VelocityLowpassSettings, blur_dispatches, blur_sigma, decay_weight, gaussian_weights
from .shaders import LowpassBlur2D, LowpassStep2D


class VelocityLowpass2D(FlowBase):
    """The 2D velocity lowpass: a flow's 2D velocity through one trail, read as a velocity and an acceleration.

    Pipeline:
        1. Take the velocity into the input FBO (`set_velocity`)
        2. Move the previous trail with its own motion (`advect`), then blend the input in, a unit-gain Ema over
           `attack` or `smooth` (temporal smoothing); with `acceleration`, the step over dt
        3. Blur the trail and the acceleration with it, by this step's share of `blur` (spatial diffusion)

    FBOs (from FlowBase):
        - input_fbo: The velocity
        - output_fbo: The trail
    """

    def __init__(self, settings: VelocityLowpassSettings | None = None) -> None:
        super().__init__()

        self._input_internal_format = GL_RG16F
        self._output_internal_format = GL_RG16F
        self._input_kind = TextureKind.VELOCITY
        self._output_kind = TextureKind.VELOCITY

        self.settings: VelocityLowpassSettings = settings or VelocityLowpassSettings()
        self._acceleration_fbo: SwapFbo = SwapFbo()     # the trail's rate, widths/s² (RG16F, xy)

        self._step_shader: LowpassStep2D = LowpassStep2D()
        self._blur_shader: LowpassBlur2D = LowpassBlur2D()

    @property
    def trail(self) -> Texture:
        """The trail, xy in image widths per second (RG16F)."""
        return self._output

    @property
    def trail_acceleration(self) -> Texture:
        """The trail's rate, xy in image widths per second² (RG16F), as of the last update that asked for it."""
        return self._acceleration_fbo

    @property
    def ports(self) -> dict[str, Port]:
        return {'input': Port(self._input_fbo, TextureKind.VELOCITY),
                'trail': Port(self._output_fbo, TextureKind.VELOCITY),
                'trail_acceleration': Port(self._acceleration_fbo, TextureKind.ACCELERATION)}

    def allocate(self, width: int, height: int, output_width: int | None = None, output_height: int | None = None) -> None:
        """Allocate the trail at `width` by `height`; the output size is the trail's own."""
        super().allocate(width, height, output_width, output_height)
        # No velocity outside the image, as in the fluid: a trail that leaves it is gone, and the edge doesn't copy
        # itself back in through the advect and the blur
        self._output_fbo.set_wrap(GL_CLAMP_TO_BORDER, (0.0, 0.0, 0.0, 0.0))
        self._acceleration_fbo.allocate(self._output_fbo.width, self._output_fbo.height, GL_RG16F)
        self._acceleration_fbo.set_wrap(GL_CLAMP_TO_BORDER, (0.0, 0.0, 0.0, 0.0))
        FlowUtil.zero(self._acceleration_fbo)
        self._step_shader.allocate()
        self._blur_shader.allocate()

    def deallocate(self) -> None:
        super().deallocate()
        self._acceleration_fbo.deallocate()
        self._step_shader.deallocate()
        self._blur_shader.deallocate()

    def reset(self) -> None:
        """Reset the trail and its rate to zero."""
        super().reset()
        FlowUtil.zero(self._acceleration_fbo)

    def set_velocity(self, velocity: Texture) -> None:
        """Take a flow's 2D velocity as the trail's next input."""
        if not self._allocated:
            return
        FlowUtil.blit(self._input_fbo, velocity)

    def update(self, delta_time: float = 1.0 / 60.0, acceleration: bool = False) -> None:
        """Advance the trail with the input by `delta_time` (s): temporal, then spatial smoothing; with
        `acceleration`, also its rate."""
        if not self._allocated:
            return
        s: VelocityLowpassSettings = self.settings

        # Stages 1 and 2 in one dispatch: the trail moves with its motion (x is widths/s, y widths/s on the x scale,
        # so uv per step is v * dt, and v * dt * width / height on y), then one unit-gain Ema step per pixel,
        # weight * previous + (1 - weight) * input; with `acceleration`, the step over dt
        trail: SwapFbo = self._output_fbo
        step: float = s.advect * delta_time
        release: float = decay_weight(delta_time, s.smooth)
        attack: float = decay_weight(delta_time, s.attack) if s.attack > 0.0 else release
        rate: float = 1.0 / delta_time if delta_time > 0.0 else 0.0
        trail.swap()
        self._step_shader.use(trail.back, self._input_fbo, trail, self._acceleration_fbo if acceleration else None,
                              (step, step * trail.width / max(trail.height, 1)), attack, release, rate)

        # Stage 3: Spatial diffusion, this step's sigma, the trail and the rate in one dispatch
        sigma: float = blur_sigma(s.blur, delta_time, trail.width)
        if sigma > 0.0:
            blurred: list[SwapFbo | None] = [trail, self._acceleration_fbo if acceleration else None]
            count, each = blur_dispatches(sigma, LowpassBlur2D.MAX_RADIUS)
            weights: list[float] = gaussian_weights(each).tolist()
            for _ in range(count):
                for fbo in blurred:
                    if fbo is not None:
                        fbo.swap()
                self._blur_shader.use([fbo.back if fbo is not None else None for fbo in blurred], list(blurred), weights)
