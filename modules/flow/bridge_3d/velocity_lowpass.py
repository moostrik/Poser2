"""Velocity Lowpass, the bridge's lowpass: the flow's 3D velocity advected, smoothed over `attack` and `smooth`
and blurred, the reading `trail` (widths/s), its rate `trail_acceleration` (widths/s², written only when asked)
and its z position (red the z position times its weight, green the weight). One dispatch moves, steps and rates
them (`LowpassStep`), one blurs them (`LowpassBlur`). How and why: modules/flow/BRIDGE_3D.md, "The trail" and
"The z positions".
"""

from OpenGL.GL import GL_CLAMP_TO_BORDER, GL_RG32F, GL_RGBA16F

from modules.gl import SwapFbo, Texture
from .. import FlowBase, FlowUtil, Port, TextureKind
from ..bridge import VelocityLowpassSettings, blur_dispatches, blur_sigma, decay_weight, gaussian_weights
from .shaders import LowpassBlur, LowpassStep


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


