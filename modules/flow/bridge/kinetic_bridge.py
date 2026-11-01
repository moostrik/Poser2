"""Kinetic Bridge.

Reads the motion's change instead of its speed: two trails on the optical flow, fast and slow, and
their difference projected on the direction of the motion. Runs on each new optical flow frame.
"""
from enum import IntEnum
import math

from OpenGL.GL import GL_LINEAR, GL_LINEAR_MIPMAP_LINEAR, GL_R16F, GL_R32F, GL_RG16F, GL_RGBA16F, GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MIN_FILTER, glBindTexture, glGenerateMipmap, glTexParameteri

from modules.gl import Fbo, SwapFbo, Texture
from modules.settings import Field, BaseSettings
from .. import FlowBase, FlowUtil
from ..shaders import Trail, GaussianBlur
from .kinetics import decay_weight
from .shaders import KineticCurl, KineticEmission, KineticHold, KineticInput, KineticOnset, KineticPatch, KineticProjection, KineticReservoir, KineticVelocity, Projection


class BlurPosition(IntEnum):
    BEFORE = 0      # blur the optical flow once, then both trails read it
    INSIDE = 1      # blur inside each trail's feedback loop, as the trail bridge does


class KineticBridgeSettings(BaseSettings):
    """Configuration for the kinetic bridge."""
    scale = Field(10.0, min=-50.0, max=50.0, description="Converts optical flow to simulation velocity")
    fast_time = Field(25.0, min=0.0, max=500.0, description="Fast trail time constant (ms)")
    slow_time = Field(150.0, min=0.0, max=1000.0, description="Slow trail time constant (ms)")
    blur_position: Field[BlurPosition] = Field(BlurPosition.BEFORE, description="Blur the flow once before the trails, or inside each trail")
    blur_radius = Field(2.0, min=0.0, max=10.0, description="Gaussian blur radius in pixels")
    blur_steps = Field(2, min=0, max=8, description="Number of Gaussian blur passes")
    stop_gate = Field(0.0, min=0.0, max=1.0, description="How far a release must be a slowdown of its patch; 0 lets all through")
    stop_tile = Field(128.0, min=8.0, max=1024.0, description="Size of a stop gate patch (pixels of the bridge)")
    release_hold = Field(100.0, min=0.0, max=1000.0, description="Decay of the held release (ms)")
    emission_gain = Field(1.0, min=0.0, max=20.0, description="Ink emitted per unit of held release")
    drop_velocity = Field(1.0, min=0.0, max=1.0, description="How far a release blends the velocity to the slow trail")
    onset_force = Field(0.5, min=0.0, max=10.0, description="Velocity added along the motion per unit of onset")
    spin = Field(0.0, min=0.0, max=60.0, description="Vorticity force from the motion's own curl at the injection depth")
    reservoir_enabled = Field(False, description="Motion stores ink per pixel; a release spends it")
    reservoir_leak = Field(4.0, min=0.1, max=60.0, description="Time constant of the reservoir's leak (s)")


class KineticBridge(FlowBase):
    """Two trails on the optical flow and their difference along and across the motion.

    The trails are unit-gain Emas, so they meet on a steady motion and fast - slow is the change alone.
    The gain of the trail bridge, 1 / (1 - trail_weight), is in `scale` here. With a depth velocity
    the trails are 3D, so a motion that starts or stops toward the camera reads like any other.

    Pipeline:
        1. Receive the optical flow and the depth velocity via set_velocity() → input_fbo (RGBA16F, xyz)
        2. Blur it (BlurPosition.BEFORE)
        3. Advance the fast and slow trails, blurring each inside its loop (BlurPosition.INSIDE)
        4. Project fast - slow on the direction of the motion → acceleration, turn (R16F)
        5. Peak-hold the release, max(0, -acceleration) → emission (R16F), the density bridge's alpha,
           through the stop gate where that is on: patch mean speeds (RG16F, mipmapped) say a patch is slowing;
           with the reservoir (R32F) motion stores the ink per pixel and the release spends it
        6. Blend the fast trail toward the slow one where it releases → velocity (RG16F), depth_velocity (R16F)
        7. max(0, acceleration) along the fast trail → onset (RGBA16F, xyz), a second velocity input
        8. 2D curl of the fast trail → curl (R16F), the extra input to the fluid's vorticity force
    """

    def __init__(self, settings: KineticBridgeSettings | None = None) -> None:
        super().__init__()

        self.settings: KineticBridgeSettings = settings or KineticBridgeSettings()

        self._input_internal_format = GL_RGBA16F
        self._output_internal_format = GL_RG16F

        self._has_depth: bool = False

        self._fast_fbo: SwapFbo = SwapFbo()
        self._slow_fbo: SwapFbo = SwapFbo()
        self._acceleration_fbo: Fbo = Fbo()
        self._turn_fbo: Fbo = Fbo()
        self._depth_velocity_fbo: Fbo = Fbo()
        self._depth_acceleration_fbo: Fbo = Fbo()
        self._patch_fbo: Fbo = Fbo()
        self._held_release_fbo: SwapFbo = SwapFbo()
        self._reservoir_fbo: SwapFbo = SwapFbo()
        self._emission_fbo: Fbo = Fbo()
        self._onset_fbo: Fbo = Fbo()
        self._curl_fbo: Fbo = Fbo()

        self._input_shader: KineticInput = KineticInput()
        self._trail_shader: Trail = Trail()
        self._blur_shader: GaussianBlur = GaussianBlur()
        self._projection_shader: KineticProjection = KineticProjection()
        self._patch_shader: KineticPatch = KineticPatch()
        self._hold_shader: KineticHold = KineticHold()
        self._reservoir_shader: KineticReservoir = KineticReservoir()
        self._emission_shader: KineticEmission = KineticEmission()
        self._velocity_shader: KineticVelocity = KineticVelocity()
        self._onset_shader: KineticOnset = KineticOnset()
        self._curl_shader: KineticCurl = KineticCurl()

    @property
    def velocity(self) -> Texture:
        """Velocity output for the fluid (RG16F)."""
        return self._output

    @property
    def velocity_input(self) -> Texture:
        """Optical flow input buffer, blurred when BlurPosition.BEFORE."""
        return self._input

    @property
    def depth_velocity(self) -> Texture:
        """Velocity toward the camera for the fluid (R16F)."""
        return self._depth_velocity_fbo

    @property
    def fast(self) -> Texture:
        """Fast trail (RGBA16F, xyz)."""
        return self._fast_fbo.texture

    @property
    def slow(self) -> Texture:
        """Slow trail (RGBA16F, xyz)."""
        return self._slow_fbo.texture

    @property
    def depth_acceleration(self) -> Texture:
        """The z of fast - slow (R16F), positive when the motion toward the camera grows."""
        return self._depth_acceleration_fbo

    @property
    def acceleration(self) -> Texture:
        """fast - slow along the motion (R16F): + onset, - release."""
        return self._acceleration_fbo

    @property
    def turn(self) -> Texture:
        """fast - slow across the motion (R16F), signed."""
        return self._turn_fbo

    @property
    def held_release(self) -> Texture:
        """The peak-held release (R16F)."""
        return self._held_release_fbo.texture

    @property
    def reservoir(self) -> Texture:
        """Ink stored per pixel (R32F)."""
        return self._reservoir_fbo.texture

    @property
    def emission(self) -> Texture:
        """Ink emitted (R16F), the density bridge's alpha."""
        return self._emission_fbo

    @property
    def onset(self) -> Texture:
        """The push of a starting motion (RGBA16F, xyz), a second velocity input for the fluid."""
        return self._onset_fbo

    @property
    def curl(self) -> Texture:
        """2D curl of the fast trail (R16F)."""
        return self._curl_fbo

    def allocate(self, width: int, height: int, output_width: int | None = None, output_height: int | None = None) -> None:
        """Allocate all fields at the bridge resolution.

        Args:
            width: Field width
            height: Field height
            output_width: Optional velocity output width (defaults to width)
            output_height: Optional velocity output height (defaults to height)
        """
        super().allocate(width, height, output_width, output_height)
        self._fast_fbo.allocate(width, height, GL_RGBA16F)
        self._slow_fbo.allocate(width, height, GL_RGBA16F)
        self._acceleration_fbo.allocate(width, height, GL_R16F)
        self._turn_fbo.allocate(width, height, GL_R16F)
        self._depth_velocity_fbo.allocate(width, height, GL_R16F)
        self._depth_acceleration_fbo.allocate(width, height, GL_R16F)
        self._patch_fbo.allocate(width // 2, height // 2, GL_RG16F)
        glBindTexture(GL_TEXTURE_2D, self._patch_fbo.tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glGenerateMipmap(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, 0)
        self._held_release_fbo.allocate(width, height, GL_R16F)
        self._reservoir_fbo.allocate(width, height, GL_R32F)        # a slow fill vanishes in 16 bits
        self._emission_fbo.allocate(width, height, GL_R16F)
        self._onset_fbo.allocate(width, height, GL_RGBA16F)
        self._curl_fbo.allocate(width, height, GL_R16F)
        self.reset()

        self._input_shader.allocate()
        self._trail_shader.allocate()
        self._blur_shader.allocate()
        self._projection_shader.allocate()
        self._patch_shader.allocate()
        self._hold_shader.allocate()
        self._reservoir_shader.allocate()
        self._emission_shader.allocate()
        self._velocity_shader.allocate()
        self._onset_shader.allocate()
        self._curl_shader.allocate()

    def deallocate(self) -> None:
        """Release all resources."""
        super().deallocate()
        self._fast_fbo.deallocate()
        self._slow_fbo.deallocate()
        self._acceleration_fbo.deallocate()
        self._turn_fbo.deallocate()
        self._depth_velocity_fbo.deallocate()
        self._depth_acceleration_fbo.deallocate()
        self._patch_fbo.deallocate()
        self._held_release_fbo.deallocate()
        self._reservoir_fbo.deallocate()
        self._emission_fbo.deallocate()
        self._onset_fbo.deallocate()
        self._curl_fbo.deallocate()
        self._input_shader.deallocate()
        self._trail_shader.deallocate()
        self._blur_shader.deallocate()
        self._projection_shader.deallocate()
        self._patch_shader.deallocate()
        self._hold_shader.deallocate()
        self._reservoir_shader.deallocate()
        self._emission_shader.deallocate()
        self._velocity_shader.deallocate()
        self._onset_shader.deallocate()
        self._curl_shader.deallocate()

    def reset(self) -> None:
        """Reset all fields to zero."""
        super().reset()
        FlowUtil.zero(self._fast_fbo)
        FlowUtil.zero(self._slow_fbo)
        FlowUtil.zero(self._acceleration_fbo)
        FlowUtil.zero(self._turn_fbo)
        FlowUtil.zero(self._depth_velocity_fbo)
        FlowUtil.zero(self._depth_acceleration_fbo)
        FlowUtil.zero(self._held_release_fbo)
        FlowUtil.zero(self._reservoir_fbo)
        FlowUtil.zero(self._emission_fbo)
        FlowUtil.zero(self._onset_fbo)
        FlowUtil.zero(self._curl_fbo)

    def set_velocity(self, velocity: Texture, depth_velocity: Texture | None = None) -> None:
        """Set the new optical flow frame, and the velocity toward the camera if there is depth."""
        self._has_depth = depth_velocity is not None
        self._input_fbo.begin()
        self._input_shader.use(velocity, depth_velocity)
        self._input_fbo.end()

    def update(self, delta_time: float = 1.0 / 60.0) -> None:
        """Advance on a new optical flow frame.

        Args:
            delta_time: Seconds since the previous optical flow frame
        """
        if not self._allocated:
            return

        s: KineticBridgeSettings = self.settings
        blur_before: bool = s.blur_position == BlurPosition.BEFORE
        if blur_before:
            self._blur(self._input_fbo)

        self._advance_trail(self._fast_fbo, decay_weight(delta_time, s.fast_time), not blur_before)
        self._advance_trail(self._slow_fbo, decay_weight(delta_time, s.slow_time), not blur_before)

        self._acceleration_fbo.begin()
        self._projection_shader.use(self.fast, self.slow, Projection.ACCELERATION)
        self._acceleration_fbo.end()

        self._turn_fbo.begin()
        self._projection_shader.use(self.fast, self.slow, Projection.TURN)
        self._turn_fbo.end()

        if s.stop_gate > 0.0:
            self._patch_fbo.begin()
            self._patch_shader.use(self.fast, self.slow)
            self._patch_fbo.end()
            glBindTexture(GL_TEXTURE_2D, self._patch_fbo.tex_id)
            glGenerateMipmap(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, 0)

        self._held_release_fbo.swap()
        self._held_release_fbo.begin()
        self._hold_shader.use(self.acceleration, self._held_release_fbo.back_texture, decay_weight(delta_time, s.release_hold),
                              self._patch_fbo, s.stop_gate, math.log2(s.stop_tile) - 1.0)      # the patch texture is half size
        self._held_release_fbo.end()

        if s.reservoir_enabled:
            self._reservoir_fbo.swap()
            self._reservoir_fbo.begin()
            self._reservoir_shader.use(self._reservoir_fbo.back_texture, self.fast, self.held_release,
                                       decay_weight(delta_time, s.reservoir_leak * 1000.0), delta_time, s.emission_gain)
            self._reservoir_fbo.end()

        self._emission_fbo.begin()
        self._emission_shader.use(self.held_release, self.reservoir, s.emission_gain, s.reservoir_enabled)
        self._emission_fbo.end()

        self._output_fbo.begin()
        self._velocity_shader.use(self.fast, self.slow, self.acceleration, s.drop_velocity)
        self._output_fbo.end()

        if self._has_depth:
            self._depth_velocity_fbo.begin()
            self._velocity_shader.use(self.fast, self.slow, self.acceleration, s.drop_velocity, True)
            self._depth_velocity_fbo.end()

        if s.onset_force > 0.0:
            self._onset_fbo.begin()
            self._onset_shader.use(self.fast, self.acceleration, s.onset_force)
            self._onset_fbo.end()

    def update_depth_acceleration(self) -> None:
        """Compute the z of fast - slow, for its debug view."""
        if not self._allocated:
            return
        self._depth_acceleration_fbo.begin()
        self._projection_shader.use(self.fast, self.slow, Projection.DEPTH_ACCELERATION)
        self._depth_acceleration_fbo.end()

    def update_curl(self) -> None:
        """Compute the curl of the fast trail, for the spin input or its debug view."""
        if not self._allocated:
            return
        self._curl_fbo.begin()
        self._curl_shader.use(self.fast)
        self._curl_fbo.end()

    def _advance_trail(self, trail: SwapFbo, weight: float, blur: bool) -> None:
        """One unit-gain Ema step: weight * previous + (1 - weight) * scale * input."""
        trail.swap()
        trail.begin()
        self._trail_shader.use(trail.back_texture, self._input, weight, (1.0 - weight) * self.settings.scale)
        trail.end()
        if blur:
            self._blur(trail)

    def _blur(self, field: SwapFbo) -> None:
        """Gaussian blur in place."""
        radius: float = self.settings.blur_radius
        if radius <= 0.0:
            return
        for _ in range(self.settings.blur_steps):
            for horizontal in (True, False):
                field.swap()
                field.begin()
                self._blur_shader.use(field.back_texture, radius, horizontal)
                field.end()
