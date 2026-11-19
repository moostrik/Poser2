"""Velocity Bandpass 2D, the 2D bridge's bandpass: the motion's changes of speed from a fast and a slow trail of the
flow's 2D velocity, split by their regions. Out come the readings `change`, `start`, `bow`, `stop`, `wake`, each as a
velocity (widths/s, RG16F xy) and, when asked, an acceleration (widths/s²), and each part's size `<part>_speed`
(R16F, widths/s). One dispatch steps both trails (`BandpassTrails2D`); it steps every tick on its dt while it runs.
How and why: modules/flow/BRIDGE_2D.md.
"""

import logging
import math

from OpenGL.GL import GL_LINEAR, GL_LINEAR_MIPMAP_LINEAR, GL_R16F, GL_RG16F, GL_RGBA16F, GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MIN_FILTER, glBindTexture, glGenerateMipmap, glTexParameteri

from modules.gl import Fbo, SwapFbo, Texture, shaders
from .. import FlowBase, FlowUtil, Port, TextureKind
from ..bridge import Change, VelocityBandpassSettings, change_acceleration_scale, decay_weight, step_peak
from .shaders import BandpassAccelerations2D, BandpassChanges2D, BandpassTrails2D, VelocityBandpassRegion2D, VelocityBandpassRegionMean2D, VelocityBandpassRegionSpeed2D

logger = logging.getLogger(__name__)


class VelocityBandpass2D(FlowBase):
    """Two short trails on the flow's 2D velocity and the starts and stops of their difference.

    Pipeline:
        1. Take the flow's velocity → input_fbo (RG16F, xy)
        2. Advance the fast and slow trails, unit-gain Emas
        3. With a start region weight, the flow's velocity where a pixel moves and whether it moves (half size,
           mipmapped), its region means over the moving pixels (a texel per region), and those means' own fast and
           slow trails; with a stop region weight, the region mean speeds of the two trails (RG16F, half size,
           mipmapped)
        4. Split the rise, max(0, fast - slow along the motion), by its region into the start and the bow, and the
           fall, max(0, -(fast - slow)), into the stop and the wake
        5. Each along its direction → start, bow, stop, wake; the whole difference → change
        6. With `acceleration`, each over the band's width → <reading>_acceleration

    FBOs (from FlowBase):
        - input_fbo: The velocity
        - output_fbo: unused (the lowpass's form); the readings are their own FBOs
    """

    def __init__(self, settings: VelocityBandpassSettings | None = None) -> None:
        super().__init__()

        self._input_internal_format = GL_RG16F
        self._output_internal_format = GL_RG16F
        self._input_kind = TextureKind.VELOCITY
        self._output_kind = TextureKind.VELOCITY

        self.settings: VelocityBandpassSettings = settings or VelocityBandpassSettings()

        self._fast_fbo: SwapFbo = SwapFbo()
        self._slow_fbo: SwapFbo = SwapFbo()
        self._moving_fbo: Fbo = Fbo()                   # the start's: velocity * moving, moving
        self._moving_mean_fbo: Fbo = Fbo()              # a region's mean velocity over its moving pixels, a texel per region
        self._region_fast_fbo: SwapFbo = SwapFbo()      # that mean, through the fast trail
        self._region_slow_fbo: SwapFbo = SwapFbo()      # and through the slow trail
        self._region_speed_fbo: Fbo = Fbo()             # the stop's: fast speed, slow speed
        self._speeds: dict[Change, Fbo] = {change: Fbo() for change in Change}         # each part's size (R16F)
        self._velocities: dict[Change, Fbo] = {change: Fbo() for change in Change}     # each part as a velocity
        self._accelerations: dict[Change, Fbo] = {change: Fbo() for change in Change}  # and as an acceleration
        self._change_fbo: Fbo = Fbo()                   # the whole change, as a velocity
        self._change_acceleration_fbo: Fbo = Fbo()      # and as an acceleration
        self._size: tuple[int, int] = (0, 0)
        self._warned_peak: bool = False     # slow_smooth not slower than fast_smooth was logged

        self._trail_shader: shaders.WeightedSum = shaders.WeightedSum()
        self._trails_shader: BandpassTrails2D = BandpassTrails2D()
        self._changes_shader: BandpassChanges2D = BandpassChanges2D()
        self._accelerations_shader: BandpassAccelerations2D = BandpassAccelerations2D()
        self._moving_shader: VelocityBandpassRegion2D = VelocityBandpassRegion2D()
        self._moving_mean_shader: VelocityBandpassRegionMean2D = VelocityBandpassRegionMean2D()
        self._region_speed_shader: VelocityBandpassRegionSpeed2D = VelocityBandpassRegionSpeed2D()

    @property
    def velocity_input(self) -> Texture:
        """The flow's velocity."""
        return self._input

    @property
    def fast(self) -> Texture:
        """Fast trail (RG16F, xy)."""
        return self._fast_fbo

    @property
    def slow(self) -> Texture:
        """Slow trail (RG16F, xy)."""
        return self._slow_fbo

    @property
    def change(self) -> Texture:
        """The whole change, (fast − slow) / step_peak, signed (RG16F, xy), image widths per second."""
        return self._change_fbo

    @property
    def change_acceleration(self) -> Texture:
        """The whole change as an acceleration, (fast − slow) over the band's width, signed (RG16F, xy), widths/s², as
        of the last update that asked for it."""
        return self._change_acceleration_fbo

    def speed(self, change: Change) -> Texture:
        """A part's size (R16F), image widths per second."""
        return self._speeds[change]

    def velocity(self, change: Change) -> Texture:
        """A part along its direction (RG16F, xy), widths/s: a start or a bow along the fast trail's, a stop or a
        wake along the slow trail's, the motion it ended."""
        return self._velocities[change]

    def acceleration(self, change: Change) -> Texture:
        """A part as an acceleration (RG16F, xy), widths/s², as of the last update that asked for it."""
        return self._accelerations[change]

    @property
    def ports(self) -> dict[str, Port]:
        ports: dict[str, Port] = {'input': Port(self._input_fbo, TextureKind.VELOCITY),
                                  'fast': Port(self._fast_fbo, TextureKind.VELOCITY),
                                  'slow': Port(self._slow_fbo, TextureKind.VELOCITY),
                                  'change': Port(self._change_fbo, TextureKind.VELOCITY),
                                  'change_acceleration': Port(self._change_acceleration_fbo, TextureKind.ACCELERATION)}
        for change in Change:
            name: str = change.name.lower()
            ports[name] = Port(self._velocities[change], TextureKind.VELOCITY)
            ports[f'{name}_acceleration'] = Port(self._accelerations[change], TextureKind.ACCELERATION)
            ports[f'{name}_speed'] = Port(self._speeds[change], TextureKind.SCALAR)
        return ports

    def _readings(self) -> tuple[Fbo, ...]:
        """Every reading's FBO, both units, and the parts' sizes."""
        return (*self._speeds.values(), *self._velocities.values(), *self._accelerations.values(),
                self._change_fbo, self._change_acceleration_fbo)

    def allocate(self, width: int, height: int, output_width: int | None = None, output_height: int | None = None) -> None:
        """Allocate all fields at the bridge resolution."""
        super().allocate(width, height, output_width, output_height)
        self._size = (width, height)
        self._fast_fbo.allocate(width, height, GL_RG16F)
        self._slow_fbo.allocate(width, height, GL_RG16F)
        # the moving velocity and its share (RGB used; RGBA16F, as RGB16F is not sure to be renderable), the speeds
        for regions, internal_format in ((self._moving_fbo, GL_RGBA16F), (self._region_speed_fbo, GL_RG16F)):
            regions.allocate(width // 2, height // 2, internal_format)
            glBindTexture(GL_TEXTURE_2D, regions.tex_id)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glGenerateMipmap(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, 0)
        for change in Change:
            self._speeds[change].allocate(width, height, GL_R16F)
            self._velocities[change].allocate(width, height, GL_RG16F)
            self._accelerations[change].allocate(width, height, GL_RG16F)
        self._change_fbo.allocate(width, height, GL_RG16F)
        self._change_acceleration_fbo.allocate(width, height, GL_RG16F)
        self._follow_start_region_size()
        self.reset()

        for shader in self._shaders():
            shader.allocate()

    def _shaders(self) -> tuple:
        return (self._trail_shader, self._trails_shader, self._changes_shader, self._accelerations_shader,
                self._moving_shader, self._moving_mean_shader, self._region_speed_shader)

    def deallocate(self) -> None:
        """Release all resources."""
        super().deallocate()
        for fbo in (self._fast_fbo, self._slow_fbo, self._moving_fbo, self._moving_mean_fbo, self._region_fast_fbo,
                    self._region_slow_fbo, self._region_speed_fbo, *self._readings()):
            fbo.deallocate()
        self._size = (0, 0)
        for shader in self._shaders():
            shader.deallocate()

    def reset(self) -> None:
        """Reset all fields to zero."""
        super().reset()
        for fbo in (self._fast_fbo, self._slow_fbo, self._region_fast_fbo, self._region_slow_fbo, *self._readings()):
            FlowUtil.zero(fbo)

    def _start_region_grid(self) -> tuple[int, int]:
        """A texel per region of `start_region_size` pixels."""
        size: float = max(self.settings.start_region_size, 1.0)
        return (max(1, round(self._size[0] / size)), max(1, round(self._size[1] / size)))

    def _follow_start_region_size(self) -> None:
        """Allocate the start's per-region fields a texel per region, again when `start_region_size` changes, from rest."""
        grid: tuple[int, int] = self._start_region_grid()
        if (self._moving_mean_fbo.width, self._moving_mean_fbo.height) == grid and self._moving_mean_fbo.allocated:
            return
        for fbo in (self._moving_mean_fbo, self._region_fast_fbo, self._region_slow_fbo):
            fbo.deallocate()
            fbo.allocate(*grid, GL_RG16F)
            FlowUtil.zero(fbo)

    def set_velocity(self, velocity: Texture) -> None:
        """Take a flow's 2D velocity, x, y (RG) in image widths per second, as the next input."""
        if not self._allocated:
            return
        FlowUtil.blit(self._input_fbo, velocity)

    def update(self, delta_time: float = 1.0 / 60.0, acceleration: bool = False) -> None:
        """Advance the trails by `delta_time` and read the changes; with `acceleration` also write each reading as an
        acceleration."""
        if not self._allocated:
            return
        s: VelocityBandpassSettings = self.settings
        fast_weight: float = decay_weight(delta_time, s.fast_smooth)
        slow_weight: float = decay_weight(delta_time, s.slow_smooth)

        for fbo in (self._fast_fbo, self._slow_fbo):
            fbo.swap()
        self._trails_shader.use(self._input, self._fast_fbo.back, self._slow_fbo.back, self._fast_fbo, self._slow_fbo,
                                fast_weight, slow_weight)

        # The mip level of a region: the region textures are half size
        start_lod: float = math.log2(max(s.start_region_size, 2.0)) - 1.0
        stop_lod: float = math.log2(max(s.stop_region_size, 2.0)) - 1.0
        if s.start_region_weight > 0.0:
            self._follow_start_region_size()
            self._moving_fbo.begin()
            self._moving_shader.use(self._input)
            self._moving_fbo.end()
            self._mipmap(self._moving_fbo)
            self._moving_mean_fbo.begin()
            self._moving_mean_shader.use(self._moving_fbo, start_lod)
            self._moving_mean_fbo.end()
            self._advance_trail(self._region_fast_fbo, self._moving_mean_fbo, fast_weight)
            self._advance_trail(self._region_slow_fbo, self._moving_mean_fbo, slow_weight)
        if s.stop_region_weight > 0.0:
            self._region_speed_fbo.begin()
            self._region_speed_shader.use(self.fast, self.slow)
            self._region_speed_fbo.end()
            self._mipmap(self._region_speed_fbo)

        peak: float = step_peak(s.fast_smooth, s.slow_smooth)
        if peak <= 0.0 and not self._warned_peak:
            self._warned_peak = True
            logger.warning("VelocityBandpass2D: slow_smooth (%.0f ms) is not slower than fast_smooth (%.0f ms), so there are no changes",
                           s.slow_smooth, s.fast_smooth)
        self._warned_peak = self._warned_peak and peak <= 0.0
        scale: float = 1.0 / peak if peak > 0.0 else 0.0
        # As an acceleration the difference is over the band's width instead (`band_width`): × dt over the pulse is its Δv
        acceleration_scale: float = change_acceleration_scale(s.fast_smooth, s.slow_smooth, delta_time)

        # The four parts, each its size and its velocity, in one dispatch; then the whole change and, with
        # `acceleration`, every reading over the band's width instead of the step peak, in one more
        sizes, parts, rates = self._speeds, self._velocities, self._accelerations
        speeds: tuple[Texture, Texture, Texture, Texture] = (sizes[Change.START], sizes[Change.BOW], sizes[Change.STOP], sizes[Change.WAKE])
        velocities: tuple[Texture, Texture, Texture, Texture] = (parts[Change.START], parts[Change.BOW], parts[Change.STOP], parts[Change.WAKE])
        self._changes_shader.use(self.fast, self.slow, self._region_fast_fbo, self._region_slow_fbo, self._region_speed_fbo,
                                 speeds, velocities, scale, s.start_region_weight, s.stop_region_weight, stop_lod)
        accelerations: tuple[Texture, Texture, Texture, Texture, Texture] | None = (
            self._change_acceleration_fbo, rates[Change.START], rates[Change.BOW], rates[Change.STOP], rates[Change.WAKE]) if acceleration else None
        self._accelerations_shader.use(self.fast, self.slow, velocities, self._change_fbo, accelerations,
                                       scale, acceleration_scale, peak)

    @staticmethod
    def _mipmap(regions: Fbo) -> None:
        """Make a region texture's mips, its region means."""
        glBindTexture(GL_TEXTURE_2D, regions.tex_id)
        glGenerateMipmap(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, 0)

    def _advance_trail(self, trail: SwapFbo, input: Texture, weight: float) -> None:
        """One unit-gain Ema step: weight * previous + (1 - weight) * input."""
        trail.swap()
        trail.begin()
        self._trail_shader.use(trail.back, weight, input, 1.0 - weight)
        trail.end()
