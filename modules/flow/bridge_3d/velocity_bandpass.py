"""Velocity Bandpass.

Next to the lowpass, which reads the motion itself, this reads the motion's changes: its velocity changes,
Δv, where a move starts and where it stops. Two short trails on the flow's velocity, fast and slow, unit-gain
Emas; their difference is the band between their times. Along the motion's direction it rises where a move
starts and falls where it stops. The difference is divided by its step peak (`step_peak`): a limb at speed v
that starts or stops at once reads v, so the changes are in the motion's own unit, image widths per second; a
gradual start or stop reads less, by how much of the speed changes within the trails' times.

Each reading also comes as an acceleration, widths/s²: the same difference over the band's width (`band_width`,
`slow_smooth` − `fast_smooth` for a short frame) instead of the step peak. For any change of speed the
difference's integral over time is the band's width × Δv, so the acceleration × dt adds up to the Δv once,
whatever the frame rate and the trails' times; the velocity's height reads the Δv only for a sudden change.

A pixel cannot tell a move starting from a limb arriving at it, nor a move stopping from a limb leaving
it: its speed rises or falls either way. The region around it (`start_region_size`, `stop_region_size`)
can, so the region splits each pixel's rise and fall in two, by its region weight (`start_region_weight`,
`stop_region_weight`; 0 = all of it is the region's):
    start   the share of a rise its region shares: a part of the body speeding up
    bow     the rest: a limb arriving, the wave in front of a moving limb
    stop    the share of a fall its region shares: a part of the body slowing down
    wake    the rest: a limb leaving, the wake behind it
Start + bow is the pixel's whole rise, stop + wake its whole fall. A change lasts about `slow_smooth`, as
the slow trail catches up; it takes no hold, so what a stop injects is the speed it ended.

What a region shares:
    rise    the region's mean velocity over its moving pixels (those the motion gives a velocity), smoothed per region
            by the fast and the slow trail, rises: a limb arriving adds moving pixels at its own speed and
            leaves that mean as it was, a limb speeding up raises it, and jitter on a still body moves no
            pixel. Taken over all of a region's pixels, an arriving limb would raise the mean like a speedup;
            taken of the two trails, the slow trail's lag at the pixels it arrives at would too
    fall    the region's mean speed of the fast trail falls below the slow one's: jitter never lowers it, and a
            turning limb keeps it, where a mean velocity would shrink and read as a stop. The trails lag at the
            pixels a limb leaves, so a leaving limb lowers it too, and part of a wake reads as a stop

The changes read the flow's x, y and z alike. They filter in time only: noise within a frame is the
motion's to filter. No blur and no advect: the changes stay sharp.

Out come the readings, each as a velocity in image widths per second and, when asked, an acceleration in
widths/s² (RGBA16F, xyz):
    change                          (fast − slow) itself, signed: the whole change, before the direction and the regions
    start, bow, stop, wake          the motion's direction times its part of the change: a start and a bow along
                                    the fast trail's direction, a stop and a wake along the slow one's, the motion
                                    it ended
and each part's size, `<part>_speed` (R16F, widths/s).

Steps every tick on its dt while it runs.
"""

import logging
import math

import numpy as np
from OpenGL.GL import GL_LINEAR, GL_LINEAR_MIPMAP_LINEAR, GL_R16F, GL_RG16F, GL_RGBA16F, GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MIN_FILTER, glBindTexture, glGenerateMipmap, glTexParameteri

from modules.gl import Fbo, SwapFbo, Texture
from modules.settings import Field, BaseSettings
from .. import FlowBase, FlowUtil, Port, TextureKind
from ..shaders import Trail
from .trails import EPSILON, decay_weight, direction
from .shaders import Change, VelocityBandpassChange, VelocityBandpassDirection, VelocityBandpassRegion, VelocityBandpassRegionMean, VelocityBandpassRegionSpeed, WeightedSum

logger = logging.getLogger(__name__)


def step_peak(fast_smooth: float, slow_smooth: float) -> float:
    """The largest difference of a slow and a fast Ema (time constants in one unit) after a unit step: where
    the slow one still holds what the fast one let go. 1 with an instant fast trail, 0 when the slow
    trail is not slower. Continuous, as `decay_weight` is."""
    if slow_smooth <= fast_smooth:
        return 0.0
    if fast_smooth <= 0.0:
        return 1.0
    t: float = fast_smooth * slow_smooth / (slow_smooth - fast_smooth) * math.log(slow_smooth / fast_smooth)
    return math.exp(-t / slow_smooth) - math.exp(-t / fast_smooth)


def band_width(fast_smooth: float, slow_smooth: float, delta_time: float) -> float:
    """The band's width, s: what (fast − slow) adds up to over time, × dt, per unit of a change of speed. For the
    two Emas stepped at `delta_time` it is dt × (w_slow / (1 − w_slow) − w_fast / (1 − w_fast)), w each one's
    `decay_weight`; as dt shrinks it is (slow_smooth − fast_smooth), and it stays exact where a frame is longer
    than the fast trail. 0 when the slow trail is not slower."""
    if slow_smooth <= fast_smooth:
        return 0.0
    if delta_time <= 0.0:
        return (slow_smooth - fast_smooth) / 1000.0

    def memory(smooth: float) -> float:
        weight: float = decay_weight(delta_time, smooth) if smooth > 0.0 else 0.0
        return delta_time * weight / (1.0 - weight)
    return memory(slow_smooth) - memory(fast_smooth)


def change_acceleration_scale(fast_smooth: float, slow_smooth: float, delta_time: float) -> float:
    """What (fast − slow) is multiplied by to read as an acceleration, widths/s²: 1 over the band's width
    (`band_width`, s; the times in ms), so its sum × dt over a change of speed is that change; 0 when the slow
    trail is not slower."""
    width: float = band_width(fast_smooth, slow_smooth, delta_time)
    return 1.0 / width if width > 0.0 else 0.0


class VelocityBandpassSettings(BaseSettings):
    """Configuration for the velocity bandpass: its two trails and the start and stop regions."""
    fast_smooth = Field(25.0, min=0.0, max=500.0, step=1.0, description="Fast trail smoothing time, ms; 0 = the flow itself")
    slow_smooth = Field(150.0, min=0.0, max=1000.0, step=1.0, description="Slow trail smoothing time, ms: what a change is read against")
    start_region_weight = Field(1.0, min=0.0, max=1.0, step=0.05, newline=True, description="How much the region around a start must speed up too, the rest the bow; 0 = all start")
    stop_region_weight = Field(1.0, min=0.0, max=1.0, step=0.05, description="How much the region around a stop must slow down too, the rest the wake; 0 = all stop")
    start_region_size = Field(128.0, min=8.0, max=1024.0, step=8.0, newline=True, description="Size of the region around a start, pixels of the bridge")
    stop_region_size = Field(128.0, min=8.0, max=1024.0, step=8.0, description="Size of the region around a stop, pixels of the bridge")


class VelocityBandpass(FlowBase):
    """Two short trails on the flow and the starts and stops of their difference.

    Pipeline:
        1. Take the flow's velocity → input_fbo (RGBA16F, xyz)
        2. Advance the fast and slow trails, unit-gain Emas
        3. With a start region weight, the flow's velocity where a pixel moves and whether it moves (RGBA16F,
           half size, mipmapped), its region means over the moving pixels (a texel per region), and those
           means' own fast and slow trails; with a stop region weight, the region mean speeds of the two
           trails (RG16F, half size, mipmapped)
        4. Split the rise, max(0, fast - slow along the motion), by its region into the start and the bow, and
           the fall, max(0, -(fast - slow)), into the stop and the wake
        5. Each along its direction → start, bow, stop, wake; the whole difference → change
        6. With `acceleration`, each over (slow_smooth − fast_smooth) → <reading>_acceleration

    FBOs (from FlowBase):
        - input_fbo: The velocity
        - output_fbo: unused (the lowpass's form); the readings are their own FBOs
    """

    def __init__(self, settings: VelocityBandpassSettings | None = None) -> None:
        super().__init__()

        self._input_internal_format = GL_RGBA16F
        self._output_internal_format = GL_RGBA16F
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

        self._trail_shader: Trail = Trail()
        self._moving_shader: VelocityBandpassRegion = VelocityBandpassRegion()
        self._moving_mean_shader: VelocityBandpassRegionMean = VelocityBandpassRegionMean()
        self._region_speed_shader: VelocityBandpassRegionSpeed = VelocityBandpassRegionSpeed()
        self._change_shader: VelocityBandpassChange = VelocityBandpassChange()
        self._direction_shader: VelocityBandpassDirection = VelocityBandpassDirection()
        self._sum_shader: WeightedSum = WeightedSum()

    @property
    def velocity_input(self) -> Texture:
        """The flow's velocity."""
        return self._input

    @property
    def fast(self) -> Texture:
        """Fast trail (RGBA16F, xyz)."""
        return self._fast_fbo

    @property
    def slow(self) -> Texture:
        """Slow trail (RGBA16F, xyz)."""
        return self._slow_fbo

    @property
    def change(self) -> Texture:
        """The whole change, (fast − slow) / step_peak, signed (RGBA16F, xyz), image widths per second."""
        return self._change_fbo

    @property
    def change_acceleration(self) -> Texture:
        """The whole change as an acceleration, (fast − slow) / (slow_smooth − fast_smooth), signed (RGBA16F, xyz),
        widths/s², as of the last update that asked for it."""
        return self._change_acceleration_fbo

    def speed(self, change: Change) -> Texture:
        """A part's size (R16F), image widths per second."""
        return self._speeds[change]

    def velocity(self, change: Change) -> Texture:
        """A part along its direction (RGBA16F, xyz), widths/s: a start or a bow along the fast trail's, a stop or a
        wake along the slow trail's, the motion it ended."""
        return self._velocities[change]

    def acceleration(self, change: Change) -> Texture:
        """A part as an acceleration (RGBA16F, xyz), widths/s², as of the last update that asked for it."""
        return self._accelerations[change]

    @property
    def ports(self) -> dict[str, Port]:
        ports: dict[str, Port] = {'input': Port(self._input_fbo, TextureKind.VELOCITY),
                                  'fast': Port(self._fast_fbo, TextureKind.VELOCITY),
                                  'slow': Port(self._slow_fbo, TextureKind.VELOCITY),
                                  'change': Port(self._change_fbo, TextureKind.VELOCITY),
                                  'change_acceleration': Port(self._change_acceleration_fbo, TextureKind.VELOCITY)}
        for change in Change:
            name: str = change.name.lower()
            ports[name] = Port(self._velocities[change], TextureKind.VELOCITY)
            ports[f'{name}_acceleration'] = Port(self._accelerations[change], TextureKind.VELOCITY)
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
        self._fast_fbo.allocate(width, height, GL_RGBA16F)
        self._slow_fbo.allocate(width, height, GL_RGBA16F)
        for regions, internal_format in ((self._moving_fbo, GL_RGBA16F), (self._region_speed_fbo, GL_RG16F)):
            regions.allocate(width // 2, height // 2, internal_format)
            glBindTexture(GL_TEXTURE_2D, regions.tex_id)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glGenerateMipmap(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, 0)
        for change in Change:
            self._speeds[change].allocate(width, height, GL_R16F)
            self._velocities[change].allocate(width, height, GL_RGBA16F)
            self._accelerations[change].allocate(width, height, GL_RGBA16F)
        self._change_fbo.allocate(width, height, GL_RGBA16F)
        self._change_acceleration_fbo.allocate(width, height, GL_RGBA16F)
        self._follow_start_region_size()
        self.reset()

        self._trail_shader.allocate()
        self._moving_shader.allocate()
        self._moving_mean_shader.allocate()
        self._region_speed_shader.allocate()
        self._change_shader.allocate()
        self._direction_shader.allocate()
        self._sum_shader.allocate()

    def deallocate(self) -> None:
        """Release all resources."""
        super().deallocate()
        for fbo in (self._fast_fbo, self._slow_fbo, self._moving_fbo, self._moving_mean_fbo, self._region_fast_fbo,
                    self._region_slow_fbo, self._region_speed_fbo, *self._readings()):
            fbo.deallocate()
        self._size = (0, 0)
        self._trail_shader.deallocate()
        self._moving_shader.deallocate()
        self._moving_mean_shader.deallocate()
        self._region_speed_shader.deallocate()
        self._change_shader.deallocate()
        self._direction_shader.deallocate()
        self._sum_shader.deallocate()

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
            fbo.allocate(*grid, GL_RGBA16F)
            FlowUtil.zero(fbo)

    def set_velocity(self, velocity: Texture) -> None:
        """Take a flow's 3D velocity, x, y, z (RGB) in image widths per second, as the next input."""
        if not self._allocated:
            return
        FlowUtil.blit(self._input_fbo, velocity)

    def update(self, delta_time: float = 1.0 / 60.0, acceleration: bool = False) -> None:
        """Advance the trails by `delta_time` and read the changes.

        Args:
            delta_time: Seconds since the previous update
            acceleration: Also write each reading as an acceleration
        """
        if not self._allocated:
            return
        s: VelocityBandpassSettings = self.settings
        fast_weight: float = decay_weight(delta_time, s.fast_smooth)
        slow_weight: float = decay_weight(delta_time, s.slow_smooth)

        self._advance_trail(self._fast_fbo, self._input, fast_weight)
        self._advance_trail(self._slow_fbo, self._input, slow_weight)

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
            logger.warning("VelocityBandpass: slow_smooth (%.0f ms) is not slower than fast_smooth (%.0f ms), so there are no changes",
                           s.slow_smooth, s.fast_smooth)
        self._warned_peak = self._warned_peak and peak <= 0.0
        scale: float = 1.0 / peak if peak > 0.0 else 0.0
        # As an acceleration the difference is over the band's width instead (`band_width`): × dt over the pulse is its Δv
        acceleration_scale: float = change_acceleration_scale(s.fast_smooth, s.slow_smooth, delta_time)

        self._sum(self._change_fbo, [(self.fast, scale), (self.slow, -scale)])
        if acceleration:
            self._sum(self._change_acceleration_fbo, [(self.fast, acceleration_scale), (self.slow, -acceleration_scale)])

        for change in Change:
            rise: bool = change.sign > 0.0
            region_weight: float = s.start_region_weight if rise else s.stop_region_weight
            region_fast: Texture = self._region_fast_fbo if rise else self._region_speed_fbo
            region_slow: Texture = self._region_slow_fbo if rise else self._region_speed_fbo
            speed: Fbo = self._speeds[change]
            speed.begin()
            self._change_shader.use(self.fast, self.slow, change, scale, region_fast, region_slow, region_weight, stop_lod)
            speed.end()

            velocity: Fbo = self._velocities[change]
            velocity.begin()
            self._direction_shader.use(self.fast if rise else self.slow, speed)
            velocity.end()
            if acceleration:
                # the same pulse over the band's width instead of the step peak
                self._sum(self._accelerations[change], [(velocity, acceleration_scale * peak)])

    def _sum(self, target: Fbo, sources: list[tuple[Texture, float]]) -> None:
        """The sources at their weights, summed into `target`."""
        target.begin()
        self._sum_shader.use(sources)
        target.end()

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
        self._trail_shader.use(trail.back, input, weight, 1.0 - weight)
        trail.end()


# ── numpy reference ─────────────────────────────────────────────────────

def velocity_change(fast: np.ndarray, slow: np.ndarray, sign: float, share: np.ndarray | float = 1.0,
                     edge: bool = False, scale: float = 1.0) -> np.ndarray:
    """One change (..., ) from the trails (..., 3): max(sign * (fast - slow) along the motion, 0) times `scale`
    (1 / `step_peak`), times the share its region shares (a start or a stop), or with `edge` times the rest (a bow
    or a wake). Mirrors velocity_bandpass_change.frag."""
    unit: np.ndarray = direction(fast, slow)
    part: np.ndarray | float = 1.0 - share if edge else share
    return np.maximum(0.0, sign * np.sum((fast - slow) * unit, axis=-1)) * scale * part


def moving_mean(velocity: np.ndarray) -> np.ndarray:
    """A region's mean velocity (3,) over its moving pixels, those of its pixels (N, 3) the motion gives a velocity;
    0 where none moves. Mirrors velocity_bandpass_region.frag and velocity_bandpass_region_mean.frag."""
    moving: np.ndarray = np.linalg.norm(velocity, axis=-1) > EPSILON
    return np.mean(velocity[moving], axis=0) if moving.any() else np.zeros(3)


def start_region_share(region_fast: np.ndarray, region_slow: np.ndarray, region_weight: float) -> float:
    """How much of a start counts, 0..1, from its region's mean moving velocity through the fast and the slow trail
    (`moving_mean` of each frame, smoothed): as far as it grows; region weight 0 lets all count. Mirrors
    velocity_bandpass_change.frag."""
    fast: float = float(np.linalg.norm(region_fast))
    change: float = (fast - float(np.linalg.norm(region_slow))) / max(fast, EPSILON)
    return 1.0 + (min(max(change, 0.0), 1.0) - 1.0) * region_weight


def stop_region_share(fast: np.ndarray, slow: np.ndarray, region_weight: float) -> float:
    """How much of a stop counts, 0..1, from the trails over its region's pixels (N, 3): as far as their mean speed
    falls; region weight 0 lets all count. Mirrors velocity_bandpass_change.frag."""
    fast_speed: float = float(np.mean(np.linalg.norm(fast, axis=-1)))
    slow_speed: float = float(np.mean(np.linalg.norm(slow, axis=-1)))
    change: float = (slow_speed - fast_speed) / max(slow_speed, EPSILON)
    return 1.0 + (min(max(change, 0.0), 1.0) - 1.0) * region_weight


def change_velocity(trail: np.ndarray, change: np.ndarray) -> np.ndarray:
    """The change (...) along the trail's direction (..., 3). Mirrors velocity_bandpass_direction.frag."""
    length: np.ndarray = np.linalg.norm(trail, axis=-1, keepdims=True)
    unit: np.ndarray = np.divide(trail, length, out=np.zeros_like(trail), where=length > EPSILON)
    return unit * np.maximum(0.0, change)[..., None]


def whole_change(fast: np.ndarray, slow: np.ndarray, fast_smooth: float, slow_smooth: float,
                 delta_time: float, acceleration: bool = False) -> np.ndarray:
    """The whole change (..., 3), signed: (fast − slow) / step_peak as a velocity, widths/s, or with `acceleration`
    (fast − slow) over the band's width, widths/s². Mirrors the bandpass's change passes."""
    if acceleration:
        return (fast - slow) * change_acceleration_scale(fast_smooth, slow_smooth, delta_time)
    peak: float = step_peak(fast_smooth, slow_smooth)
    return (fast - slow) / peak if peak > 0.0 else np.zeros_like(fast)
