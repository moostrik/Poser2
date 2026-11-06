"""Velocity Event Bridge.

Next to the velocity trail, which is the motion's push, this reads the motion's events: where a move
starts and where it stops. Two short trails on the flow's velocity, fast and slow, unit-gain Emas; their
difference along the motion's direction rises where a move starts and falls where it stops. The
difference is divided by its step peak (`step_peak`): a limb at speed v that starts or stops at once
reads v, so the events are in the motion's own strength, image widths per second; a gradual start or
stop reads less, by how much of the speed changes within the trails' times.

A pixel cannot tell a move starting from a limb arriving at it, nor a move stopping from a limb leaving
it: its speed rises or falls either way. The region around it (`start_region_size`, `stop_region_size`)
can, so the region splits each pixel's rise and fall in two, by its region weight (`start_region_weight`,
`stop_region_weight`; 0 = all of it is the region's):
    start   the share of a rise its region shares: a part of the body speeding up
    bow     the rest: a limb arriving, the wave in front of a moving limb
    stop    the share of a fall its region shares: a part of the body slowing down
    wake    the rest: a limb leaving, the wake behind it
Start + bow is the pixel's whole rise, stop + wake its whole fall. An event lasts about `slow_smooth`, as
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

The events read the flow's x, y and z alike. They filter in time only: noise within a frame is the
motion's to filter. No blur and no advect: the events stay sharp.

Out come, in image widths per second like the flow:
    start, bow, stop, wake          the events (R16F)
    <event>_velocity                the motion's direction times its event (RGBA16F, xyz): a start and a bow along
                                    the fast trail's direction, a stop and a wake along the slow one's, the motion
                                    it ended

Steps every tick on its dt, as the other bridges do.
"""

import logging
import math

import numpy as np
from OpenGL.GL import GL_LINEAR, GL_LINEAR_MIPMAP_LINEAR, GL_R16F, GL_RG16F, GL_RGBA16F, GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MIN_FILTER, glBindTexture, glGenerateMipmap, glTexParameteri

from modules.gl import Fbo, SwapFbo, Texture
from modules.settings import Field, BaseSettings
from .. import FlowBase, FlowUtil, Port, TextureKind
from ..shaders import Trail
from .kinetics import EPSILON, decay_weight, direction
from .shaders import MotionEvent, KineticOnset, KineticPatch, VelocityEvent, VelocityEventRegion, VelocityEventRegionMean

logger = logging.getLogger(__name__)


def step_peak(fast_smooth: float, slow_smooth: float) -> float:
    """The largest difference of a slow and a fast Ema (time constants in s) after a unit step: where
    the slow one still holds what the fast one let go. 1 with an instant fast trail, 0 when the slow
    trail is not slower. Continuous, as `decay_weight` is."""
    if slow_smooth <= fast_smooth:
        return 0.0
    if fast_smooth <= 0.0:
        return 1.0
    t: float = fast_smooth * slow_smooth / (slow_smooth - fast_smooth) * math.log(slow_smooth / fast_smooth)
    return math.exp(-t / slow_smooth) - math.exp(-t / fast_smooth)


class VelocityEventBridgeSettings(BaseSettings):
    """Configuration for the velocity event bridge: its two trails and the start and stop regions."""
    fast_smooth = Field(0.025, min=0.0, max=0.5, step=0.005, description="Fast trail smoothing time, s; 0 = the flow itself")
    slow_smooth = Field(0.15, min=0.0, max=1.0, step=0.005, description="Slow trail smoothing time, s: what a change is read against")
    start_region_weight = Field(1.0, min=0.0, max=1.0, step=0.05, newline=True, description="How much the region around a start must speed up too, the rest the bow; 0 = all start")
    stop_region_weight = Field(1.0, min=0.0, max=1.0, step=0.05, description="How much the region around a stop must slow down too, the rest the wake; 0 = all stop")
    start_region_size = Field(128.0, min=8.0, max=1024.0, step=8.0, newline=True, description="Size of the region around a start, pixels of the bridge")
    stop_region_size = Field(128.0, min=8.0, max=1024.0, step=8.0, description="Size of the region around a stop, pixels of the bridge")


class VelocityEventBridge(FlowBase):
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
        5. Each along its direction → start_velocity, bow_velocity, stop_velocity, wake_velocity

    FBOs (from FlowBase):
        - input_fbo: The velocity
        - output_fbo: unused (the trail bridges' form); the outputs are their own FBOs
    """

    def __init__(self, settings: VelocityEventBridgeSettings | None = None) -> None:
        super().__init__()

        self._input_internal_format = GL_RGBA16F
        self._output_internal_format = GL_RGBA16F
        self._input_kind = TextureKind.VELOCITY
        self._output_kind = TextureKind.VELOCITY

        self.settings: VelocityEventBridgeSettings = settings or VelocityEventBridgeSettings()

        self._fast_fbo: SwapFbo = SwapFbo()
        self._slow_fbo: SwapFbo = SwapFbo()
        self._moving_fbo: Fbo = Fbo()                   # the start's: velocity * moving, moving
        self._moving_mean_fbo: Fbo = Fbo()              # a region's mean velocity over its moving pixels, a texel per region
        self._region_fast_fbo: SwapFbo = SwapFbo()      # that mean, through the fast trail
        self._region_slow_fbo: SwapFbo = SwapFbo()      # and through the slow trail
        self._region_speed_fbo: Fbo = Fbo()             # the stop's: fast speed, slow speed
        self._events: dict[MotionEvent, Fbo] = {event: Fbo() for event in MotionEvent}
        self._velocities: dict[MotionEvent, Fbo] = {event: Fbo() for event in MotionEvent}
        self._size: tuple[int, int] = (0, 0)
        self._warned_peak: bool = False     # slow_smooth not slower than fast_smooth was logged

        self._trail_shader: Trail = Trail()
        self._moving_shader: VelocityEventRegion = VelocityEventRegion()
        self._moving_mean_shader: VelocityEventRegionMean = VelocityEventRegionMean()
        self._region_speed_shader: KineticPatch = KineticPatch()   # the trails' speeds, whose mips are a region's means
        self._event_shader: VelocityEvent = VelocityEvent()
        self._direction_shader: KineticOnset = KineticOnset()     # unit of a trail times an amount

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

    def event(self, event: MotionEvent) -> Texture:
        """The event (R16F), image widths per second."""
        return self._events[event]

    def event_velocity(self, event: MotionEvent) -> Texture:
        """The event along its direction (RGBA16F, xyz): a start or a bow along the fast trail's, a stop or a
        wake along the slow trail's, the motion it ended."""
        return self._velocities[event]

    @property
    def ports(self) -> dict[str, Port]:
        ports: dict[str, Port] = {**super().ports,
                                  'fast': Port(self._fast_fbo, TextureKind.VELOCITY),
                                  'slow': Port(self._slow_fbo, TextureKind.VELOCITY)}
        for event in MotionEvent:
            name: str = event.name.lower()
            ports[name] = Port(self._events[event], TextureKind.SCALAR)
            ports[f'{name}_velocity'] = Port(self._velocities[event], TextureKind.VELOCITY)
        return ports

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
        for event in MotionEvent:
            self._events[event].allocate(width, height, GL_R16F)
            self._velocities[event].allocate(width, height, GL_RGBA16F)
        self._follow_start_region_size()
        self.reset()

        self._trail_shader.allocate()
        self._moving_shader.allocate()
        self._moving_mean_shader.allocate()
        self._region_speed_shader.allocate()
        self._event_shader.allocate()
        self._direction_shader.allocate()

    def deallocate(self) -> None:
        """Release all resources."""
        super().deallocate()
        for fbo in (self._fast_fbo, self._slow_fbo, self._moving_fbo, self._moving_mean_fbo, self._region_fast_fbo,
                    self._region_slow_fbo, self._region_speed_fbo, *self._events.values(), *self._velocities.values()):
            fbo.deallocate()
        self._size = (0, 0)
        self._trail_shader.deallocate()
        self._moving_shader.deallocate()
        self._moving_mean_shader.deallocate()
        self._region_speed_shader.deallocate()
        self._event_shader.deallocate()
        self._direction_shader.deallocate()

    def reset(self) -> None:
        """Reset all fields to zero."""
        super().reset()
        for fbo in (self._fast_fbo, self._slow_fbo, self._region_fast_fbo, self._region_slow_fbo, *self._events.values(),
                    *self._velocities.values()):
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

    def update(self, delta_time: float = 1.0 / 60.0) -> None:
        """Advance the trails by `delta_time` and read the events.

        Args:
            delta_time: Seconds since the previous update
        """
        if not self._allocated:
            return
        s: VelocityEventBridgeSettings = self.settings
        fast_weight: float = decay_weight(delta_time, s.fast_smooth * 1000.0)
        slow_weight: float = decay_weight(delta_time, s.slow_smooth * 1000.0)

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
            logger.warning("VelocityEventBridge: slow_smooth (%.3f s) is not slower than fast_smooth (%.3f s), so there are no events",
                           s.slow_smooth, s.fast_smooth)
        self._warned_peak = self._warned_peak and peak <= 0.0
        scale: float = 1.0 / peak if peak > 0.0 else 0.0

        for event in MotionEvent:
            rise: bool = event.sign > 0.0
            region_weight: float = s.start_region_weight if rise else s.stop_region_weight
            region_fast: Texture = self._region_fast_fbo if rise else self._region_speed_fbo
            region_slow: Texture = self._region_slow_fbo if rise else self._region_speed_fbo
            amount: Fbo = self._events[event]
            amount.begin()
            self._event_shader.use(self.fast, self.slow, event, scale, region_fast, region_slow, region_weight, stop_lod)
            amount.end()

            velocity: Fbo = self._velocities[event]
            velocity.begin()
            self._direction_shader.use(self.fast if rise else self.slow, amount, 1.0)
            velocity.end()

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

def velocity_event(fast: np.ndarray, slow: np.ndarray, sign: float, share: np.ndarray | float = 1.0,
                   edge: bool = False, scale: float = 1.0) -> np.ndarray:
    """One event (..., ) from the trails (..., 3): max(sign * (fast - slow) along the motion, 0) times `scale`
    (1 / `step_peak`), times the share its region shares (a start or a stop), or with `edge` times the rest (a bow
    or a wake). Mirrors velocity_event.frag."""
    unit: np.ndarray = direction(fast, slow)
    part: np.ndarray | float = 1.0 - share if edge else share
    return np.maximum(0.0, sign * np.sum((fast - slow) * unit, axis=-1)) * scale * part


def moving_mean(velocity: np.ndarray) -> np.ndarray:
    """A region's mean velocity (3,) over its moving pixels, those of its pixels (N, 3) the motion gives a velocity;
    0 where none moves. Mirrors velocity_event_region.frag and velocity_event_region_mean.frag."""
    moving: np.ndarray = np.linalg.norm(velocity, axis=-1) > EPSILON
    return np.mean(velocity[moving], axis=0) if moving.any() else np.zeros(3)


def start_region_share(region_fast: np.ndarray, region_slow: np.ndarray, region_weight: float) -> float:
    """How much of a start counts, 0..1, from its region's mean moving velocity through the fast and the slow trail
    (`moving_mean` of each frame, smoothed): as far as it grows; region weight 0 lets all count. Mirrors
    velocity_event.frag."""
    fast: float = float(np.linalg.norm(region_fast))
    change: float = (fast - float(np.linalg.norm(region_slow))) / max(fast, EPSILON)
    return 1.0 + (min(max(change, 0.0), 1.0) - 1.0) * region_weight


def stop_region_share(fast: np.ndarray, slow: np.ndarray, region_weight: float) -> float:
    """How much of a stop counts, 0..1, from the trails over its region's pixels (N, 3): as far as their mean speed
    falls; region weight 0 lets all count. Mirrors velocity_event.frag."""
    fast_speed: float = float(np.mean(np.linalg.norm(fast, axis=-1)))
    slow_speed: float = float(np.mean(np.linalg.norm(slow, axis=-1)))
    change: float = (slow_speed - fast_speed) / max(slow_speed, EPSILON)
    return 1.0 + (min(max(change, 0.0), 1.0) - 1.0) * region_weight


def event_velocity(trail: np.ndarray, event: np.ndarray) -> np.ndarray:
    """The event (...) along the trail's direction (..., 3). Mirrors kinetic_onset.frag at strength 1."""
    length: np.ndarray = np.linalg.norm(trail, axis=-1, keepdims=True)
    unit: np.ndarray = np.divide(trail, length, out=np.zeros_like(trail), where=length > EPSILON)
    return unit * np.maximum(0.0, event)[..., None]
