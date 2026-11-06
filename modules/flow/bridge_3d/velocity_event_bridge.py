"""Velocity Event Bridge.

Next to the velocity trail, which is the motion's push, this reads the motion's events: where a move
starts and where it stops. Two short trails on the flow's velocity, fast and slow, unit-gain Emas; their
difference along the motion's direction rises at a start and falls at a stop. Each event is peak-held, a
start over `start_hold` and a stop over `stop_hold`.

A pixel cannot tell a move starting from a limb arriving at it, nor a move stopping from a limb leaving
it: its speed rises or falls either way. A patch (`start_patch_size`, `stop_patch_size`) can, so each event counts as far as its
patch changes the same way (`start_gate`, `stop_gate`):
    start   the patch's mean velocity over its moving pixels (faster than START_FLOOR), smoothed per patch
            by the fast and the slow trail, rises: a limb arriving adds moving pixels at its own speed and
            leaves that mean as it was, a limb speeding up raises it, and jitter on a still body moves no
            pixel. Taken over all of a patch's pixels, an arriving limb would raise the mean like a speedup;
            taken of the two trails, the slow trail's lag at the pixels it arrives at would too
    stop    the patch's mean speed of the fast trail falls below the slow one's: jitter never lowers it, and a
            turning limb keeps it, where a mean velocity would shrink and read as a stop

The events read the flow's x, y and z alike; the gates keep z's noise out as they keep x/y's. No blur and no
advect: the events stay sharp.

Out come, in image widths per second like the flow:
    start, stop                     the held events (R16F)
    start_velocity, stop_velocity   the motion's direction times its event (RGBA16F, xyz): a start along the
                                    fast trail's direction, a stop along the slow one's, the motion it ended

Steps every tick on its dt, as the other bridges do.
"""

import math

import numpy as np
from OpenGL.GL import GL_LINEAR, GL_LINEAR_MIPMAP_LINEAR, GL_R16F, GL_RG16F, GL_RGBA16F, GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MIN_FILTER, glBindTexture, glGenerateMipmap, glTexParameteri

from modules.gl import Fbo, SwapFbo, Texture
from modules.settings import Field, BaseSettings
from .. import FlowBase, FlowUtil, Port, TextureKind
from ..shaders import Trail
from .kinetics import EPSILON, decay_weight, direction
from .shaders import Event, KineticOnset, KineticPatch, VelocityEvent, VelocityEventPatch, VelocityEventPatchMean

START_FLOOR: float = 0.01   # widths/s: a pixel slower than this does not move, a patch slower has no speedup; START_FLOOR in velocity_event.frag


class VelocityEventBridgeSettings(BaseSettings):
    """Configuration for the velocity event bridge: its two trails, the holds and the start and stop gates."""
    fast_smooth = Field(0.025, min=0.0, max=0.5, step=0.005, description="Fast trail smoothing time, s; 0 = the flow itself")
    slow_smooth = Field(0.15, min=0.0, max=1.0, step=0.005, description="Slow trail smoothing time, s: what a change is read against")
    start_hold = Field(0.0, min=0.0, max=1.0, step=0.01, newline=True, description="Decay of a held start, s; 0 = no hold")
    stop_hold = Field(0.1, min=0.0, max=1.0, step=0.01, description="Decay of a held stop, s; 0 = no hold")
    start_gate = Field(1.0, min=0.0, max=1.0, step=0.05, newline=True, description="How far a start must be a speedup of its patch's moving pixels; 0 lets all through")
    stop_gate = Field(1.0, min=0.0, max=1.0, step=0.05, description="How far a stop must be a slowdown of its patch; 0 lets all through")
    start_patch_size = Field(128.0, min=8.0, max=1024.0, step=8.0, newline=True, description="Size of the patch the start gate reads, pixels of the bridge")
    stop_patch_size = Field(128.0, min=8.0, max=1024.0, step=8.0, description="Size of the patch the stop gate reads, pixels of the bridge")


class VelocityEventBridge(FlowBase):
    """Two short trails on the flow and the starts and stops of their difference.

    Pipeline:
        1. Take the flow's velocity → input_fbo (RGBA16F, xyz)
        2. Advance the fast and slow trails, unit-gain Emas
        3. With the start gate on, the flow's velocity where a pixel moves and whether it moves (RGBA16F,
           half size, mipmapped), its patch means over the moving pixels (a texel per patch), and those
           means' own fast and slow trails; with the stop gate on, the patch mean speeds of the two trails
           (RG16F, half size, mipmapped)
        4. Hold the start, max(0, fast - slow along the motion), and the stop, max(0, -(fast - slow)), each
           through its gate and over its hold
        5. Each along its direction → start_velocity, stop_velocity

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
        self._moving_fbo: Fbo = Fbo()                   # the start gate's: velocity * moving, moving
        self._moving_mean_fbo: Fbo = Fbo()              # a patch's mean velocity over its moving pixels, a texel per patch
        self._patch_fast_fbo: SwapFbo = SwapFbo()       # that mean, through the fast trail
        self._patch_slow_fbo: SwapFbo = SwapFbo()       # and through the slow trail
        self._patch_fbo: Fbo = Fbo()                    # the stop gate's: fast speed, slow speed
        self._start_fbo: SwapFbo = SwapFbo()
        self._stop_fbo: SwapFbo = SwapFbo()
        self._start_velocity_fbo: Fbo = Fbo()
        self._stop_velocity_fbo: Fbo = Fbo()
        self._size: tuple[int, int] = (0, 0)

        self._trail_shader: Trail = Trail()
        self._moving_shader: VelocityEventPatch = VelocityEventPatch()
        self._moving_mean_shader: VelocityEventPatchMean = VelocityEventPatchMean()
        self._patch_shader: KineticPatch = KineticPatch()
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

    @property
    def start(self) -> Texture:
        """The held start (R16F), image widths per second."""
        return self._start_fbo

    @property
    def stop(self) -> Texture:
        """The held stop (R16F), image widths per second."""
        return self._stop_fbo

    @property
    def start_velocity(self) -> Texture:
        """The start along the fast trail's direction (RGBA16F, xyz)."""
        return self._start_velocity_fbo

    @property
    def stop_velocity(self) -> Texture:
        """The stop along the slow trail's direction, the motion it ended (RGBA16F, xyz)."""
        return self._stop_velocity_fbo

    @property
    def ports(self) -> dict[str, Port]:
        return {**super().ports,
                'fast': Port(self._fast_fbo, TextureKind.VELOCITY),
                'slow': Port(self._slow_fbo, TextureKind.VELOCITY),
                'start': Port(self._start_fbo, TextureKind.SCALAR),
                'stop': Port(self._stop_fbo, TextureKind.SCALAR),
                'start_velocity': Port(self._start_velocity_fbo, TextureKind.VELOCITY),
                'stop_velocity': Port(self._stop_velocity_fbo, TextureKind.VELOCITY)}

    def allocate(self, width: int, height: int, output_width: int | None = None, output_height: int | None = None) -> None:
        """Allocate all fields at the bridge resolution."""
        super().allocate(width, height, output_width, output_height)
        self._size = (width, height)
        self._fast_fbo.allocate(width, height, GL_RGBA16F)
        self._slow_fbo.allocate(width, height, GL_RGBA16F)
        for patch, internal_format in ((self._moving_fbo, GL_RGBA16F), (self._patch_fbo, GL_RG16F)):
            patch.allocate(width // 2, height // 2, internal_format)
            glBindTexture(GL_TEXTURE_2D, patch.tex_id)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glGenerateMipmap(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, 0)
        self._start_fbo.allocate(width, height, GL_R16F)
        self._stop_fbo.allocate(width, height, GL_R16F)
        self._start_velocity_fbo.allocate(width, height, GL_RGBA16F)
        self._stop_velocity_fbo.allocate(width, height, GL_RGBA16F)
        self._follow_patch_size()
        self.reset()

        self._trail_shader.allocate()
        self._moving_shader.allocate()
        self._moving_mean_shader.allocate()
        self._patch_shader.allocate()
        self._event_shader.allocate()
        self._direction_shader.allocate()

    def deallocate(self) -> None:
        """Release all resources."""
        super().deallocate()
        for fbo in (self._fast_fbo, self._slow_fbo, self._moving_fbo, self._moving_mean_fbo, self._patch_fast_fbo,
                    self._patch_slow_fbo, self._patch_fbo, self._start_fbo, self._stop_fbo, self._start_velocity_fbo,
                    self._stop_velocity_fbo):
            fbo.deallocate()
        self._size = (0, 0)
        self._trail_shader.deallocate()
        self._moving_shader.deallocate()
        self._moving_mean_shader.deallocate()
        self._patch_shader.deallocate()
        self._event_shader.deallocate()
        self._direction_shader.deallocate()

    def reset(self) -> None:
        """Reset all fields to zero."""
        super().reset()
        for fbo in (self._fast_fbo, self._slow_fbo, self._patch_fast_fbo, self._patch_slow_fbo, self._start_fbo, self._stop_fbo,
                    self._start_velocity_fbo, self._stop_velocity_fbo):
            FlowUtil.zero(fbo)

    def _patch_grid(self) -> tuple[int, int]:
        """A texel per patch of `start_patch_size` pixels."""
        size: float = max(self.settings.start_patch_size, 1.0)
        return (max(1, round(self._size[0] / size)), max(1, round(self._size[1] / size)))

    def _follow_patch_size(self) -> None:
        """Allocate the start gate's per-patch fields a texel per patch, again when `start_patch_size` changes, from rest."""
        grid: tuple[int, int] = self._patch_grid()
        if (self._moving_mean_fbo.width, self._moving_mean_fbo.height) == grid and self._moving_mean_fbo.allocated:
            return
        for fbo in (self._moving_mean_fbo, self._patch_fast_fbo, self._patch_slow_fbo):
            fbo.deallocate()
            fbo.allocate(*grid, GL_RGBA16F)
            FlowUtil.zero(fbo)

    def set_velocity(self, velocity: Texture) -> None:
        """Take a flow's 3D velocity, x, y, z (RGB) in image widths per second, as the next input."""
        if not self._allocated:
            return
        FlowUtil.blit(self._input_fbo, velocity)

    def update(self, delta_time: float = 1.0 / 60.0) -> None:
        """Advance the trails and the held events by `delta_time`.

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

        # The mip level of a patch: the patch textures are half size
        start_lod: float = math.log2(max(s.start_patch_size, 2.0)) - 1.0
        stop_lod: float = math.log2(max(s.stop_patch_size, 2.0)) - 1.0
        if s.start_gate > 0.0:
            self._follow_patch_size()
            self._moving_fbo.begin()
            self._moving_shader.use(self._input, START_FLOOR)
            self._moving_fbo.end()
            self._mipmap(self._moving_fbo)
            self._moving_mean_fbo.begin()
            self._moving_mean_shader.use(self._moving_fbo, start_lod)
            self._moving_mean_fbo.end()
            self._advance_trail(self._patch_fast_fbo, self._moving_mean_fbo, fast_weight)
            self._advance_trail(self._patch_slow_fbo, self._moving_mean_fbo, slow_weight)
        if s.stop_gate > 0.0:
            self._patch_fbo.begin()
            self._patch_shader.use(self.fast, self.slow)
            self._patch_fbo.end()
            self._mipmap(self._patch_fbo)

        for held, event, hold, gate, patch_fast, patch_slow in (
                (self._start_fbo, Event.START, s.start_hold, s.start_gate, self._patch_fast_fbo, self._patch_slow_fbo),
                (self._stop_fbo, Event.STOP, s.stop_hold, s.stop_gate, self._patch_fbo, self._patch_fbo)):
            held.swap()
            held.begin()
            self._event_shader.use(self.fast, self.slow, held.back, event, decay_weight(delta_time, hold * 1000.0),
                                   patch_fast, patch_slow, gate, stop_lod)
            held.end()

        self._start_velocity_fbo.begin()
        self._direction_shader.use(self.fast, self.start, 1.0)
        self._start_velocity_fbo.end()

        self._stop_velocity_fbo.begin()
        self._direction_shader.use(self.slow, self.stop, 1.0)
        self._stop_velocity_fbo.end()

    @staticmethod
    def _mipmap(patch: Fbo) -> None:
        """Make a patch texture's mips, its patch means."""
        glBindTexture(GL_TEXTURE_2D, patch.tex_id)
        glGenerateMipmap(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, 0)

    def _advance_trail(self, trail: SwapFbo, input: Texture, weight: float) -> None:
        """One unit-gain Ema step: weight * previous + (1 - weight) * input."""
        trail.swap()
        trail.begin()
        self._trail_shader.use(trail.back, input, weight, 1.0 - weight)
        trail.end()


# ── numpy reference ─────────────────────────────────────────────────────

def velocity_event(fast: np.ndarray, slow: np.ndarray, held: np.ndarray, sign: float, decay: float,
                   gate: np.ndarray | float = 1.0) -> np.ndarray:
    """One held event (..., ) from the trails (..., 3): max(sign * (fast - slow) along the motion, 0) through the
    gate, or what remains of the last one when that is more. Mirrors velocity_event.frag."""
    unit: np.ndarray = direction(fast, slow)
    event: np.ndarray = np.maximum(0.0, sign * np.sum((fast - slow) * unit, axis=-1)) * gate
    return np.maximum(event, held * decay)


def moving_mean(velocity: np.ndarray) -> np.ndarray:
    """A patch's mean velocity (3,) over its moving pixels, those of its pixels (N, 3) faster than START_FLOOR;
    0 where none moves. Mirrors velocity_event_patch.frag and velocity_event_patch_mean.frag."""
    moving: np.ndarray = np.linalg.norm(velocity, axis=-1) > START_FLOOR
    return np.mean(velocity[moving], axis=0) if moving.any() else np.zeros(3)


def event_start_gate(patch_fast: np.ndarray, patch_slow: np.ndarray, strength: float) -> float:
    """How much of a start counts, 0..1, from a patch's mean moving velocity through the fast and the slow trail
    (`moving_mean` of each frame, smoothed): as far as it grows, against START_FLOOR; strength 0 lets all
    through. Mirrors velocity_event.frag."""
    fast: float = float(np.linalg.norm(patch_fast))
    change: float = (fast - float(np.linalg.norm(patch_slow))) / max(fast, START_FLOOR)
    return 1.0 + (min(max(change, 0.0), 1.0) - 1.0) * strength


def event_stop_gate(fast: np.ndarray, slow: np.ndarray, strength: float) -> float:
    """How much of a stop counts, 0..1, from the trails over one patch's pixels (N, 3): as far as its mean speed
    falls; strength 0 lets all through. Mirrors velocity_event.frag."""
    fast_speed: float = float(np.mean(np.linalg.norm(fast, axis=-1)))
    slow_speed: float = float(np.mean(np.linalg.norm(slow, axis=-1)))
    change: float = (slow_speed - fast_speed) / max(slow_speed, EPSILON)
    return 1.0 + (min(max(change, 0.0), 1.0) - 1.0) * strength


def event_velocity(trail: np.ndarray, event: np.ndarray) -> np.ndarray:
    """The event (...) along the trail's direction (..., 3). Mirrors kinetic_onset.frag at strength 1."""
    length: np.ndarray = np.linalg.norm(trail, axis=-1, keepdims=True)
    unit: np.ndarray = np.divide(trail, length, out=np.zeros_like(trail), where=length > EPSILON)
    return unit * np.maximum(0.0, event)[..., None]
