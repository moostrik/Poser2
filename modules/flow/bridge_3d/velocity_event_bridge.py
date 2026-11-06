"""Velocity Event Bridge.

Next to the velocity trail, which is the motion's push, this reads the motion's events: where a move
starts and where it stops. Two short trails on the flow's velocity, fast and slow, unit-gain Emas; their
difference along the motion's direction rises at a start and falls at a stop. Each event is peak-held over
`hold`. A start counts as far as its patch speeds up too (`start_gate`), a stop as far as it slows down
(`stop_gate`), both over patches of `stop_tile`: a limb arriving at a pixel or leaving it changes that
pixel's speed, not its patch's, so it reads as neither. The start gate reads the patch's mean x/y
velocity, in which jitter in every direction cancels, held to a floor (START_FLOOR) so a still patch
reads no speedup; in a mean speed the jitter adds up, and more in the fast trail than in the slow one, so
every patch would read as speeding up. The stop gate reads the mean speed, which jitter never lowers and a
turning limb keeps, where its mean velocity would shrink and read as a stop. No blur and no advect: the
events stay sharp.

Out come, in image widths per second like the flow:
    start, stop                     the held events (R16F)
    start_velocity, stop_velocity   the motion's direction times its event (RGBA16F, xyz): a start along the
                                    fast trail's direction, a stop along the slow one's, the motion it ended

z goes in times `z_weight`, so a move toward or away from the camera can start and stop as well, as far as
its noise allows. Steps every tick on its dt, as the other bridges do.
"""

import math

import numpy as np
from OpenGL.GL import GL_LINEAR, GL_LINEAR_MIPMAP_LINEAR, GL_R16F, GL_RG16F, GL_RGBA16F, GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MIN_FILTER, glBindTexture, glGenerateMipmap, glTexParameteri

from modules.gl import Fbo, SwapFbo, Texture
from modules.settings import Field, BaseSettings
from .. import FlowBase, FlowUtil, Port, TextureKind
from ..shaders import Trail
from .kinetics import EPSILON, decay_weight, direction
from .shaders import Event, KineticOnset, KineticPatch, ScaleVelocity, VelocityEvent, VelocityEventPatch

START_FLOOR: float = 0.01   # widths/s: a patch's mean x/y speed below this has no speedup; START_FLOOR in velocity_event.frag


class VelocityEventBridgeSettings(BaseSettings):
    """Configuration for the velocity event bridge: its two trails, z, the hold and the start and stop gates."""
    fast_smooth = Field(0.025, min=0.0, max=0.5, step=0.005, description="Fast trail smoothing time, s; 0 = the flow itself")
    slow_smooth = Field(0.15, min=0.0, max=1.0, step=0.005, description="Slow trail smoothing time, s: what a change is read against")
    z_weight = Field(1.0, min=0.0, max=1.0, step=0.05, description="z in the events; 0 = x/y only")
    hold = Field(0.1, min=0.0, max=1.0, step=0.01, newline=True, description="Decay of a held start or stop, s; 0 = no hold")
    start_gate = Field(1.0, min=0.0, max=1.0, step=0.05, description="How far a start must be a speedup of its patch; 0 lets all through")
    stop_gate = Field(1.0, min=0.0, max=1.0, step=0.05, description="How far a stop must be a slowdown of its patch; 0 lets all through")
    stop_tile = Field(128.0, min=8.0, max=1024.0, step=8.0, description="Size of a gate's patch, both gates, pixels of the bridge")


class VelocityEventBridge(FlowBase):
    """Two short trails on the flow and the starts and stops of their difference.

    Pipeline:
        1. Take the flow's velocity, z times `z_weight` → input_fbo (RGBA16F, xyz)
        2. Advance the fast and slow trails, unit-gain Emas
        3. With the start gate on, the patch mean x/y velocities (RGBA16F, mipmapped); with the stop gate
           on, the patch mean speeds (RG16F, mipmapped)
        4. Hold the start, max(0, fast - slow along the motion), and the stop, max(0, -(fast - slow)), each through its gate
        5. Each along its direction → start_velocity, stop_velocity

    FBOs (from FlowBase):
        - input_fbo: The velocity, z weighted
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
        self._start_patch_fbo: Fbo = Fbo()  # the start gate's: fast x/y, slow x/y
        self._patch_fbo: Fbo = Fbo()        # the stop gate's: fast speed, slow speed
        self._start_fbo: SwapFbo = SwapFbo()
        self._stop_fbo: SwapFbo = SwapFbo()
        self._start_velocity_fbo: Fbo = Fbo()
        self._stop_velocity_fbo: Fbo = Fbo()

        self._scale_shader: ScaleVelocity = ScaleVelocity()
        self._trail_shader: Trail = Trail()
        self._start_patch_shader: VelocityEventPatch = VelocityEventPatch()
        self._patch_shader: KineticPatch = KineticPatch()
        self._event_shader: VelocityEvent = VelocityEvent()
        self._direction_shader: KineticOnset = KineticOnset()     # unit of a trail times an amount

    @property
    def velocity_input(self) -> Texture:
        """The flow's velocity, z times z_weight."""
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
        self._fast_fbo.allocate(width, height, GL_RGBA16F)
        self._slow_fbo.allocate(width, height, GL_RGBA16F)
        for patch, internal_format in ((self._start_patch_fbo, GL_RGBA16F), (self._patch_fbo, GL_RG16F)):
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
        self.reset()

        self._scale_shader.allocate()
        self._trail_shader.allocate()
        self._start_patch_shader.allocate()
        self._patch_shader.allocate()
        self._event_shader.allocate()
        self._direction_shader.allocate()

    def deallocate(self) -> None:
        """Release all resources."""
        super().deallocate()
        for fbo in (self._fast_fbo, self._slow_fbo, self._start_patch_fbo, self._patch_fbo, self._start_fbo, self._stop_fbo,
                    self._start_velocity_fbo, self._stop_velocity_fbo):
            fbo.deallocate()
        self._scale_shader.deallocate()
        self._trail_shader.deallocate()
        self._start_patch_shader.deallocate()
        self._patch_shader.deallocate()
        self._event_shader.deallocate()
        self._direction_shader.deallocate()

    def reset(self) -> None:
        """Reset all fields to zero."""
        super().reset()
        for fbo in (self._fast_fbo, self._slow_fbo, self._start_fbo, self._stop_fbo, self._start_velocity_fbo, self._stop_velocity_fbo):
            FlowUtil.zero(fbo)

    def set_velocity(self, velocity: Texture) -> None:
        """Take a flow's 3D velocity, x, y, z (RGB) in image widths per second, z times z_weight, as the next input."""
        if not self._allocated:
            return
        self._input_fbo.begin()
        self._scale_shader.use(velocity, (1.0, 1.0, self.settings.z_weight))
        self._input_fbo.end()

    def update(self, delta_time: float = 1.0 / 60.0) -> None:
        """Advance the trails and the held events by `delta_time`.

        Args:
            delta_time: Seconds since the previous update
        """
        if not self._allocated:
            return
        s: VelocityEventBridgeSettings = self.settings

        self._advance_trail(self._fast_fbo, decay_weight(delta_time, s.fast_smooth * 1000.0))
        self._advance_trail(self._slow_fbo, decay_weight(delta_time, s.slow_smooth * 1000.0))

        if s.start_gate > 0.0:
            self._update_patch(self._start_patch_fbo, self._start_patch_shader)
        if s.stop_gate > 0.0:
            self._update_patch(self._patch_fbo, self._patch_shader)

        decay: float = decay_weight(delta_time, s.hold * 1000.0)
        lod: float = math.log2(s.stop_tile) - 1.0            # the patch textures are half size
        for held, event, gate, patch in ((self._start_fbo, Event.START, s.start_gate, self._start_patch_fbo),
                                         (self._stop_fbo, Event.STOP, s.stop_gate, self._patch_fbo)):
            held.swap()
            held.begin()
            self._event_shader.use(self.fast, self.slow, held.back, event, decay, patch, gate, lod)
            held.end()

        self._start_velocity_fbo.begin()
        self._direction_shader.use(self.fast, self.start, 1.0)
        self._start_velocity_fbo.end()

        self._stop_velocity_fbo.begin()
        self._direction_shader.use(self.slow, self.stop, 1.0)
        self._stop_velocity_fbo.end()

    def _update_patch(self, patch: Fbo, shader: KineticPatch | VelocityEventPatch) -> None:
        """Write the trails into a gate's patch texture and make its mips, the patch means."""
        patch.begin()
        shader.use(self.fast, self.slow)
        patch.end()
        glBindTexture(GL_TEXTURE_2D, patch.tex_id)
        glGenerateMipmap(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, 0)

    def _advance_trail(self, trail: SwapFbo, weight: float) -> None:
        """One unit-gain Ema step: weight * previous + (1 - weight) * input."""
        trail.swap()
        trail.begin()
        self._trail_shader.use(trail.back, self._input, weight, 1.0 - weight)
        trail.end()


# ── numpy reference ─────────────────────────────────────────────────────

def velocity_event(fast: np.ndarray, slow: np.ndarray, held: np.ndarray, sign: float, decay: float,
                   gate: np.ndarray | float = 1.0) -> np.ndarray:
    """One held event (..., ) from the trails (..., 3): max(sign * (fast - slow) along the motion, 0) through the
    gate, or what remains of the last one when that is more. Mirrors velocity_event.frag."""
    unit: np.ndarray = direction(fast, slow)
    event: np.ndarray = np.maximum(0.0, sign * np.sum((fast - slow) * unit, axis=-1)) * gate
    return np.maximum(event, held * decay)


def event_gate(fast: np.ndarray, slow: np.ndarray, sign: float, strength: float) -> float:
    """How much of an event counts, from the trails over one patch's pixels (N, 3), 0..1; strength 0 lets
    every event through. A start (sign +1) as far as the patch's mean x/y velocity grows, against
    START_FLOOR; a stop (sign -1) as far as its mean speed falls. Mirrors velocity_event.frag."""
    if sign > 0.0:
        patch_fast: float = float(np.linalg.norm(np.mean(fast[:, :2], axis=0)))
        patch_slow: float = float(np.linalg.norm(np.mean(slow[:, :2], axis=0)))
        change: float = (patch_fast - patch_slow) / max(patch_fast, START_FLOOR)
    else:
        fast_speed: float = float(np.mean(np.linalg.norm(fast, axis=-1)))
        slow_speed: float = float(np.mean(np.linalg.norm(slow, axis=-1)))
        change = (slow_speed - fast_speed) / max(slow_speed, EPSILON)
    return 1.0 + (min(max(change, 0.0), 1.0) - 1.0) * strength


def event_velocity(trail: np.ndarray, event: np.ndarray) -> np.ndarray:
    """The event (...) along the trail's direction (..., 3). Mirrors kinetic_onset.frag at strength 1."""
    length: np.ndarray = np.linalg.norm(trail, axis=-1, keepdims=True)
    unit: np.ndarray = np.divide(trail, length, out=np.zeros_like(trail), where=length > EPSILON)
    return unit * np.maximum(0.0, event)[..., None]
