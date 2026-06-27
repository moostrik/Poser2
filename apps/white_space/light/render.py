"""Render — threaded LED light renderer running at a fixed rate.

Drives the per-tick loop: advances the clock and motor, renders the active layers (each pulls the
board slices it needs and blends itself into the per-tick Frame), applies master post-processing,
and forwards the Frame to the board and to registered output callbacks (UDP sender, audio, render).
"""

import math
from threading import Event, Thread
from typing import Any, Callable

import numpy as np

from modules.utils import HotReloadMethods
from modules.gl import FpsCounter
from modules.tracker.panoramic.settings import DistortionSettings

from .clock import Clock, Tick
from .frame import Frame, FrameCallback
from .motor import MotorController, MotorState
from .playhead import Playhead
from .settings import LightSettings, LowLayerId, HighLayerId
from .layers import BaseLayer, PoseWaves, Fill, Pulse, Chase, Lines, Random, Harmonic, PlayerLines, CameraLight, PlayheadFlash, PlayheadLow, PlayheadHigh
from ..board import Board

import logging
logger = logging.getLogger(__name__)

# Crossfade dead-zone: each fade starts `deadzone` ABOVE the mode below it and ends `deadzone` BELOW
# its own mode rpm, so a band around every mode rpm shows only that mode (no neighbour bleed on jitter).
CROSSFADE_DEADZONE: float = 0.05


def _ease(t: float) -> float:
    """Sine ease-in-out on [0,1] — the crossfade accelerates out of one comp and settles into the next."""
    return 0.5 - 0.5 * math.cos(math.pi * t)


def crossfade_weights(rpm: float, idle_rpm: float, low_rpm: float,
                      high_cross_rpm: float, deadzone: float = CROSSFADE_DEADZONE) -> tuple[float, float, float]:
    """``(w_idle, w_low, w_high)`` for the motor rpm. Each comp fades in over a band *inside* the gap
    between mode rpms — from a deadzone above the mode below it to a deadzone below its own mode rpm —
    so the dead zone around each mode rpm keeps the neighbouring comp off under rpm jitter (e.g. in low
    mode the high comp stays fully off). The fade is sine-eased in/out. High ends at ``high_cross_rpm``
    (nothing higher to bleed). Weights sum to 1."""
    eps = 1e-6
    s = min(1.0, max(0.0, (rpm - idle_rpm * (1.0 + deadzone)) / max(low_rpm * (1.0 - deadzone) - idle_rpm * (1.0 + deadzone), eps)))
    h = min(1.0, max(0.0, (rpm - low_rpm * (1.0 + deadzone)) / max(high_cross_rpm - low_rpm * (1.0 + deadzone), eps)))
    s_in, h_in = _ease(s), _ease(h)
    return (1.0 - s_in) * (1.0 - h_in), s_in * (1.0 - h_in), h_in


class Render(Thread):
    """Runs the LED render loop at a fixed rate (light_rate Hz). Each tick it advances the clock
    and motor, renders the active layers — which pull the board slices they need — and publishes
    the resulting Frame to the board and to the registered output callbacks (UDP sender, audio,
    render board).
    """

    def __init__(self, config: LightSettings, distortion: DistortionSettings, board: Board, pose_stage: int) -> None:
        super().__init__(daemon=True, name="LightRenderer")

        self._stop_event = Event()

        self._config: LightSettings = config
        self._board: Board          = board
        self._pose_stage: int       = pose_stage
        self._motor_controller      = MotorController(config.motor)
        self._playhead              = Playhead(config.playhead)
        self._clock                 = Clock(config.clock, config.light_rate)

        resolution: int             = config.light_resolution
        num_players: int            = config.max_poses

        # Composition settings are grouped by category; build each layer once. The test layers are
        # shared between the low and high pools.
        low, high, test = config.low, config.high, config.test
        fill   = Fill   (resolution, test.fill,   board)
        pulse  = Pulse  (resolution, test.pulse,  board)
        chase  = Chase  (resolution, test.chase,  board)
        lines  = Lines  (resolution, test.lines,  board)
        random = Random (resolution, test.random, board)
        test_low  = {LowLayerId.fill: fill,  LowLayerId.pulse: pulse,  LowLayerId.chase: chase,
                     LowLayerId.lines: lines, LowLayerId.random: random}
        test_high = {HighLayerId.fill: fill,  HighLayerId.pulse: pulse,  HighLayerId.chase: chase,
                     HighLayerId.lines: lines, HighLayerId.random: random}

        # Pools scoped per slot: idle/low select from _low_pool, high selects from _high_pool.
        self._low_pool: dict[LowLayerId, BaseLayer] = {
            LowLayerId.playhead:       PlayheadLow  (resolution, low.playhead,       board),
            LowLayerId.playhead_flash: PlayheadFlash(resolution, low.playhead_flash, board, pose_stage),
            **test_low,
        }
        self._high_pool: dict[HighLayerId, BaseLayer] = {
            HighLayerId.pose_waves:   PoseWaves   (resolution, num_players, high.pose_waves, self._clock.interval, board, pose_stage),
            HighLayerId.harmonic:     Harmonic    (resolution, high.harmonic,     board),
            HighLayerId.player_lines: PlayerLines (resolution, high.player_lines, board, pose_stage),
            HighLayerId.calibration:  CameraLight (resolution, high.calibration, distortion, config.num_cameras, board),
            HighLayerId.playhead:     PlayheadHigh (resolution, high.playhead,    board),
            **test_high,
        }
        # Reusable scratch frame each slot's selected layers blend into before crossfading.
        self._scratch = Frame(resolution, Tick(0.0, 0.0, 0.0, 0.0, 0))
        # Layer instances selected last tick (any slot), to reset a layer when it is deselected.
        self._prev_selected: set[BaseLayer] = set()

        self.fps_counter = FpsCounter()
        self._update_callbacks: list[Callable[[], Any]] = []
        self._render_callbacks: list[FrameCallback] = []

        self.hot_reloader = HotReloadMethods(self.__class__, True)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._motor_controller.start()
        super().start()

    def stop(self) -> None:
        self._motor_controller.stop()
        self._stop_event.set()
        if self.is_alive():
            self.join()

    def notify_fall(self, *args: object) -> None:
        """Signal a revolution fall edge; forwarded to the motor controller.
        Accepts and ignores any receiver callback args (OSC address/values)."""
        self._motor_controller.notify_fall()

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                tick = self._clock.next_tick()   # blocks until the next frame deadline
                self._update(tick)
            except Exception:
                logger.exception("Error in light render update")

    # ------------------------------------------------------------------
    # Per-tick
    # ------------------------------------------------------------------

    def _update(self, tick: Tick) -> None:
        # Advance the motor + playhead FIRST so this tick's playhead is published to the
        # board before the update callbacks run the pose-LERP (whose final node reads it).
        motor = self._motor_controller.tick()                   # commands rpm from its mode
        self._playhead.tick(tick.dt, motor)                     # NCO tracks the motor
        # NaN while the light isn't rotating (motor STOPPED) → downstream pose-phase features
        # read "no meaningful playhead" instead of a stale held angle (offset applied).
        playhead = self._playhead.phase
        self._board.set_playhead(playhead)
        # The measured MotorState rides the composition Frame (motor=...) below — no
        # separate board slice needed; nothing reads the raw phase this-tick.

        # Pre-render state callbacks (sequencer; interpolators drive the pose-LERP,
        # which now reads this tick's board playhead).
        for cb in self._update_callbacks:
            try:
                cb()
            except Exception:
                logger.exception("Error in light render update callback")

        frame = Frame(self._config.light_resolution, tick, motor, playhead=playhead)
        self._compose(frame, tick, motor)

        # Master brightness (composition level). The lamp gamma/floor mapping lives in osc_light.
        m = self._config.master
        if m != 1.0:
            frame.white *= m
            frame.blue  *= m

        self._board.set_composition_output(frame)
        self._notify_render(frame)
        self.fps_counter.tick()

    def _compose(self, frame: Frame, tick: Tick, motor: MotorState) -> None:
        """Crossfade the three speed-selected layers into the frame by motor rpm.

        Each selected slot's layer is rendered into the reusable scratch frame (its blend applies
        into the zeroed scratch), then accumulated by weight. The high slot is ring-rolled by
        `light_offset` first. Only slots with weight > 0 render.
        """
        cfg = self._config
        rpm = motor.measured_rpm if motor.locked else motor.target_rpm
        w_idle, w_low, w_high = crossfade_weights(
            rpm, cfg.motor.idle_rpm, cfg.motor.low_rpm, cfg.high_cross_rpm)
        shift = int(round(cfg.light_offset / (2.0 * np.pi) * frame.resolution)) % frame.resolution

        # Resolve each slot's selected layers from its own scoped pool.
        idle = self._resolve(self._low_pool,  cfg.idle_layers)
        low  = self._resolve(self._low_pool,  cfg.low_layers)
        high = self._resolve(self._high_pool, cfg.high_layers)
        self._reset_deselected(idle + low + high)

        self._add_slot(frame, tick, motor, idle, w_idle, 0)
        self._add_slot(frame, tick, motor, low,  w_low,  0)
        self._add_slot(frame, tick, motor, high, w_high, shift)   # high slot gets the light_offset roll

    @staticmethod
    def _resolve(pool: dict, ids: list) -> list[BaseLayer]:
        """Selected layer instances for a slot, in selection (blend) order; missing ids skipped."""
        return [pool[i] for i in ids if i in pool]

    def _add_slot(self, frame: Frame, tick: Tick, motor: MotorState,
                  layers: list[BaseLayer], weight: float, shift: int) -> None:
        """Render a slot's selected layers into the scratch frame — each blends in via its own blend
        mode — then add `weight ×` the composite to the output; `shift` ring-rolls (high light_offset)."""
        if weight <= 0.0 or not layers:
            return
        scratch = self._scratch
        scratch.tick, scratch.motor, scratch.playhead = tick, motor, frame.playhead
        scratch.light_img.fill(0.0)
        drew = False
        for layer in layers:
            try:
                layer.render(scratch)   # blends into the (shared) scratch via its own blend mode
                drew = True
            except Exception:
                logger.exception("Error in %s.render", layer.__class__.__name__)
        if not drew:
            return
        sw, sb = scratch.white, scratch.blue
        if shift:
            sw, sb = np.roll(sw, shift), np.roll(sb, shift)
        frame.white += weight * sw
        frame.blue  += weight * sb

    def _reset_deselected(self, selected: list[BaseLayer]) -> None:
        """Reset layers no longer selected in any slot, so they restart fresh when reselected."""
        current = set(selected)
        for layer in self._prev_selected - current:
            layer.reset()
        self._prev_selected = current


    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def add_update_callback(self, callback: Callable[[], Any]) -> None:
        """Register a callback fired at the start of each tick, before rendering.
        Use for time-driven state advances (sequencer, interpolators).
        """
        self._update_callbacks.append(callback)

    def add_render_callback(self, callback: FrameCallback) -> None:
        """Register a callback fired after each tick with the new Frame.
        Use for output consumers (hardware sender, audio).
        """
        self._render_callbacks.append(callback)

    def _notify_render(self, frame: Frame) -> None:
        for cb in self._render_callbacks:
            try:
                cb(frame)
            except Exception:
                logger.exception("Error in light render output callback")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def fps(self) -> float:
        return self.fps_counter.get_fps()

    def reset(self) -> None:
        for layer in set(self._low_pool.values()) | set(self._high_pool.values()):
            layer.reset()
