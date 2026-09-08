"""Conductor — the light system's threaded tick loop: owns the 30 Hz time base and the
fixed per-tick order (motor → playhead → update callbacks → compositor → output).

It draws nothing and decides nothing itself: the state machine (an update callback) decides
looks and motor commands, the layers draw, the Compositor mixes. The Conductor guarantees
that every tick those happen exactly once, in exactly that order, on one thread — and
forwards the finished Frame to the board and the output callbacks (UDP sender, audio, render).
"""

from threading import Event, Thread
from typing import Any, Callable

from modules.utils import HotReloadMethods
from modules.gl import FpsCounter
from modules.tracker.panoramic.settings import DistortionSettings

from .clock import Clock, Tick
from .frame import Frame, FrameCallback
from .motor import MotorController, MotorMode
from .playhead import Playhead
from .settings import LightSettings, LayerId
from .layers import (BaseLayer, Compositor, Mix, PoseWaves, Fill, Pulse, Chase, Lines, Random,
                     Harmonic, PlayerLines, CameraLight, PlayheadFlash, HauntedFlash,
                     PlayheadLow, PlayheadHigh)
from modules.board import PlayheadSignals

from ..board import Board

import logging
logger = logging.getLogger(__name__)

# Spun-content layers: their pixels ride the fast ring, so they get the light_phase shift.
# Doubles as the high-layer knowledge for the debug auto-follow (until the LowLayer/HighLayer
# base classes carry the flag).
_SHIFTED: set[LayerId] = {LayerId.pose_waves, LayerId.harmonic, LayerId.player_lines,
                          LayerId.calibration, LayerId.playhead_marker}


def _debug_motor_mode(selection: list[LayerId]) -> MotorMode:
    """The debug auto-follow: derive the motor regime from the selected debug layers —
    any high layer → HIGH, else any low layer → LOW, empty selection → STOPPED. Selecting
    a layer is the only gesture: the speed rides the selection, inside debug's consent."""
    if any(id in _SHIFTED for id in selection):
        return MotorMode.HIGH
    if selection:
        return MotorMode.LOW
    return MotorMode.STOPPED


class Conductor(Thread):
    """Runs the light loop at a fixed rate (light_rate Hz); see the module docstring."""

    def __init__(self, config: LightSettings, distortion: DistortionSettings, board: Board, pose_stage: int) -> None:
        super().__init__(daemon=True, name="LightConductor")

        self._stop_event = Event()

        self._config: LightSettings = config
        self._board: Board          = board
        self._pose_stage: int       = pose_stage
        # Boot failsafe #3: a preset saved mid-debug (debug on + a high layer ticked) must
        # never auto-derive HIGH at power-on — the installation always wakes in the show.
        config.debug = False
        self._motor_controller      = MotorController(config.motor)
        self._playhead              = Playhead(config.playhead)
        self._clock                 = Clock(config.clock, config.light_rate)

        resolution: int             = config.light_resolution
        num_players: int            = config.max_poses

        # The unified layer pool — one instance per LayerId, each reading its own settings group.
        L = config.layers
        self.layers: dict[LayerId, BaseLayer] = {
            LayerId.playhead_lamp:   PlayheadLow  (resolution, L.playhead_lamp,   board),
            LayerId.playhead_flash:  PlayheadFlash(resolution, L.playhead_flash,  board, pose_stage),
            LayerId.haunt_flash:     HauntedFlash (resolution, L.haunt_flash,     board, pose_stage),
            LayerId.pose_waves:      PoseWaves    (resolution, num_players, L.pose_waves, self._clock.interval, board, pose_stage),
            LayerId.harmonic:        Harmonic     (resolution, L.harmonic,        board),
            LayerId.player_lines:    PlayerLines  (resolution, L.player_lines,    board, pose_stage),
            LayerId.calibration:     CameraLight  (resolution, L.calibration, distortion, config.num_cameras, board),
            LayerId.playhead_marker: PlayheadHigh (resolution, L.playhead_marker, board),
            LayerId.fill:   Fill  (resolution, L.fill,   board),
            LayerId.pulse:  Pulse (resolution, L.pulse,  board),
            LayerId.chase:  Chase (resolution, L.chase,  board),
            LayerId.lines:  Lines (resolution, L.lines,  board),
            LayerId.random: Random(resolution, L.random, board),
        }

        self._compositor = Compositor(config, self.layers, _SHIFTED)

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
                logger.exception("Error in light conductor update")

    # ------------------------------------------------------------------
    # State-machine command channels (forwarded; the machine is the sole caller)
    # ------------------------------------------------------------------

    def set_mix(self, entries: Mix) -> None:
        """Forward the state machine's mix to the Compositor."""
        self._compositor.set_mix(entries)

    def reset_layers(self, ids: list[LayerId]) -> None:
        """Forward an explicit layer reset (a show state's ``enter()``) to the Compositor."""
        self._compositor.reset_layers(ids)

    def set_motor_mode(self, mode: MotorMode | None) -> None:
        """Forward the state machine's motor command (None = relinquish to settings.mode)."""
        self._motor_controller.set_mode(mode)

    # ------------------------------------------------------------------
    # Per-tick
    # ------------------------------------------------------------------

    def _update(self, tick: Tick) -> None:
        # Debug auto-follow: while the debug override is on, the motor follows the selected
        # debug layers' regime (outranking the machine); off relinquishes back to the machine.
        self._motor_controller.set_debug_mode(
            _debug_motor_mode(list(self._config.debug_layers)) if self._config.debug else None)

        # Advance motor + playhead and publish the playhead BEFORE the update callbacks: the
        # state machine and pose-LERP read it. Phase is NaN while the motor is STOPPED (no
        # meaningful playhead); the bar counter stays monotonic throughout.
        motor = self._motor_controller.tick()
        self._playhead.tick(tick.dt, motor)
        playhead = self._playhead.phase
        self._board.set_playhead(PlayheadSignals(
            phase=playhead, bars=self._playhead.bars, synced=self._playhead.synced,
            ring_formed=self._playhead.ring_formed, spin_down=self._playhead.spin_down))

        self._notify_update()

        frame = Frame(self._config.light_resolution, tick, motor, playhead=playhead)
        self._compositor.render(frame)

        # Master brightness
        m = self._config.master
        if m != 1.0:
            frame.white *= m
            frame.blue  *= m

        self._board.set_composition_output(frame)
        self._notify_render(frame)
        self.fps_counter.tick()

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def add_update_callback(self, callback: Callable[[], Any]) -> None:
        """Register a callback fired at the start of each tick, before rendering.
        Use for time-driven state advances (state machine, interpolators).
        """
        self._update_callbacks.append(callback)

    def add_render_callback(self, callback: FrameCallback) -> None:
        """Register a callback fired after each tick with the new Frame.
        Use for output consumers (hardware sender, audio).
        """
        self._render_callbacks.append(callback)

    def _notify_update(self) -> None:
        for cb in self._update_callbacks:
            try:
                cb()
            except Exception:
                logger.exception("Error in light conductor update callback")

    def _notify_render(self, frame: Frame) -> None:
        for cb in self._render_callbacks:
            try:
                cb(frame)
            except Exception:
                logger.exception("Error in light conductor render callback")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def fps(self) -> float:
        return self.fps_counter.get_fps()
