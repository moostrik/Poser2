"""Conductor — the light system's threaded tick loop: owns the 30 Hz time base and the
fixed per-tick order (motor → playhead → update callbacks → command → compositor → output).

It draws nothing and decides nothing itself: the state machine (an update callback) decides
looks and motor commands, the layers draw, the Compositor mixes. The Conductor guarantees
that every tick those happen exactly once, in exactly that order, on one thread — and
forwards the finished Frame to the board and the output callbacks (UDP sender, audio, render).
"""

from threading import Event, Thread
from typing import Any, Callable

from modules.utils import HotReloadMethods, ThreadPriority, set_current_thread_priority
from modules.gl import FpsCounter
from modules.tracker.panoramic.settings import DistortionSettings

from .clock import Clock, Tick
from .frame import Frame, FrameCallback
from .motor import MotorController, MotorMode
from .playhead import Playhead
from .settings import LightSettings, LayerId, DebugLayer
from .layers import (BaseLayer, Compositor, Mix, PoseWaves, Fill, Pulse, Chase, Lines, Random,
                     Harmonic, PlayerLines, CameraLight, PlayheadFlash, PlayheadHaunted,
                     Searchlight, ProjectionPlayhead, PlayheadTest, SoundLight, PoseInstrument, Flood,
                     WindDown)
from modules.board import PlayheadSignals

from ..board import Board

import logging
logger = logging.getLogger(__name__)


def _debug_motor_mode(selection: DebugLayer, layers: dict[LayerId, BaseLayer]) -> MotorMode | None:
    """The debug auto-follow: derive the motor mode from the selected debug layer's
    class — `ProjectionLayer` → PROJECTION, `BeamLayer` → BEAM; OFF → None (debug disarmed, the machine
    owns the motor). Selecting a layer is the only gesture: choosing it IS turning debug
    on, and the speed rides the selection."""
    if selection == DebugLayer.OFF:
        return None
    layer = layers.get(LayerId(int(selection)))
    if layer is None:
        return None
    return MotorMode.PROJECTION if layer.SHIFTED else MotorMode.BEAM


class Conductor(Thread):
    """Runs the light loop at a fixed rate (light_rate Hz); see the module docstring."""

    def __init__(self, config: LightSettings, distortion: DistortionSettings, board: Board, pose_stage: int) -> None:
        super().__init__(daemon=True, name="LightConductor")

        self._stop_event = Event()

        self._config: LightSettings = config
        self._board: Board          = board
        self._pose_stage: int       = pose_stage
        # Boot failsafe #3: a preset saved mid-debug (a projection layer selected) must never
        # auto-derive PROJECTION at power-on — the installation always wakes in the show.
        config.debug = DebugLayer.OFF
        self._motor_controller      = MotorController(config.motor)
        self._playhead              = Playhead(config.playhead)
        self._clock                 = Clock(config.clock, config.light_rate)

        resolution: int             = config.light_resolution
        num_players: int            = config.max_poses

        # The unified layer pool — one instance per LayerId, each reading its own settings
        # group (beam_layers / projection_layers, mirroring the folder taxonomy); each layer's
        # mode lives in its class (BeamLayer/ProjectionLayer).
        LO, HI = config.beam_layers, config.projection_layers
        self.layers: dict[LayerId, BaseLayer] = {
            LayerId.sound_light:        SoundLight     (resolution, LO.sound_light,      board),
            LayerId.searchlight:       Searchlight    (resolution, LO.searchlight,     board),
            LayerId.playhead_flash:     PlayheadFlash  (resolution, LO.playhead_flash,   board, pose_stage),
            LayerId.wind_down:          WindDown       (resolution, LO.wind_down,        board),
            LayerId.playhead_haunted:   PlayheadHaunted(resolution, LO.playhead_haunted, board, pose_stage),
            LayerId.playhead_test:      PlayheadTest   (resolution, LO.playhead_test,    board),
            LayerId.pose_instrument:    PoseInstrument (resolution, HI.pose_instrument,  board, pose_stage),
            LayerId.projection_playhead:      ProjectionPlayhead   (resolution, HI.projection_playhead,    board),
            LayerId.flood:              Flood          (resolution, HI.flood,            board),
            LayerId.test_pose_waves:    PoseWaves      (resolution, num_players, HI.test_pose_waves, self._clock.interval, board, pose_stage),
            LayerId.test_harmonic:      Harmonic       (resolution, HI.test_harmonic,    board),
            LayerId.test_player_lines:  PlayerLines    (resolution, HI.test_player_lines, board, pose_stage),
            LayerId.test_calibration:   CameraLight    (resolution, HI.test_calibration, distortion, config.num_cameras, board),
            LayerId.test_fill:   Fill  (resolution, HI.test_fill,   board),
            LayerId.test_pulse:  Pulse (resolution, HI.test_pulse,  board),
            LayerId.test_chase:  Chase (resolution, HI.test_chase,  board),
            LayerId.test_lines:  Lines (resolution, HI.test_lines,  board),
            LayerId.test_random: Random(resolution, HI.test_random, board),
        }

        self._compositor = Compositor(config, self.layers)

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
        # Outrank the process's other threads (inference, GL, analytics): measured to take the
        # clock's lateness under in-process native load from ~4 ms mean to ~60 µs. HIGHEST, not
        # TIME_CRITICAL — no measurable difference, and this thread busy-spins ~1 ms per tick.
        # The light sender and UDP receiver run at this same level: one pipeline, one priority.
        set_current_thread_priority(ThreadPriority.HIGHEST)
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
        """Forward the state machine's motor command (None = relinquish → STOPPED)."""
        self._motor_controller.set_mode(mode)

    # ------------------------------------------------------------------
    # Per-tick
    # ------------------------------------------------------------------

    def _update(self, tick: Tick) -> None:
        # Debug auto-follow: while a debug layer is selected, the motor follows its mode
        # (outranking the machine); OFF relinquishes back to the machine.
        self._motor_controller.set_debug_mode(
            _debug_motor_mode(self._config.debug, self.layers))

        # Measure the motor, advance the playhead under the command in force over this dt, and
        # publish the playhead BEFORE the update callbacks: the state machine reads it this tick
        # (no added latency). Phase is NaN while the motor is STOPPED (no meaningful playhead);
        # the bar counter stays monotonic throughout.
        motor   = self._motor_controller.tick()
        command = self._motor_controller.command
        self._playhead.tick(tick.dt, motor, command)
        playhead = self._playhead.phase
        self._board.set_playhead(PlayheadSignals(
            phase=playhead, bars=self._playhead.bars, synced=self._playhead.synced,
            ring_formed=self._playhead.ring_formed))

        self._notify_update()

        # The state machine commands the motor from inside those callbacks, and on a state's
        # entry tick it also hands the Compositor that state's mix — so the frame takes the
        # command as it stands now: the rpm it is sent with is the rpm its content is drawn for.
        command = self._motor_controller.command

        frame = Frame(self._config.light_resolution, tick, motor, command, playhead=playhead)
        self._compositor.render(frame)

        # Master brightness — the ring and the beam lights alike
        m = self._config.master
        if m != 1.0:
            frame.white      *= m
            frame.blue       *= m
            frame.beam_lights *= m

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
