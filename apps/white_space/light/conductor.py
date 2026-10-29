"""Conductor — the light system's threaded tick loop: owns the ``light_rate`` time base and the
fixed per-tick order (motor → playhead → update callbacks → command → compositor → output).

It draws nothing and decides nothing itself: the state machine (an update callback) decides
looks and motor commands, the layers draw, the Compositor mixes. The Conductor guarantees
that every tick those happen exactly once, in exactly that order, on one thread — and
forwards the finished LightFrame to the board and the output callbacks (UDP sender, audio, render).
"""

from threading import Event, Thread
from typing import Any, Callable

from modules.gl import FpsCounter

from .clock import Clock, Tick
from .frame import LightFrame, LightFrameCallback
from .motor import MotorController, MotorMode
from .playhead import Playhead
from .settings import LightSettings, LayerId, DebugLayer
from .layers import (BaseLayer, Compositor, Mix,
                     BeamBlueSound, BeamPlayhead, BeamFlash, BeamWindDown, BeamTest,
                     PoseInstrument, PoseInstrumentSettings, ProjectionPlayhead, Flood,
                     TestPlayerLines, TestCalibration, TestFill, TestPulse, TestChase, TestLines,
                     TestRandom, TestPoseWaves, TestHarmonic)
from modules.board import PlayheadSignals

from ..render_board import RenderBoard

import logging
logger = logging.getLogger(__name__)


def _debug_motor_mode(selection: DebugLayer, layers: dict[LayerId, BaseLayer]) -> MotorMode | None:
    """The debug auto-follow: derive the motor mode from the selected debug layer's class —
    `ProjectionLayer.MODE` → PROJECTION, `BeamLayer.MODE` → BEAM; OFF → None (debug disarmed,
    the machine owns the motor). Selecting a layer is the only gesture: choosing it IS turning
    debug on, and the speed rides the selection."""
    if selection == DebugLayer.OFF:
        return None
    layer = layers.get(LayerId(int(selection)))
    if layer is None:
        return None
    return layer.MODE


class Conductor(Thread):
    """Runs the light loop at a fixed rate (light_rate Hz); see the module docstring."""

    def __init__(self, settings: LightSettings, instrument: PoseInstrumentSettings, board: RenderBoard, pose_stage: int) -> None:
        super().__init__(daemon=True, name="LightConductor")

        self._stop_event = Event()

        self._settings: LightSettings = settings
        self._board: RenderBoard      = board
        self._motor_controller      = MotorController(settings.motor)
        self._playhead              = Playhead(settings.playhead)
        self._clock                 = Clock(settings.clock)

        resolution: int             = settings.light_resolution
        max_players: int            = settings.max_players

        # The unified layer pool — one instance per LayerId, each reading its own settings
        # group (beam_layers / projection_layers, mirroring the folder taxonomy); each layer's
        # mode lives in its class (BeamLayer/ProjectionLayer).
        LO, HI = settings.beam_layers, settings.projection_layers
        self.layers: dict[LayerId, BaseLayer] = {
            LayerId.beam_blue_sound:     BeamBlueSound      (resolution, LO.beam_blue_sound,     board),
            LayerId.beam_playhead:       BeamPlayhead       (resolution, LO.beam_playhead,       board),
            LayerId.beam_flash:          BeamFlash          (resolution, LO.beam_flash,          board, pose_stage),
            LayerId.beam_wind_down:      BeamWindDown       (resolution, LO.beam_wind_down,      board),
            LayerId.beam_sync_wind_down: BeamWindDown       (resolution, LO.beam_sync_wind_down, board, strobe=instrument.sync_strobe),
            LayerId.beam_test:           BeamTest           (resolution, LO.beam_test,           board),
            LayerId.pose_instrument:     PoseInstrument     (resolution, HI.pose_instrument,     instrument, board, pose_stage),
            LayerId.projection_playhead: ProjectionPlayhead (resolution, HI.projection_playhead, instrument.playhead, instrument.mask, board, pose_stage),
            LayerId.flood:               Flood              (resolution, HI.flood,               board),
            LayerId.test_player_lines:   TestPlayerLines    (resolution, HI.test_player_lines,   board, pose_stage),
            LayerId.test_calibration:    TestCalibration    (resolution, HI.test_calibration,    settings.num_cams, board),
            LayerId.test_fill:           TestFill           (resolution, HI.test_fill,           board),
            LayerId.test_pulse:          TestPulse          (resolution, HI.test_pulse,          board),
            LayerId.test_chase:          TestChase          (resolution, HI.test_chase,          board),
            LayerId.test_lines:          TestLines          (resolution, HI.test_lines,          board),
            LayerId.test_random:         TestRandom         (resolution, HI.test_random,         board),
            LayerId.test_pose_waves:     TestPoseWaves      (resolution, max_players, HI.test_pose_waves, board, pose_stage),
            LayerId.test_harmonic:       TestHarmonic       (resolution, HI.test_harmonic,       board),
        }

        self._compositor = Compositor(settings, self.layers)
        # Nothing in the preset is forced at start; the one saved state that spins the fixture up
        # at power-on is worth a line in the log.
        if not settings.motor.simulate and _debug_motor_mode(settings.debug, self.layers) is MotorMode.PROJECTION:
            logger.warning("Starting with debug layer %s selected: the fixture spins up to PROJECTION at once",
                           DebugLayer(int(settings.debug)).name)

        self.fps_counter = FpsCounter()
        self._update_callbacks: list[Callable[[Tick], Any]] = []
        self._light_frame_callbacks: list[LightFrameCallback] = []


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
        # No thread priority boost: inside the process no thread outranks another — the GIL hands
        # over on its own schedule regardless of OS priority, so a boost only preempts the pose,
        # video and render threads without making this one's Python run sooner (profiled
        # 2026-10-08: an elevated conductor caused playback stutter). Timing is protected by
        # keeping the tick's work short; the clock's late_max/busy_max/overruns are the watchdog.
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
            _debug_motor_mode(self._settings.debug, self.layers))

        # Measure the motor, advance the playhead under the command in force over this dt, and
        # publish the playhead BEFORE the update callbacks: the state machine reads it this tick
        # (no added latency). Phase is NaN while the motor is STOPPED (no meaningful playhead);
        # the bar counter stays monotonic throughout.
        motor   = self._motor_controller.tick()
        command = self._motor_controller.command
        self._playhead.tick(tick.dt, motor, command)
        playhead = self._playhead.phase
        self._board.set_playhead(PlayheadSignals(
            phase=playhead, bars=self._playhead.bars, is_locked=self._playhead.is_locked,
            is_projecting=self._playhead.is_projecting))

        self._notify_update(tick)

        # The state machine commands the motor from inside those callbacks, and on a state's
        # entry tick it also hands the Compositor that state's mix — so the frame takes the
        # command as it stands now: the rpm it is sent with is the rpm its content is drawn for.
        command = self._motor_controller.command

        frame = LightFrame(self._settings.light_resolution, tick, motor, command, playhead=playhead)
        self._compositor.render(frame)

        # Main brightness — the projection and the beam lights alike
        m = self._settings.brightness
        if m != 1.0:
            frame.white      *= m
            frame.blue       *= m
            frame.beam_lights *= m

        self._board.set_light_frame(frame)
        self._notify_light_frame(frame)
        self.fps_counter.tick()

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def add_update_callback(self, callback: Callable[[Tick], Any]) -> None:
        """Register a callback fired with the tick at the start of each one, before rendering.
        Use for time-driven state advances (the pose pipeline's output domain, the state machine).
        """
        self._update_callbacks.append(callback)

    def add_light_frame_callback(self, callback: LightFrameCallback) -> None:
        """Register a callback fired after each tick with the new LightFrame.
        Use for output consumers (hardware sender, audio).
        """
        self._light_frame_callbacks.append(callback)

    def _notify_update(self, tick: Tick) -> None:
        for cb in self._update_callbacks:
            try:
                cb(tick)
            except Exception:
                logger.exception("Error in light conductor update callback")

    def _notify_light_frame(self, frame: LightFrame) -> None:
        for cb in self._light_frame_callbacks:
            try:
                cb(frame)
            except Exception:
                logger.exception("Error in light conductor light frame callback")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def fps(self) -> float:
        return self.fps_counter.get_fps()
