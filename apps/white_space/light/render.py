"""Render — threaded LED light renderer running at a fixed rate.

Drives the per-tick loop: advances the clock and motor, renders the active layers (each pulls the
board slices it needs and blends itself into the per-tick Frame), applies master post-processing,
and forwards the Frame to the board and to registered output callbacks (UDP sender, audio, render).
"""

from threading import Event, Thread
from typing import Any, Callable

from modules.utils import HotReloadMethods
from modules.gl import FpsCounter
from modules.tracker.panoramic.settings import DistortionSettings

from .clock import Clock, Tick
from .frame import Frame, FrameCallback
from .motor import MotorController
from .playhead import Playhead
from .settings import LightSettings, LowLayerId, HighLayerId
from .layers import BaseLayer, Crossfade, PoseWaves, Fill, Pulse, Chase, Lines, Random, Harmonic, PlayerLines, CameraLight, PlayheadFlash, PlayheadLow, PlayheadHigh
from ..board import Board

import logging
logger = logging.getLogger(__name__)


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

        # Layers scoped per slot, constructed explicitly (like the GPU renderer's self.L): idle/low
        # select from low_layers, high selects from high_layers. The test layers are shared by both
        # slots as a per-scope instance reading the same config.test.* settings.
        low, high, test = config.low, config.high, config.test
        self.low_layers: dict[LowLayerId, BaseLayer] = {
            LowLayerId.playhead:       PlayheadLow  (resolution, low.playhead,       board),
            LowLayerId.playhead_flash: PlayheadFlash(resolution, low.playhead_flash, board, pose_stage),
            LowLayerId.fill:   Fill  (resolution, test.fill,   board),
            LowLayerId.pulse:  Pulse (resolution, test.pulse,  board),
            LowLayerId.chase:  Chase (resolution, test.chase,  board),
            LowLayerId.lines:  Lines (resolution, test.lines,  board),
            LowLayerId.random: Random(resolution, test.random, board),
        }
        self.high_layers: dict[HighLayerId, BaseLayer] = {
            HighLayerId.pose_waves:   PoseWaves   (resolution, num_players, high.pose_waves, self._clock.interval, board, pose_stage),
            HighLayerId.harmonic:     Harmonic    (resolution, high.harmonic,     board),
            HighLayerId.player_lines: PlayerLines (resolution, high.player_lines, board, pose_stage),
            HighLayerId.calibration:  CameraLight (resolution, high.calibration, distortion, config.num_cameras, board),
            HighLayerId.playhead:     PlayheadHigh (resolution, high.playhead,    board),
            HighLayerId.fill:   Fill  (resolution, test.fill,   board),
            HighLayerId.pulse:  Pulse (resolution, test.pulse,  board),
            HighLayerId.chase:  Chase (resolution, test.chase,  board),
            HighLayerId.lines:  Lines (resolution, test.lines,  board),
            HighLayerId.random: Random(resolution, test.random, board),
        }

        self._crossfade = Crossfade(config, self.low_layers, self.high_layers, board)

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
        # Advance motor + playhead and publish the playhead BEFORE the update callbacks, whose
        # pose-LERP reads it. Phase is NaN while the motor is STOPPED (no meaningful playhead).
        motor = self._motor_controller.tick()
        self._playhead.tick(tick.dt, motor)
        playhead = self._playhead.phase
        self._board.set_playhead(playhead)

        self._notify_update()

        frame = Frame(self._config.light_resolution, tick, motor, playhead=playhead)
        self._crossfade.render(frame)

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
        Use for time-driven state advances (sequencer, interpolators).
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
                logger.exception("Error in light render update callback")

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
        self._crossfade.reset()
