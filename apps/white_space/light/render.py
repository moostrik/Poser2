"""Render — threaded LED light renderer running at a fixed rate.

Drives the per-tick loop: advances the clock and motor, renders the active layers (each pulls the
board slices it needs and blends itself into the per-tick Frame), applies master post-processing,
and forwards the Frame to the board and to registered output callbacks (UDP sender, audio, render).
"""

from threading import Event, Thread
from typing import Any, Callable

import numpy as np

from modules.utils import HotReloadMethods
from modules.gl import FpsCounter
from modules.tracker.panoramic.settings import DistortionSettings

from .clock import Clock, Tick
from .frame import Frame, FrameCallback
from .motor import MotorController
from .sampler import Sampler
from .settings import LightSettings, LayerId
from .layers import BaseLayer, PoseWaves, Fill, Pulse, Chase, Lines, Random, Harmonic, PlayerLines, Calibration, PlayheadFlash
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
        self._clock                 = Clock(config)
        self._sampler               = Sampler(board, pose_stage)

        resolution: int             = config.light_resolution
        num_players: int            = config.max_poses

        # Fixed layer registry — ordered list of (id, instance) pairs (order = blend order).
        # Every layer receives the board and pulls the slices it needs in _draw.
        self._layers: list[tuple[LayerId, BaseLayer]] = [
            (LayerId.pose_waves,     PoseWaves   (resolution, num_players, config.pose_waves, self._clock.interval, board, pose_stage)),
            (LayerId.fill,           Fill        (resolution, config.fill,         board)),
            (LayerId.pulse,          Pulse       (resolution, config.pulse,        board)),
            (LayerId.chase,          Chase       (resolution, config.chase,        board)),
            (LayerId.lines,          Lines       (resolution, config.lines,        board)),
            (LayerId.random,         Random      (resolution, config.random,       board)),
            (LayerId.harmonic,       Harmonic    (resolution, config.harmonic,     board)),
            (LayerId.player_lines,   PlayerLines (resolution, config.player_lines, board, pose_stage)),
            (LayerId.playhead_flash, PlayheadFlash(resolution, config.playhead_flash, board)),
            (LayerId.calibration,    Calibration (resolution, config.calibration, distortion, config.num_cameras, board)),
        ]

        self.fps_counter = FpsCounter()
        self._active_prev: set[LayerId] = set()
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
        # Pre-render state callbacks (sequencer, interpolators)
        for cb in self._update_callbacks:
            try:
                cb()
            except Exception:
                logger.exception("Error in light render update callback")

        active = set(self._config.active)

        # Reset layers that just became inactive so they start fresh on reactivation
        for layer_id, layer in self._layers:
            if layer_id not in active and layer_id in self._active_prev:
                layer.reset()
        self._active_prev = active

        # Motor target: active layers override the manual config value.
        # None means "no opinion"; 0.0 is an explicit stop command.
        active_rpm: float = self._config.target_rpm
        for layer_id, layer in self._layers:
            if layer_id in active and layer.target_rpm is not None:
                active_rpm = layer.target_rpm

        motor = self._motor_controller.tick(active_rpm)

        hits  = self._sampler.detect(motor.playhead, motor.mode)

        frame = Frame(self._config.light_resolution, tick, motor, hits=hits)

        for layer_id, layer in self._layers:
            if layer_id not in active:
                continue
            try:
                layer.render(frame)
            except Exception:
                logger.exception("Error in %s.render", layer.__class__.__name__)

        # Master output: contrast (hardness) then brightness (master)
        frame.white = self._master_process(frame.white)
        frame.blue  = self._master_process(frame.blue)

        self._board.set_composition_output(frame)
        self._notify_render(frame)
        self.fps_counter.tick()

    def _master_process(self, arr: np.ndarray) -> np.ndarray:
        """Apply contrast hardness then master brightness to a mix buffer."""
        cfg = self._config
        x = np.clip(arr, 0.0, 1.0)
        h = cfg.hardness
        if h > 0.0:
            k = 20.0 * h
            sigmoid = 1.0 / (1.0 + np.exp(-k * (x - cfg.threshold)))
            x = (1.0 - h) * x + h * sigmoid
        m = cfg.master
        if m != 1.0:
            x = x * m
        return x

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
        for _, layer in self._layers:
            layer.reset()
