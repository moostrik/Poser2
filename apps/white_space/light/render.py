"""LightRenderer — threaded LED light renderer running at a fixed rate.

Reads all data inputs from the board (pose frames, tracklets, camera VIDEO frames),
renders the active layers into a per-tick Frame, and forwards the result to the board
and to registered output callbacks (UDP sender, audio, render).
"""

from threading import Event, Thread
from time import time, sleep
from typing import Any, Callable

import numpy as np

from modules.utils import HotReloadMethods
from modules.gl import FpsCounter
from modules.tracker.panoramic.settings import DistortionSettings

from .clock import Clock
from .frame import Frame, FrameCallback
from .motor import MotorController
from .sampler import Sampler
from .settings import LightRendererSettings, LayerId
from .layers import BaseLayer, PoseWaves, Fill, Pulse, Chase, Lines, Random, Harmonic, PlayerLines, Calibration, PlayheadFlash
from ..board import Board

import logging
logger = logging.getLogger(__name__)


class Render(Thread):
    """Runs the LED render loop at a fixed rate (light_rate Hz) and forwards the
    result to registered output callbacks (UDP sender, render board, audio).

    Input:  pose frames + tracklets + camera VIDEO frames, all read from the board.
    Output: Frame via the board and registered render callbacks.
    """

    def __init__(self, config: LightRendererSettings, distortion: DistortionSettings, board: Board, pose_stage: int) -> None:
        super().__init__(daemon=True, name="LightRenderer")

        self._stop_event = Event()

        self._config           = config
        self._board            = board
        self._pose_stage       = pose_stage
        self._motor_controller = MotorController(config.motor)
        self._clock            = Clock(config)
        self._sampler          = Sampler()
        self.interval: float   = 1.0 / config.light_rate
        resolution: int        = config.light_resolution
        num_players: int       = config.max_poses

        # Fixed layer registry — ordered list of (id, instance) pairs
        self._pose_waves  = PoseWaves   (resolution, num_players, config.pose_waves, self.interval)
        self._calibration = Calibration (resolution, config.calibration, distortion, config.num_cameras)
        self._layers: list[tuple[LayerId, BaseLayer]] = [
            (LayerId.pose_waves,     self._pose_waves),
            (LayerId.fill,           Fill        (resolution, config.fill)),
            (LayerId.pulse,          Pulse       (resolution, config.pulse)),
            (LayerId.chase,          Chase       (resolution, config.chase)),
            (LayerId.lines,          Lines       (resolution, config.lines)),
            (LayerId.random,         Random      (resolution, config.random)),
            (LayerId.harmonic,       Harmonic    (resolution, config.harmonic)),
            (LayerId.player_lines,   PlayerLines (resolution, config.player_lines)),
            (LayerId.playhead_flash, PlayheadFlash(resolution, config.playhead_flash)),
            (LayerId.calibration,    self._calibration),
        ]

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

    def notify_fall(self) -> None:
        """Signal a revolution fall edge; forwarded to the motor controller."""
        self._motor_controller.notify_fall()

    def run(self) -> None:
        next_time: float = time()
        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception:
                logger.exception("Error in LightRenderer tick")
            next_time += self.interval
            sleep_time: float = next_time - time()
            if sleep_time > 0:
                sleep(sleep_time)
            else:
                next_time = time()

    # ------------------------------------------------------------------
    # Per-tick
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        # Pre-render state callbacks (sequencer, interpolators)
        for cb in self._update_callbacks:
            try:
                cb()
            except Exception:
                logger.exception("Error in LightRenderer update callback")

        # All data inputs from the board
        frames    = list(self._board.get_frames(self._pose_stage).values())
        tracklets = self._board.get_tracklets()
        for cam_id in range(self._config.num_cameras):
            img = self._board.get_video_image(cam_id)
            if img is not None:
                self._calibration.set_camera_image(cam_id, img)

        tick = self._clock.tick()

        # Motor target: active layers override the manual config value.
        # None means "no opinion"; 0.0 is an explicit stop command.
        active = set(self._config.active)
        active_rpm: float = self._config.target_rpm
        for layer_id, layer in self._layers:
            if layer_id in active and layer.target_rpm is not None:
                active_rpm = layer.target_rpm

        motor = self._motor_controller.tick(active_rpm)
        hits  = self._sampler.detect(frames, tracklets, motor.playhead, motor.mode)

        frame = Frame(self._config.light_resolution, tick, motor, hits=hits)

        # Forward latest pose data to layers that need it
        for _, layer in self._layers:
            layer.set_pose_inputs(frames, tracklets)

        # Render each active layer; the layer blends itself into the frame
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
                logger.exception("Error in LightRenderer render callback")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def fps(self) -> float:
        return self.fps_counter.get_fps()

    def reset(self) -> None:
        for _, layer in self._layers:
            layer.reset()
