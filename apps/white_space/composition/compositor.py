"""Compositor - threaded LED composition host running at a fixed rate."""

from dataclasses import replace
from threading import Event, Thread, Lock
from time import time, sleep, monotonic
from typing import Any, Callable

import numpy as np

from modules.utils import HotReloadMethods
from modules.gl import FpsCounter
from modules.tracker import Tracklet
from modules.tracker.panoramic.settings import DistortionSettings
from modules.pose.frame import Frame, FrameDict
from modules.oak import FrameType

from .transport import Clock, Transport
from .base import Composition
from ..board import Board
from .output import CompositionOutput, BUFFER_DTYPE, CompositionOutputCallback
from .motor import Motor, MotorMode
from .settings import CompositorSettings, CompositionId
from .comps import PoseWaves, Fill, Pulse, Chase, Lines, Random, Harmonic, PlayerLines, Calibration, PlayheadFlash
from .playhead_hit import PlayheadHit, detect_hits
from .draw import blend_values

import logging
logger = logging.getLogger(__name__)


class Compositor(Thread):
    """Runs the LED composition loop at a fixed rate (light_rate Hz) and forwards
    the result to registered output callbacks (UDP sender, render board, etc.).

    Input:  pose frames (latest snapshot per player) + tracklets (locked snapshot)
    Output: CompositionOutput via registered callbacks
    """

    def __init__(self, config: CompositorSettings, distortion: DistortionSettings, board: Board) -> None:
        super().__init__(daemon=True, name="Compositor")

        self._stop_event    = Event()
        self._tracklet_lock = Lock()
        self._frames_lock   = Lock()

        self._latest_tracklets: dict[int, Tracklet] = {}
        self._latest_frames:    dict[int, Frame]    = {}

        self._config          = config
        self._motor           = Motor(config.motor)
        self.interval: float  = 1.0 / config.light_rate
        resolution: int      = config.light_resolution
        num_players: int     = config.max_poses

        # Host-owned mix buffers — zeroed before each composition round
        self._white: np.ndarray = np.zeros(resolution, dtype=BUFFER_DTYPE)
        self._blue:  np.ndarray = np.zeros(resolution, dtype=BUFFER_DTYPE)
        # Per-comp temp buffers — reused each render call
        self._temp_white: np.ndarray = np.zeros(resolution, dtype=BUFFER_DTYPE)
        self._temp_blue:  np.ndarray = np.zeros(resolution, dtype=BUFFER_DTYPE)
        # Master transport clock — reads bpm, writes time/phase back as READ fields
        self._transport = Clock(config)

        # Fixed composition registry — ordered list of (id, instance) pairs
        self._pose_waves  = PoseWaves   (resolution, num_players, config.pose_waves, self.interval)
        self._calibration = Calibration (resolution, config.calibration, distortion, config.num_cameras)
        self._compositions: list[tuple[CompositionId, Composition]] = [
            (CompositionId.pose_waves,   self._pose_waves),
            (CompositionId.fill,         Fill        (resolution, config.fill)),
            (CompositionId.pulse,        Pulse       (resolution, config.pulse)),
            (CompositionId.chase,        Chase       (resolution, config.chase)),
            (CompositionId.lines,        Lines       (resolution, config.lines)),
            (CompositionId.random,       Random      (resolution, config.random)),
            (CompositionId.harmonic,     Harmonic    (resolution, config.harmonic)),
            (CompositionId.player_lines,  PlayerLines  (resolution, config.player_lines)),
            (CompositionId.playhead_flash, PlayheadFlash(resolution, config.playhead_flash)),
            (CompositionId.calibration,   self._calibration),
        ]

        self.fps_counter = FpsCounter()
        self.board = board
        self._update_callbacks: list[Callable[[], Any]] = []
        self._render_callbacks: list[CompositionOutputCallback] = []

        self._prev_playhead:  float = 0.0
        self._last_tick_mono: float = float('-inf')

        self.hot_reloader = HotReloadMethods(self.__class__, True)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._motor.start()
        super().start()

    def stop(self) -> None:
        self._motor.stop()
        self._stop_event.set()
        if self.is_alive():
            self.join()

    def notify_fall(self) -> None:
        """Signal a revolution fall edge; forwarded to the motor."""
        self._motor.notify_fall()

    def run(self) -> None:
        next_time: float = time()
        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception:
                logger.exception("Error in Compositor tick")
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
        for cb in self._update_callbacks:
            try:
                cb()
            except Exception:
                logger.exception("Error in Compositor update callback")

        frames = self._consume_frames()
        with self._tracklet_lock:
            tracklets = dict(self._latest_tracklets)

        transport = self._transport.tick()
        playhead, motor_mode = self._motor.tick()

        now_mono = monotonic()
        gap_ok = (now_mono - self._last_tick_mono) < 0.5
        hits: tuple[PlayheadHit, ...] = ()
        if gap_ok and motor_mode == MotorMode.LOW_SPEED:
            hits = detect_hits(frames, tracklets, self._prev_playhead, playhead)
        self._prev_playhead  = playhead
        self._last_tick_mono = now_mono

        transport = replace(transport, playhead=playhead, motor_mode=motor_mode, hits=hits)

        # Forward latest pose data to compositions that need it
        for comp_id, comp in self._compositions:
            comp.set_pose_inputs(frames, tracklets)

        # Zero shared mix buffers
        self._white.fill(0.0)
        self._blue.fill(0.0)

        active = set(self._config.active)
        blend  = self._config.blend

        # Render each active composition into a temp buffer, then blend into master
        for comp_id, comp in self._compositions:
            if comp_id not in active:
                continue
            self._temp_white.fill(0.0)
            self._temp_blue.fill(0.0)
            try:
                comp.render(transport, self._temp_white, self._temp_blue)
            except Exception:
                logger.exception("Error in %s.render", comp.__class__.__name__)
                continue
            blend_values(self._white, self._temp_white, 0, blend)
            blend_values(self._blue,  self._temp_blue,  0, blend)

        # Master output: contrast (hardness) then brightness (master)
        white = self._master_process(self._white)
        blue  = self._master_process(self._blue)

        # Collect motor speed target: active compositions override the manual config value.
        # None means "no opinion"; 0.0 is an explicit stop command.
        active_rpm: float = self._config.target_rpm
        for comp_id, comp in self._compositions:
            if comp_id in active and comp.target_rpm is not None:
                active_rpm = comp.target_rpm

        # Build output, update board, dispatch to render callbacks
        output = CompositionOutput(self._config.light_resolution)
        output.white = white
        output.blue  = blue
        output.target_rpm = active_rpm
        output.playhead = playhead
        output.hits = hits
        self.board.set_composition_output(output)
        self._notify_render(output)
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
    # Inputs
    # ------------------------------------------------------------------

    def set_image(self, cam_id: int, frame_type: FrameType, frame: np.ndarray) -> None:
        """Camera frame callback — routes VIDEO frames to the calibration composition."""
        if frame_type == FrameType.VIDEO:
            self._calibration.set_camera_image(cam_id, frame)

    def add_poses(self, frames: FrameDict) -> None:
        """Store the latest frame per player; called from the pose pipeline thread."""
        with self._frames_lock:
            for track_id, frame in frames.items():
                self._latest_frames[track_id] = frame

    def set_tracklets(self, tracklets: dict[int, Tracklet]) -> None:
        with self._tracklet_lock:
            self._latest_tracklets = dict(tracklets)

    def _consume_frames(self) -> list[Frame]:
        """Consume the latest frames for this tick and clear the buffer."""
        with self._frames_lock:
            frames = list(self._latest_frames.values())
            self._latest_frames.clear()
        return frames

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def add_update_callback(self, callback: Callable[[], Any]) -> None:
        """Register a callback fired at the start of each compositor tick, before composition.
        Use for time-driven state advances (sequencer, interpolators).
        """
        self._update_callbacks.append(callback)

    def add_render_callback(self, callback: CompositionOutputCallback) -> None:
        """Register a callback fired after each compositor tick with the new CompositionOutput.
        Use for output consumers (hardware sender, audio).
        """
        self._render_callbacks.append(callback)

    def _notify_render(self, output: CompositionOutput) -> None:
        for cb in self._render_callbacks:
            try:
                cb(output)
            except Exception:
                logger.exception("Error in Compositor render callback")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def fps(self) -> float:
        return self.fps_counter.get_fps()

    def reset(self) -> None:
        for _, comp in self._compositions:
            comp.reset()
