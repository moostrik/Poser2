"""Compositor - threaded LED composition host running at a fixed rate."""

from threading import Event, Thread, Lock
from time import time, sleep
from typing import Callable

import numpy as np

from modules.utils.HotReloadMethods import HotReloadMethods
from modules.gl.Utils import FpsCounter
from modules.tracker import Tracklet
from modules.pose.frame import Frame, FrameDict

from .transport import TransportClock
from .base import Composition
from .output import CompositionOutput, COMP_DTYPE, CompositionOutputCallback
from .settings import CompositorSettings
from .comps import PoseWaves, Fill, Pulse, Chase, Lines, Random, Harmonic

import logging
logger = logging.getLogger(__name__)


class Compositor(Thread):
    """Runs the LED composition loop at a fixed rate (light_rate Hz) and forwards
    the result to registered output callbacks (UDP sender, render board, etc.).

    Input:  pose frames (latest snapshot per player) + tracklets (locked snapshot)
    Output: CompositionOutput via registered callbacks
    """

    def __init__(self, config: CompositorSettings) -> None:
        super().__init__(daemon=True, name="Compositor")

        self._stop_event    = Event()
        self._tracklet_lock = Lock()
        self._frames_lock   = Lock()

        self._latest_tracklets: dict[int, Tracklet] = {}
        self._latest_frames:    dict[int, Frame]    = {}

        self._config   = config
        self.interval: float = 1.0 / config.light_rate
        resolution: int      = config.light_resolution
        num_players: int     = config.max_poses

        # Host-owned mix buffers â€” zeroed before each composition round
        self._white: np.ndarray = np.zeros(resolution, dtype=COMP_DTYPE)
        self._blue:  np.ndarray = np.zeros(resolution, dtype=COMP_DTYPE)

        # Master transport clock
        self._transport = TransportClock(config.transport)

        # Fixed composition registry
        self._pose_waves = PoseWaves(resolution, num_players, config.pose_waves, self.interval)
        self._compositions: list[Composition] = [
            self._pose_waves,
            Fill     (resolution, config.fill),
            Pulse    (resolution, config.pulse),
            Chase    (resolution, config.chase),
            Lines    (resolution, config.lines),
            Random   (resolution, config.random),
            Harmonic (resolution, config.harmonic),
        ]

        self.fps_counter = FpsCounter()
        self._output_callbacks: list[CompositionOutputCallback] = []

        self.hot_reloader = HotReloadMethods(self.__class__, True)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        super().start()

    def stop(self) -> None:
        self._stop_event.set()
        if self.is_alive():
            self.join()

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
        frames = self._snapshot_frames()
        with self._tracklet_lock:
            tracklets = dict(self._latest_tracklets)

        transport = self._transport.tick()

        # Forward latest pose data to compositions that need it
        for comp in self._compositions:
            comp.set_pose_inputs(frames, tracklets)

        # Zero shared mix buffers
        self._white.fill(0.0)
        self._blue.fill(0.0)

        # Render all enabled compositions additively
        for comp in self._compositions:
            try:
                comp.render(transport, self._white, self._blue)
            except Exception:
                logger.exception("Error in %s.render", comp.__class__.__name__)

        # Clamp to valid output range
        np.clip(self._white, 0.0, 1.0, out=self._white)
        np.clip(self._blue,  0.0, 1.0, out=self._blue)

        # Build output and dispatch
        output = CompositionOutput(self._config.light_resolution)
        output.light_0 = self._white
        output.light_1 = self._blue
        self._notify_output(output)
        self.fps_counter.tick()

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------

    def add_poses(self, frames: FrameDict) -> None:
        """Store the latest frame per player; called from the pose pipeline thread."""
        with self._frames_lock:
            for track_id, frame in frames.items():
                self._latest_frames[track_id] = frame

    def set_tracklets(self, tracklets: dict[int, Tracklet]) -> None:
        with self._tracklet_lock:
            self._latest_tracklets = dict(tracklets)

    def _snapshot_frames(self) -> list[Frame]:
        """Consume the latest frames for this tick and clear the buffer."""
        with self._frames_lock:
            frames = list(self._latest_frames.values())
            self._latest_frames.clear()
        return frames

    # ------------------------------------------------------------------
    # Output callbacks
    # ------------------------------------------------------------------

    def add_output_callback(self, callback: CompositionOutputCallback) -> None:
        self._output_callbacks.append(callback)

    def _notify_output(self, output: CompositionOutput) -> None:
        for cb in self._output_callbacks:
            try:
                cb(output)
            except Exception:
                logger.exception("Error in Compositor output callback")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def fps(self) -> float:
        return self.fps_counter.get_fps()

    def reset(self) -> None:
        for comp in self._compositions:
            comp.reset()
