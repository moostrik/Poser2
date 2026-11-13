"""The cameras' sync: one callback each time every camera's newest frame (or tracklet message) is in.

The submits only queue a message's camera and time, on the threads they come on, at most `WAITING` per camera;
beyond that the oldest is dropped, counted in `dropped`. The sync's own thread decides the syncs in the order the
messages came and calls the callbacks outside its lock. While the thread does not run, a message is decided on the
thread it comes on."""

import time
from collections import deque
from threading import Condition, Lock, Thread
from typing import Any, Callable

from modules.camera import CameraFrame, CameraTracklets, FrameType
from modules.settings import BaseSettings, Field
from modules.utils import PerformanceTimer

import logging
logger = logging.getLogger(__name__)

WAITING: int = 16       # messages per camera waiting for the sync's thread


class CameraSyncSettings(BaseSettings):
    num_cams:    Field[int]   = Field(1, access=Field.INIT, visible=False)
    fps:            Field[float] = Field(30.0, access=Field.INIT, visible=False)


class CameraSync(Thread):
    """Syncs the cameras: when the camera with the largest gap to the next frame submits, every camera's newest
    frame is as close together as it gets, and the sync fires with that moment. Frames and tracklet messages are
    two separate instances with the same rule (`frame_sync`, `tracklet_sync`); see the module docstring for the
    thread."""

    def __init__(self, settings: CameraSyncSettings, verbose: bool = False, stream_name: str = '') -> None:
        super().__init__(daemon=True, name=f'CameraSync {stream_name}'.strip())
        num_cams: int = settings.num_cams
        self.verbose: bool = verbose
        self.stream_name: str = stream_name
        self.max_gap_s: float = 1.0 / settings.fps
        self.min_sync_interval_s: float = 0.75 / settings.fps
        self.last_sync_time_s: float = 0.0

        self._timestamp_history: deque[tuple[int, float]] = deque(maxlen=10 * num_cams)
        self._callback_lock: Lock = Lock()
        self._callbacks: list[Callable[[float], Any]] = []  # run in registration order, with the sync's time
        self._wake: Condition = Condition()
        self._running: bool = False
        self._inbox: deque[tuple[int, float, float]] = deque()     # (cam_id, timestamp, the moment it was queued)
        self._queued: dict[int, int] = {}                           # what the inbox holds per camera
        self.dropped: int = 0
        self._inline_lock: Lock = Lock()            # messages decided on the callers' threads while the thread does not run
        self._wait: PerformanceTimer = PerformanceTimer(name=f"Wait camera sync {stream_name}".strip())

    # ── Lifecycle ─────────────────────────────────────────────────────

    def start(self) -> None:
        with self._wake:
            self._running = True
        super().start()

    def stop(self) -> None:
        """The thread ends; the messages still waiting are dropped and logged."""
        with self._wake:
            self._running = False
            dropped: int = len(self._inbox)
            self._inbox.clear()
            self._queued.clear()
            self._wake.notify()
        if dropped:
            logger.info("%s: %d messages dropped at stop", self.name, dropped)
        if self.is_alive():
            self.join(timeout=2.0)

    def run(self) -> None:
        while True:
            with self._wake:
                while self._running and not self._inbox:
                    self._wake.wait()
                if not self._running:
                    return
                cam_id, timestamp, queued = self._inbox.popleft()
                self._queued[cam_id] -= 1
            self._wait.add_time((time.perf_counter() - queued) * 1000.0, report=False)
            try:
                self._decide(cam_id, timestamp)
            except Exception:
                logger.exception("%s: a sync failed", self.name)

    # ── Inputs: each queues and returns ───────────────────────────────

    def submit_frame(self, frame: CameraFrame) -> None:
        """A camera's frame callback. Only VIDEO frames count: a sync is one frame per camera."""
        if frame.kind == FrameType.VIDEO:
            self._submit(frame.cam_id, frame.timestamp)

    def submit_tracklets(self, cam_tracklets: CameraTracklets) -> None:
        self._submit(cam_tracklets.cam_id, cam_tracklets.timestamp)

    def _submit(self, cam_id: int, timestamp: float) -> None:
        """A message with its host time, queued for the thread, the camera's oldest dropped beyond `WAITING`."""
        with self._wake:
            if self._running:
                if self._queued.get(cam_id, 0) >= WAITING:
                    for index, (queued_cam, _, _) in enumerate(self._inbox):
                        if queued_cam == cam_id:
                            del self._inbox[index]
                            self.dropped += 1
                            break
                else:
                    self._queued[cam_id] = self._queued.get(cam_id, 0) + 1
                self._inbox.append((cam_id, timestamp, time.perf_counter()))
                self._wake.notify()
                return
        with self._inline_lock:
            self._decide(cam_id, timestamp)

    def _decide(self, cam_id: int, timestamp: float) -> None:
        """Whether the message completes a sync; the sync carries the time of the message that completed it."""
        self._timestamp_history.append((cam_id, timestamp))
        sync_cam_id: int = CameraSync._find_sync_camera(self._timestamp_history, self.max_gap_s)

        if cam_id == sync_cam_id:
            time_since_last_sync: float = timestamp - self.last_sync_time_s
            self.last_sync_time_s = timestamp

            if time_since_last_sync >= self.min_sync_interval_s:
                self._notify_callbacks(timestamp)
            elif self.verbose:
                logger.info(f"{self.stream_name}Skipped sync (only {time_since_last_sync:.3f}s since last, from cam {cam_id})")
    @staticmethod
    def _find_sync_camera(timestamp_history: deque[tuple[int, float]], max_gap_s: float) -> int:
        """The camera whose frame completes a sync: the one with the largest average gap to the next frame, gaps
        over `max_gap_s` (a dropped frame) left out."""
        camera_ids: set[int] = CameraSync._get_unique_cameras_in_history(timestamp_history)

        if len(timestamp_history) < 2:
            return next(iter(camera_ids))

        # Calculate average gap to next frame for each camera
        cam_gaps: dict[int, list[float]] = {cam_id: [] for cam_id in camera_ids}

        # Look through history and find gap from each cam_id to next timestamp
        for i in range(len(timestamp_history) - 1):
            cam_id, timestamp = timestamp_history[i]
            next_timestamp: float = timestamp_history[i + 1][1]
            gap: float = next_timestamp - timestamp

            # Only include gaps within normal range
            if gap <= max_gap_s:
                cam_gaps[cam_id].append(gap)

        # Find camera with largest average gap
        max_avg_gap = 0.0
        sync_cam: int = next(iter(camera_ids))
        has_valid_gaps = False

        for cam_id, gaps in cam_gaps.items():
            if gaps:
                has_valid_gaps = True
                avg_gap: float = sum(gaps) / len(gaps)
                if avg_gap > max_avg_gap:
                    max_avg_gap: float = avg_gap
                    sync_cam = cam_id

        # Fallback: if all gaps filtered, use any camera
        if not has_valid_gaps:
            sync_cam = next(iter(camera_ids))

        return sync_cam

    @staticmethod
    def _get_max_time_diff(timestamp_history: deque[tuple[int, float]]) -> float:
        """
        Find the maximum time difference between the latest frames of all cameras.

        Returns:
            Maximum time difference in seconds, or 0.0 if not all cameras have frames
        """
        if len(timestamp_history) == 0:
            return 0.0

        camera_ids: set[int] = CameraSync._get_unique_cameras_in_history(timestamp_history)

        if not camera_ids:  # Extra safety
            return 0.0

        # Find the latest timestamp for each camera
        latest_timestamps: dict[int, float] = {}

        for cam_id, timestamp in reversed(timestamp_history):
            if cam_id not in latest_timestamps:
                latest_timestamps[cam_id] = timestamp
            if len(latest_timestamps) == len(camera_ids):
                break

        # Need at least 2 cameras for meaningful diff
        if len(latest_timestamps) < 2:
            return 0.0

        timestamps = list(latest_timestamps.values())
        return max(timestamps) - min(timestamps)

    @staticmethod
    def _get_unique_cameras_in_history(timestamp_history: deque[tuple[int, float]]) -> set[int]:
        cameras: set[int] = set()
        for cam_id, _ in timestamp_history:
            cameras.add(cam_id)
        return cameras

    def add_sync_callback(self, callback: Callable[[float], Any]) -> None:
        with self._callback_lock:
            if callback not in self._callbacks:
                self._callbacks.append(callback)

    def _notify_callbacks(self, timestamp: float) -> None:
        with self._callback_lock:
            callbacks: list[Callable[[float], Any]] = list(self._callbacks)
        for callback in callbacks:
            try:
                callback(timestamp)
            except Exception:
                logger.exception("%s: error in a sync callback", self.name)
