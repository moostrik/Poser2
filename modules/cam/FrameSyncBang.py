from threading import Lock
from typing import Callable, Any
import numpy as np
import time
from collections import deque

from modules.cam.depthcam.Definitions import FrameType
from modules.cam.Settings import Settings
from modules.utils.HotReloadMethods import HotReloadMethods


class FrameSyncBang:

    def __init__(self, settings: Settings, verbose: bool = False, stream_name: str = '') -> None:
        num_cams: int = settings.num
        self.verbose: bool = verbose
        self.stream_name: str = stream_name
        self.max_gap_s: float = 1.0 / settings.fps
        self.min_bang_interval_s: float = 0.75 / settings.fps
        self.last_bang_time_s: float = 0.0

        self._timestamp_history: deque[tuple[int, float]] = deque(maxlen=10 * num_cams)
        self._callbacks: set[Callable[[], None]] = set()
        self._lock = Lock()

        self.hot_reloader = HotReloadMethods(self.__class__, True, True)

    def add_frame(self, *args) -> None:
        cam_id: int = args[0]
        timestamp: float = time.time()

        with self._lock:
            self._timestamp_history.append((cam_id, timestamp))
            trigger_cam_id: int = FrameSyncBang._find_sync_trigger_camera(self._timestamp_history, self.max_gap_s)

            if cam_id == trigger_cam_id:
                # if self.verbose:
                #     print(f"{self.stream_name}Max time diff: {FrameSyncBang._get_max_time_diff(self._timestamp_history):.3f}, from cam {cam_id}")

                current_time_s: float = time.time()
                time_since_last_bang: float = current_time_s - self.last_bang_time_s
                self.last_bang_time_s = current_time_s

                if time_since_last_bang >= self.min_bang_interval_s:
                    self._notify_callbacks()
                else:
                    if self.verbose:
                        print(f"{self.stream_name}Skipped bang (only {time_since_last_bang:.3f}s since last, from cam {cam_id})")

    @staticmethod
    def _find_sync_trigger_camera(timestamp_history: deque[tuple[int, float]], max_gap_s: float) -> int:
        """
        Find the camera that should trigger synchronization.
        This is the camera with the largest gap to the next frame, ensuring
        all cameras have their closest possible frame combination when triggered.

        Args:
            timestamp_history: History of (cam_id, timestamp) tuples
            max_gap_s: Maximum gap in seconds to consider (filters out dropped frames)

        Returns:
            Camera ID that should trigger the sync callback
        """

        camera_ids: set[int] = FrameSyncBang._get_unique_cameras_in_history(timestamp_history)

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
        trigger_cam: int = next(iter(camera_ids))
        has_valid_gaps = False

        for cam_id, gaps in cam_gaps.items():
            if gaps:
                has_valid_gaps = True
                avg_gap: float = sum(gaps) / len(gaps)
                if avg_gap > max_avg_gap:
                    max_avg_gap: float = avg_gap
                    trigger_cam = cam_id

        # Fallback: if all gaps filtered, use any camera
        if not has_valid_gaps:
            # print(f"Warning: All frame gaps exceeded {max_gap_s:.3f}s threshold")
            trigger_cam = next(iter(camera_ids))

        return trigger_cam

    @staticmethod
    def _get_max_time_diff(timestamp_history: deque[tuple[int, float]]) -> float:
        """
        Find the maximum time difference between the latest frames of all cameras.

        Returns:
            Maximum time difference in seconds, or 0.0 if not all cameras have frames
        """
        if len(timestamp_history) == 0:
            return 0.0

        camera_ids: set[int] = FrameSyncBang._get_unique_cameras_in_history(timestamp_history)

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

    def add_callback(self, callback: Callable[[], Any]) -> None:
        with self._lock:
            self._callbacks.add(callback)

    def _notify_callbacks(self) -> None:
        for callback in self._callbacks:
            callback()
