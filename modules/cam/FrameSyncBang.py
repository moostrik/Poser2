from threading import Lock
from typing import Callable, Optional
import numpy as np
import time

from modules.cam.depthcam.Definitions import FrameType
from modules.Settings import Settings
from modules.utils.HotReloadMethods import HotReloadMethods



class FrameSyncBang:
    """
    Synchronizes frames from multiple cameras and notifies when all frames are received.
    Uses timestamp-based matching with "keep latest frame" strategy for minimal latency.
    """

    def __init__(self, settings: Settings) -> None:
        """
        Initialize the frame synchronizer.

        Args:
            camera_ids: List of camera IDs to synchronize
            frame_type: Type of frames to synchronize (default: VIDEO)
            max_time_diff_ms: Maximum allowed time difference between frames in milliseconds
        """
        self.camera_ids: set[int] = set(range(settings.camera_num))
        self.max_time_diff_ms: float = 1000 / settings.camera_fps
        self._lock = Lock()
        self._current_frames: dict[int, float] = {}  # cam_id -> (frame, timestamp)
        self._callbacks: set[Callable[[], None]] = set()

        self.hot_reloader = HotReloadMethods(self.__class__, True, True)

    def add_frame(self, cam_id: int, frame_type: FrameType, frame: np.ndarray) -> None:
        self.add(cam_id)

    def add(self, cam_id: int) -> None:
        """
        Add a frame from a camera. Keeps only the latest frame per camera.
        Notifies callbacks if all frames are received and within time threshold.

        Args:
            cam_id: Camera ID
            frame_type: Type of frame
            frame: Frame data
            timestamp: Frame timestamp in seconds (uses current time if None)
        """

        if cam_id not in self.camera_ids:
            return

        timestamp = time.time()

        with self._lock:
            # Always replace with latest frame (strategy #4: predictive dropping)
            self._current_frames[cam_id] = (timestamp)

            # Check if all cameras have frames
            if len(self._current_frames) == len(self.camera_ids):
                # Get timestamp range
                timestamps: list[float] = [ts for ts in self._current_frames.values()]
                min_ts: float = min(timestamps)
                max_ts: float = max(timestamps)
                time_diff_ms: float = (max_ts - min_ts) * 1000

                # Only sync if frames are close enough in time
                if time_diff_ms <= self.max_time_diff_ms:
                    print(f"FrameSyncBang: Synchronized frames with time diff {time_diff_ms:.2f} ms, max allowed {self.max_time_diff_ms} ms")
                    self._notify_callbacks()
                    self._current_frames.clear()

    def add_callback(self, callback: Callable[[], None]) -> None:
        """
        Add a callback to be notified when all frames are received.

        Args:
            callback: Function to call when all frames are received
        """
        with self._lock:
            self._callbacks.add(callback)

    def remove_callback(self, callback: Callable[[], None]) -> None:
        """
        Remove a callback.

        Args:
            callback: Callback to remove
        """
        with self._lock:
            self._callbacks.discard(callback)

    def _notify_callbacks(self) -> None:
        """
        Notify all registered callbacks with the synchronized frames.

        Args:
            frames: Dictionary of {cam_id: frame}
        """
        for callback in self._callbacks:
            callback()

    def reset(self) -> None:
        """
        Clear current frames and start fresh synchronization.
        """
        with self._lock:
            self._current_frames.clear()

    def clear_callbacks(self) -> None:
        """
        Remove all callbacks.
        """
        with self._lock:
            self._callbacks.clear()

    def get_sync_stats(self) -> dict[int, float]:
        """
        Get current synchronization statistics.

        Returns:
            Dict with camera IDs and their frame ages in milliseconds
        """
        with self._lock:
            if not self._current_frames:
                return {}

            current_time: float = time.time()
            return {
                cam_id: (current_time - ts) * 1000
                for cam_id, ts in self._current_frames.items()
            }