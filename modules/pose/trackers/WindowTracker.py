"""Tracks pose windows for multiple tracks independently."""

from threading import Lock
from traceback import print_exc
from typing import Callable

from .TrackerBase import TrackerBase
from modules.pose.nodes.windows.WindowNode import WindowNode, FeatureWindow
from modules.pose.Frame import FrameDict


# Type alias for window dict callbacks
WindowDictCallback = Callable[[dict[int, FeatureWindow]], None]


class WindowTracker(TrackerBase):
    """Tracks multiple poses, maintaining a separate WindowNode for each.

    Each track_id gets its own WindowNode which maintains an independent buffer.
    Windows are automatically reset when their pose is lost.

    Note: This tracker has TWO callback mechanisms:
    - add_poses_callback: Receives input poses (FrameDict) - for chaining with other pose processors
    - add_window_callback: Receives output windows (dict[int, FeatureWindow]) - for DataHub/visualization
    """

    def __init__(self, num_tracks: int, window_factory: Callable[[], WindowNode]) -> None:
        """Initialize tracker with a window node per track."""
        super().__init__()

        self._window_factory = window_factory
        self._windows: dict[int, WindowNode] = {
            id: window_factory() for id in range(num_tracks)
        }

        # Window output callbacks (separate from pose callbacks)
        self._window_callbacks: set[WindowDictCallback] = set()
        self._window_callback_lock = Lock()

    def process(self, poses: FrameDict) -> dict[int, FeatureWindow]:
        """Process poses through window nodes.

        Returns:
            dict mapping track_id to FeatureWindow (may exclude tracks if emit_partial=False and window not full)
        """

        # Reset windows for poses that are no longer present
        for id in self._windows:
            if id not in poses:
                self.reset_at(id)

        results: dict[int, FeatureWindow] = {}

        for id, pose in poses.items():
            try:
                window = self._windows[id].process(pose)
                if window is not None:  # Only add if window was emitted
                    results[id] = window
            except Exception as e:
                print(f"WindowTracker: Error processing pose {id}: {e}")
                print_exc()

        # Notify pose callbacks (for downstream pose processing)
        self._notify_poses_callbacks(poses)

        # Notify window callbacks (for DataHub/visualization)
        self._notify_window_callbacks(results)

        return results

    def _notify_window_callbacks(self, windows: dict[int, FeatureWindow]) -> None:
        """Emit window callbacks.

        Broadcasts windows to all registered callbacks in a thread-safe manner.

        Args:
            windows: Dictionary of FeatureWindows to broadcast to callbacks.
        """
        with self._window_callback_lock:
            for callback in self._window_callbacks:
                try:
                    callback(windows)
                except Exception as e:
                    print(f"{self.__class__.__name__}: Error in window callback: {e}")
                    print_exc()

    def add_window_callback(self, callback: WindowDictCallback) -> None:
        """Register window output callback.

        Args:
            callback: Function to call with window dictionaries.
        """
        with self._window_callback_lock:
            self._window_callbacks.add(callback)

    def remove_window_callback(self, callback: WindowDictCallback) -> None:
        """Unregister window output callback.

        Args:
            callback: Function to remove from callbacks.
        """
        with self._window_callback_lock:
            self._window_callbacks.discard(callback)

    def reset(self) -> None:
        """Reset all window buffers."""
        for window in self._windows.values():
            window.reset()

    def reset_at(self, id: int) -> None:
        """Reset window buffer for a specific track."""
        self._windows[id].reset()
