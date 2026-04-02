"""Callback mixin for frame dict broadcasting."""

from threading import Lock
from traceback import print_exc

from modules.pose.frame.frame import FrameDict, FrameDictCallback


class PoseDictCallbackMixin:
    """Mixin providing callback management for pose dict broadcasting.

    Provides thread-safe callback registration and emission for components
    that broadcast PoseDict updates. Can be used by trackers, monitors,
    recorders, visualizers, or any component that emits pose dictionaries.

    Usage:
        class MyTracker(PoseDictCallbackMixin):
            def __init__(self):
                super().__init__()

            def process(self, poses: PoseDict):
                # Do processing...
                self._emit_callbacks(poses)
    """

    def __init__(self):
        """Initialize callback system."""
        self._poses_callbacks: set[FrameDictCallback] = set()
        self._poses_callback_lock = Lock()

    def _notify_poses_callbacks(self, poses: FrameDict) -> None:
        """Emit callbacks with poses.

        Broadcasts poses to all registered callbacks in a thread-safe manner.
        Catches and logs exceptions from callbacks to prevent one failing
        callback from affecting others.

        Args:
            poses: Dictionary of poses to broadcast to callbacks.
        """
        with self._poses_callback_lock:
            for callback in self._poses_callbacks:
                try:
                    callback(poses)
                except Exception as e:
                    print(f"{self.__class__.__name__}: Error in callback: {e}")
                    print_exc()

    def add_poses_callback(self, callback: FrameDictCallback) -> None:
        """Register output callback.

        Args:
            callback: Function to call with pose dictionaries.
        """
        with self._poses_callback_lock:
            self._poses_callbacks.add(callback)

    def remove_poses_callback(self, callback: FrameDictCallback) -> None:
        """Unregister output callback.

        Args:
            callback: Function to remove. Safe to call even if not registered.
        """
        with self._poses_callback_lock:
            self._poses_callbacks.discard(callback)
