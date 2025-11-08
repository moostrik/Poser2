"""Callback mixins for pose broadcasting."""

from threading import Lock
from traceback import print_exc

from modules.pose.Pose import Pose, PoseDict
from modules.pose.callback.types import PoseCallback, PoseDictCallback


class PoseCallbackMixin:
    """Mixin providing callback management for single pose broadcasting.

    Provides thread-safe callback registration and emission for components
    that broadcast individual Pose updates.

    Usage:
        class MyFilter(PoseCallbackMixin):
            def __init__(self):
                super().__init__()

            def process(self, pose: Pose):
                # Do processing...
                self._emit_callbacks(pose)
    """

    def __init__(self):
        """Initialize callback system."""
        self._output_callbacks: set[PoseCallback] = set()
        self._callback_lock = Lock()

    def _notify_callbacks(self, pose: Pose) -> None:
        """Emit callbacks with pose.

        Broadcasts pose to all registered callbacks in a thread-safe manner.
        Catches and logs exceptions from callbacks to prevent one failing
        callback from affecting others.

        Args:
            pose: Pose to broadcast to callbacks.
        """
        with self._callback_lock:
            for callback in self._output_callbacks:
                try:
                    callback(pose)
                except Exception as e:
                    print(f"{self.__class__.__name__}: Error in callback: {e}")
                    print_exc()

    def add_callback(self, callback: PoseCallback) -> None:
        """Register output callback.

        Args:
            callback: Function to call with poses.
        """
        with self._callback_lock:
            self._output_callbacks.add(callback)

    def remove_callback(self, callback: PoseCallback) -> None:
        """Unregister output callback.

        Args:
            callback: Function to remove. Safe to call even if not registered.
        """
        with self._callback_lock:
            self._output_callbacks.discard(callback)


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
        self._output_callbacks: set[PoseDictCallback] = set()
        self._callback_lock = Lock()

    def _notify_callbacks(self, poses: PoseDict) -> None:
        """Emit callbacks with poses.

        Broadcasts poses to all registered callbacks in a thread-safe manner.
        Catches and logs exceptions from callbacks to prevent one failing
        callback from affecting others.

        Args:
            poses: Dictionary of poses to broadcast to callbacks.
        """
        with self._callback_lock:
            for callback in self._output_callbacks:
                try:
                    callback(poses)
                except Exception as e:
                    print(f"{self.__class__.__name__}: Error in callback: {e}")
                    print_exc()

    def add_callback(self, callback: PoseDictCallback) -> None:
        """Register output callback.

        Args:
            callback: Function to call with pose dictionaries.
        """
        with self._callback_lock:
            self._output_callbacks.add(callback)

    def remove_callback(self, callback: PoseDictCallback) -> None:
        """Unregister output callback.

        Args:
            callback: Function to remove. Safe to call even if not registered.
        """
        with self._callback_lock:
            self._output_callbacks.discard(callback)