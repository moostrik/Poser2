"""Callback mixins for frame-related broadcasting."""

from threading import Lock
from traceback import print_exc

from .frame import Frame, FrameCallback, FrameDict, FrameDictCallback
from .window import FeatureWindowDict, FeatureWindowDictCallback, FrameWindowDict, FrameWindowDictCallback

class FrameCallbackMixin:
    """Mixin providing callback management for single frame broadcasting.

    Provides thread-safe callback registration and emission for components
    that broadcast individual Frame updates.

    Usage:
        class MyFilter(FrameCallbackMixin):
            def __init__(self):
                super().__init__()

            def process(self, frame: Frame):
                # Do processing...
                self._emit_callbacks(frame)
    """

    def __init__(self) -> None:
        """Initialize callback system."""
        self._frame_callbacks: set[FrameCallback] = set()
        self._frame_callback_lock: Lock = Lock()

    def _notify_frame_callbacks(self, frame: Frame) -> None:
        """Emit callbacks with pose.

        Broadcasts pose to all registered callbacks in a thread-safe manner.
        Catches and logs exceptions from callbacks to prevent one failing
        callback from affecting others.

        Args:
            frame: Frame to broadcast to callbacks.
        """
        with self._frame_callback_lock:
            for callback in self._frame_callbacks:
                try:
                    callback(frame)
                except Exception as e:
                    print(f"{self.__class__.__name__}: Error in callback: {e}")
                    print_exc()


    def add_frame_callback(self, callback: FrameCallback) -> None:
        """Register output callback.

        Args:
            callback: Function to call with frames.
        """
        with self._frame_callback_lock:
            self._frame_callbacks.add(callback)

    def remove_frame_callback(self, callback: FrameCallback) -> None:
        """Unregister output callback.

        Args:
            callback: Function to remove. Safe to call even if not registered.
        """
        with self._frame_callback_lock:
            self._frame_callbacks.discard(callback)

class FrameDictCallbackMixin:
    """Mixin providing callback management for frame dict broadcasting.

    Provides thread-safe callback registration and emission for components
    that broadcast FrameDict updates. Can be used by trackers, monitors,
    recorders, visualizers, or any component that emits frame dictionaries.

    Usage:
        class MyTracker(FrameDictCallbackMixin):
            def __init__(self):
                super().__init__()

            def process(self, frames: FrameDict):
                # Do processing...
                self._emit_callbacks(frames)
    """

    def __init__(self) -> None:
        """Initialize callback system."""
        self._frames_callbacks: set[FrameDictCallback] = set()
        self._frames_callback_lock: Lock = Lock()

    def _notify_frames_callbacks(self, frames: FrameDict) -> None:
        """Emit callbacks with frames.

        Broadcasts frames to all registered callbacks in a thread-safe manner.
        Catches and logs exceptions from callbacks to prevent one failing
        callback from affecting others.

        Args:
            frames: Dictionary of frames to broadcast to callbacks.
        """
        with self._frames_callback_lock:
            for callback in self._frames_callbacks:
                try:
                    callback(frames)
                except Exception as e:
                    print(f"{self.__class__.__name__}: Error in callback: {e}")
                    print_exc()

    def add_frames_callback(self, callback: FrameDictCallback) -> None:
        """Register output callback.

        Args:
            callback: Function to call with frame dictionaries.
        """
        with self._frames_callback_lock:
            self._frames_callbacks.add(callback)

    def remove_frames_callback(self, callback: FrameDictCallback) -> None:
        """Unregister output callback.

        Args:
            callback: Function to remove. Safe to call even if not registered.
        """
        with self._frames_callback_lock:
            self._frames_callbacks.discard(callback)


class FeatureWindowDictCallbackMixin:
    """Mixin providing callback management for feature window dict broadcasting.

    Provides thread-safe callback registration and emission for components
    that broadcast FeatureWindowDict updates (single-field windows per track).
    """

    def __init__(self) -> None:
        self._feature_window_callbacks: set[FeatureWindowDictCallback] = set()
        self._feature_window_callback_lock: Lock = Lock()

    def _notify_window_callbacks(self, windows: FeatureWindowDict) -> None:
        with self._feature_window_callback_lock:
            for callback in self._feature_window_callbacks:
                try:
                    callback(windows)
                except Exception as e:
                    print(f"{self.__class__.__name__}: Error in callback: {e}")
                    print_exc()

    def add_windows_callback(self, callback: FeatureWindowDictCallback) -> None:
        with self._feature_window_callback_lock:
            self._feature_window_callbacks.add(callback)

    def remove_windows_callback(self, callback: FeatureWindowDictCallback) -> None:
        with self._feature_window_callback_lock:
            self._feature_window_callbacks.discard(callback)


class FrameWindowDictCallbackMixin:
    """Mixin providing callback management for frame window dict broadcasting.

    Provides thread-safe callback registration and emission for components
    that broadcast FrameWindowDict updates (all fields' windows per track).
    """

    def __init__(self) -> None:
        self._frame_window_callbacks: set[FrameWindowDictCallback] = set()
        self._frame_window_callback_lock: Lock = Lock()

    def _notify_frame_window_callbacks(self, windows: FrameWindowDict) -> None:
        with self._frame_window_callback_lock:
            for callback in self._frame_window_callbacks:
                try:
                    callback(windows)
                except Exception as e:
                    print(f"{self.__class__.__name__}: Error in callback: {e}")
                    print_exc()

    def add_frame_windows_callback(self, callback: FrameWindowDictCallback) -> None:
        with self._frame_window_callback_lock:
            self._frame_window_callbacks.add(callback)

    def remove_frame_windows_callback(self, callback: FrameWindowDictCallback) -> None:
        with self._frame_window_callback_lock:
            self._frame_window_callbacks.discard(callback)
