"""Callback mixins for frame-related broadcasting."""

from modules.utils import Broadcast

from .frame import Frame, FrameCallback, FrameDict, FrameDictCallback
from .window import FrameWindowDict, FrameWindowDictCallback


class FrameCallbackMixin:
    """Mixin providing callback management for single frame broadcasting.

    Provides thread-safe callback registration and emission for components
    that broadcast individual Frame updates. Callbacks run in registration order.

    Usage:
        class MyFilter(FrameCallbackMixin):
            def __init__(self):
                super().__init__()

            def process(self, frame: Frame):
                # Do processing...
                self._notify_frame_callbacks(frame)
    """

    def __init__(self) -> None:
        """Initialize callback system."""
        self._frame_callbacks: Broadcast[Frame] = Broadcast()

    def _notify_frame_callbacks(self, frame: Frame) -> None:
        """Emit callbacks with pose.

        Broadcasts pose to all registered callbacks in a thread-safe manner.
        Catches and logs exceptions from callbacks to prevent one failing
        callback from affecting others.

        Args:
            frame: Frame to broadcast to callbacks.
        """
        self._frame_callbacks(frame)

    def add_frame_callback(self, callback: FrameCallback) -> None:
        """Register output callback.

        Args:
            callback: Function to call with frames.
        """
        self._frame_callbacks.add_callback(callback)

    def remove_frame_callback(self, callback: FrameCallback) -> None:
        """Unregister output callback.

        Args:
            callback: Function to remove. Safe to call even if not registered.
        """
        self._frame_callbacks.remove_callback(callback)


class FrameDictCallbackMixin:
    """Mixin providing callback management for frame dict broadcasting.

    Provides thread-safe callback registration and emission for components
    that broadcast FrameDict updates. Can be used by trackers, monitors,
    recorders, visualizers, or any component that emits frame dictionaries.
    Callbacks run in registration order.

    Usage:
        class MyTracker(FrameDictCallbackMixin):
            def __init__(self):
                super().__init__()

            def process(self, frames: FrameDict):
                # Do processing...
                self._notify_frames_callbacks(frames)
    """

    def __init__(self) -> None:
        """Initialize callback system."""
        self._frames_callbacks: Broadcast[FrameDict] = Broadcast()

    def _notify_frames_callbacks(self, frames: FrameDict) -> None:
        """Emit callbacks with frames.

        Broadcasts frames to all registered callbacks in a thread-safe manner.
        Catches and logs exceptions from callbacks to prevent one failing
        callback from affecting others.

        Args:
            frames: Dictionary of frames to broadcast to callbacks.
        """
        self._frames_callbacks(frames)

    def add_frames_callback(self, callback: FrameDictCallback) -> None:
        """Register output callback.

        Args:
            callback: Function to call with frame dictionaries.
        """
        self._frames_callbacks.add_callback(callback)

    def remove_frames_callback(self, callback: FrameDictCallback) -> None:
        """Unregister output callback.

        Args:
            callback: Function to remove. Safe to call even if not registered.
        """
        self._frames_callbacks.remove_callback(callback)


class FrameWindowDictCallbackMixin:
    """Mixin providing callback management for frame window dict broadcasting.

    Provides thread-safe callback registration and emission for components
    that broadcast FrameWindowDict updates (all fields' windows per track).
    Callbacks run in registration order.
    """

    def __init__(self) -> None:
        self._frame_window_callbacks: Broadcast[FrameWindowDict] = Broadcast()

    def _notify_windows_callbacks(self, windows: FrameWindowDict) -> None:
        self._frame_window_callbacks(windows)

    def add_windows_callback(self, callback: FrameWindowDictCallback) -> None:
        self._frame_window_callbacks.add_callback(callback)

    def remove_windows_callback(self, callback: FrameWindowDictCallback) -> None:
        self._frame_window_callbacks.remove_callback(callback)
