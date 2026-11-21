from abc import ABC, abstractmethod
from typing import Any, Callable

from modules.utils import Broadcast
from .window_manager import WindowManager
from .window_settings import WindowManagerSettings


class RenderBase(ABC):
    """A renderer: its update once a frame, its draw in each shown window, on the main thread's window loop."""

    def __init__(self, windows: WindowManagerSettings) -> None:
        """`windows` an app's `windows` group: the window loop's settings and its windows."""
        self._update_callbacks: Broadcast[[float]] = Broadcast(name='update')     # run in registration order
        self.window_manager: WindowManager = WindowManager(self, windows)

    def add_update_callback(self, callback: Callable[[float], Any]) -> None:
        """Register a per-frame update callback; it gets the frame's time (perf_counter, the stamp of the
        frame's update), the clock of everything that advances on the render tick."""
        self._update_callbacks.add_callback(callback)

    def remove_update_callback(self, callback: Callable[[float], Any]) -> None:
        """Unregister a per-frame update callback."""
        self._update_callbacks.remove_callback(callback)

    def _notify_update(self) -> None:
        """Dispatch all registered update callbacks with the frame's time. Call from update()."""
        self._update_callbacks(self.window_manager.frame_timestamp)

    def run(self) -> None:
        """The window loop on the calling thread, the main thread, until the windows close."""
        self.window_manager.run()

    def stop(self) -> None:
        """Ask the window loop to end; from any thread."""
        self.window_manager.stop()

    def add_exit_callback(self, callback: Callable[[], Any]) -> None:
        """Register a callback invoked when the render window closes."""
        self.window_manager.add_exit_callback(callback)

    def window_size(self, window: int) -> tuple[int, int]:
        """A window's framebuffer size (pixels): known from `allocate` on."""
        return self.window_manager.window_size(window)

    @abstractmethod
    def allocate(self) -> None: ...
    @abstractmethod
    def deallocate(self) -> None: ...
    @abstractmethod
    def update(self) -> None: ...
    @abstractmethod
    def draw(self, window: int, width: int, height: int) -> None:
        """Draw window `window`, `width` × `height` pixels, in its own context."""
    @abstractmethod
    def on_window_resize(self, window: int, width: int, height: int) -> None:
        """A window's framebuffer size changed; in window 0's context, where the renderer's objects live."""
