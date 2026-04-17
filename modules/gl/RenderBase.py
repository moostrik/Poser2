from abc import ABC, abstractmethod
from threading import Lock
from typing import Callable

import logging

from OpenGL.GL import * # type: ignore

from modules.gl.WindowManager import WindowManager, WindowSettings

logger = logging.getLogger(__name__)


class RenderBase(ABC):
    def __init__(self, window_settings: WindowSettings) -> None:
        self._update_callbacks: set[Callable] = set()
        self._update_lock = Lock()
        self.window_manager: WindowManager = WindowManager(self, window_settings)

    def add_update_callback(self, callback: Callable) -> None:
        """Register a per-frame update callback."""
        with self._update_lock:
            self._update_callbacks.add(callback)

    def remove_update_callback(self, callback: Callable) -> None:
        """Unregister a per-frame update callback."""
        with self._update_lock:
            self._update_callbacks.discard(callback)

    def _notify_update(self) -> None:
        """Dispatch all registered update callbacks. Call from update()."""
        with self._update_lock:
            for callback in self._update_callbacks:
                try:
                    callback()
                except Exception:
                    logger.exception("Error in update callback")

    def start(self) -> None:
        """Start the render loop."""
        self.window_manager.start()

    def stop(self) -> None:
        """Stop the render loop."""
        self.window_manager.stop()

    def add_exit_callback(self, callback: Callable) -> None:
        """Register a callback invoked when the render window closes."""
        self.window_manager.add_exit_callback(callback)

    @abstractmethod
    def allocate(self) -> None: ...
    @abstractmethod
    def deallocate(self) -> None: ...
    @abstractmethod
    def update(self) -> None: ...
    @abstractmethod
    def draw_main(self, width: int, height: int) -> None: ...
    @abstractmethod
    def draw_secondary(self, monitor_id: int, width: int, height: int) -> None: ...
    @abstractmethod
    def on_main_window_resize(self, width: int, height: int) -> None: ...