"""Generic typed callback mixin for broadcasting any output type."""

from threading import Lock
from typing import Callable, Generic, TypeVar

import logging
logger = logging.getLogger(__name__)

T = TypeVar('T')
Callback = Callable[[T], None]


class TypedCallbackMixin(Generic[T]):
    """Generic mixin providing callback management for broadcasting any output type."""

    def __init__(self):
        self._typed_output_callbacks: set[Callback] = set()
        self._typed_callback_lock = Lock()

    def _notify_callbacks(self, output: T) -> None:
        """Emit callbacks with output of type T."""
        with self._typed_callback_lock:
            for callback in self._typed_output_callbacks:
                try:
                    callback(output)
                except Exception as e:
                    logger.exception("Error in callback")
    def add_callback(self, callback: Callback) -> None:
        """Register output callback."""
        with self._typed_callback_lock:
            self._typed_output_callbacks.add(callback)

    def remove_callback(self, callback: Callback) -> None:
        """Unregister output callback."""
        with self._typed_callback_lock:
            self._typed_output_callbacks.discard(callback)
