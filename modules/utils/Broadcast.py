"""Composable typed callback fan-out."""

from threading import Lock
from typing import Any, Callable, Generic, TypeVar

import logging
logger = logging.getLogger(__name__)

T = TypeVar('T')
Callback = Callable[[T], Any]


class Broadcast(Generic[T]):
    """Thread-safe fan-out that broadcasts a value to registered callbacks.

    Use as a composable building block — instantiate as an attribute,
    not as a base class.  ``__call__`` makes the instance directly
    usable wherever a single callback is expected.

    Args:
        callbacks: Optional initial list of callbacks.
    """

    def __init__(self, callbacks: list[Callback] | None = None) -> None:
        self._callbacks: set[Callback] = set(callbacks) if callbacks else set()
        self._lock = Lock()

    def __call__(self, output: T) -> None:
        with self._lock:
            for callback in self._callbacks:
                try:
                    callback(output)
                except Exception:
                    logger.exception("Error in broadcast callback")

    def add_callback(self, callback: Callback) -> None:
        with self._lock:
            self._callbacks.add(callback)

    def remove_callback(self, callback: Callback) -> None:
        with self._lock:
            self._callbacks.discard(callback)
