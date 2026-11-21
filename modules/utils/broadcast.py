"""Composable typed callback fan-out."""

import logging
from threading import Lock
from typing import Any, Callable, Generic, ParamSpec

logger = logging.getLogger(__name__)

P = ParamSpec('P')


class Broadcast(Generic[P]):
    """Thread-safe fan-out that calls the registered callbacks with what it is called with.

    Callbacks run in registration order; registering one twice keeps its first position. The list is copied under its
    lock and the callbacks are called outside it; a failing callback is logged with `name` and the others still run.

    Use as a composable building block — instantiate as an attribute, not as a base class. ``__call__`` makes the
    instance directly usable wherever a single callback is expected.
    """

    def __init__(self, callbacks: list[Callable[P, Any]] | None = None, name: str = "broadcast") -> None:
        self._callbacks: list[Callable[P, Any]] = list(dict.fromkeys(callbacks)) if callbacks else []
        self._lock = Lock()
        self.name: str = name

    def __bool__(self) -> bool:
        """True when any callback is registered — lets producers skip building the payload."""
        with self._lock:
            return bool(self._callbacks)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> None:
        with self._lock:
            callbacks: list[Callable[P, Any]] = list(self._callbacks)
        for callback in callbacks:
            try:
                callback(*args, **kwargs)
            except Exception:
                logger.exception("Error in a %s callback", self.name)

    def add_callback(self, callback: Callable[P, Any]) -> None:
        with self._lock:
            if callback not in self._callbacks:
                self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[P, Any]) -> None:
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)
