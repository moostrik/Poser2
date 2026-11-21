from threading import Lock
from typing import Protocol


class HasStatus(Protocol):
    """The operator's status line (None: none) and the step the generator's warm-up reached (None: ready)."""
    def get_status(self) -> str | None: ...
    def set_status(self, text: str | None) -> None: ...
    def get_warmup(self) -> int | None: ...
    def set_warmup(self, step: int | None) -> None: ...


class StatusStoreMixin:
    """Thread-safe storage of the status line and the warm-up step."""

    def __init__(self) -> None:
        self._status_lock = Lock()
        self._status: str | None = None
        self._status_warmup: int | None = None

    def get_status(self) -> str | None:
        with self._status_lock:
            return self._status

    def set_status(self, text: str | None) -> None:
        with self._status_lock:
            self._status = text

    def get_warmup(self) -> int | None:
        with self._status_lock:
            return self._status_warmup

    def set_warmup(self, step: int | None) -> None:
        with self._status_lock:
            self._status_warmup = step
