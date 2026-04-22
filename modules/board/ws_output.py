from __future__ import annotations

from threading import Lock
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from modules.WS.WSOutput import WSOutput


class HasWSOutput(Protocol):
    """WS light pipeline output access."""
    def get_ws_output(self) -> WSOutput | None: ...
    def set_ws_output(self, output: WSOutput) -> None: ...


class WSOutputStoreMixin:
    """Thread-safe WS output storage."""

    def __init__(self) -> None:
        self._ws_output_lock = Lock()
        self._ws_output: WSOutput | None = None

    def get_ws_output(self) -> WSOutput | None:
        with self._ws_output_lock:
            return self._ws_output

    def set_ws_output(self, output: WSOutput) -> None:
        with self._ws_output_lock:
            self._ws_output = output
